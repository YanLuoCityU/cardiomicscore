import os
import time
import pandas as pd
import numpy as np
import itertools
import logging
import sys
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

class MultiOmicsComparisonEvaluator:
    """
    Compares single-omics models against multi-omics models by evaluating the incremental
    improvement in C-index on the external test set.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, logger, penalizer: float = 0.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.logger = logger
        self.penalizer = penalizer

        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        self.scores_data_path = os.path.join(self.results_dir, 'Scores/OmicsNet/Final')
        
        self.cindex_save_dir = os.path.join(self.results_dir, 'Cindex')
        os.makedirs(self.cindex_save_dir, exist_ok=True)

    def _load_model_data(self, predictor_list: list, outcome: str, data_split: str) -> pd.DataFrame:
        """
        Loads and prepares a merged DataFrame for a given predictor combination and data split,
        using the modern data loading flow.
        """
        self.logger.info(f"Loading '{data_split}' data for combination: {predictor_list} and outcome: {outcome}")
        if not predictor_list:
            self.logger.error("Predictor list cannot be empty.")
            return pd.DataFrame()

        try:
            y_path = os.path.join(self.split_seed_path, f"y_{data_split}.feather")
            e_path = os.path.join(self.split_seed_path, f"e_{data_split}.feather")
            y_df = pd.read_feather(y_path)
            e_df = pd.read_feather(e_path)
            duration_col, event_col = f'bl2{outcome}_yrs', outcome
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Could not load or process base y/e files. Error: {e}")
            return pd.DataFrame()

        for predictor in predictor_list:
            x_df = pd.DataFrame()
            path = "N/A"
            try:
                if predictor in ['AgeSex', 'Clinical', 'PANEL']:
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_{predictor}.feather")
                    x_df = pd.read_feather(path)
                elif predictor == 'Genomics':
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_Genomics.feather")
                    genomics_df = pd.read_feather(path)
                    prs_col = f'{outcome}_prs'
                    x_df = genomics_df[['eid', prs_col]].rename(columns={prs_col: 'prs'})
                elif predictor in ['Metabolomics', 'Proteomics']:
                    path = os.path.join(self.scores_data_path, f"{data_split}_scores_{predictor}.csv")
                    scores_df = pd.read_csv(path)
                    final_name = 'metscore' if predictor == 'Metabolomics' else 'proscore'
                    x_df = scores_df[['eid', outcome]].copy()
                    scaler = StandardScaler()
                    x_df[final_name] = scaler.fit_transform(x_df[[outcome]])
                    x_df = x_df.drop(columns=[outcome])
                else:
                    self.logger.warning(f"Predictor '{predictor}' has no defined loading logic. Skipping.")
                    continue
                if not x_df.empty:
                    merged_df = pd.merge(merged_df, x_df, on='eid', how='inner')
                else: return pd.DataFrame()
            except (FileNotFoundError, KeyError) as e:
                self.logger.error(f"Failed to load data for predictor '{predictor}'. Path: {path}. Error: {e}")
                return pd.DataFrame()
        
        if not merged_df.empty:
            self.logger.info(f"Loaded {len(merged_df)} common samples for {predictor_list}.")
        return merged_df

    def run_analysis(self):
        """Orchestrates the point estimate evaluation for all combinations and outcomes."""
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        DATA_SPLIT = 'external_test'
        
        baseline_combinations = [
            ['PANEL', 'Genomics'],
            ['PANEL', 'Metabolomics'],
            ['PANEL', 'Proteomics']
        ]
        omics_predictors = ['Genomics', 'Metabolomics', 'Proteomics']
        custom_sort_order = ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics']
        
        all_results = []
        
        for outcome in outcomes_to_run:
            self.logger.info(f"\n{'#'*20} PROCESSING OUTCOME: {outcome.upper()} {'#'*20}")

            for base_combo in baseline_combinations:
                self.logger.info(f"\n{'='*20} BASELINE MODEL: {base_combo} {'='*20}")
                
                df_base = self._load_model_data(base_combo, outcome, DATA_SPLIT)
                if df_base.empty:
                    self.logger.warning(f"Data for baseline model {base_combo} is empty. Skipping.")
                    continue

                base_omics = set(p for p in base_combo if p in omics_predictors)
                remaining_omics = [p for p in omics_predictors if p not in base_omics]
                
                for i in range(1, len(remaining_omics) + 1):
                    for subset in itertools.combinations(remaining_omics, i):
                        unsorted_combo = list(set(base_combo + list(subset)))
                        comparison_combo = sorted(unsorted_combo, key=lambda p: custom_sort_order.index(p))
                        
                        self.logger.info(f"\n----- Evaluating: {comparison_combo} vs. {base_combo} -----")
                        
                        df_comparison = self._load_model_data(comparison_combo, outcome, DATA_SPLIT)
                        if df_comparison.empty: continue

                        common_eids = pd.merge(df_base[['eid']], df_comparison[['eid']], on='eid', how='inner')['eid']
                        if len(common_eids) < 100:
                            self.logger.warning(f"Skipping due to small common sample size ({len(common_eids)}).")
                            continue
                            
                        df_base_common = df_base[df_base['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                        df_comparison_common = df_comparison[df_comparison['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                        try:
                            base_features = [c for c in df_base_common.columns if c not in ['duration', 'event']]
                            cph_base = CoxPHFitter(penalizer=self.penalizer).fit(df_base_common, 'duration', 'event')
                            c_index_base = 1 - concordance_index(df_base_common['duration'], cph_base.predict_partial_hazard(df_base_common[base_features]), df_base_common['event'])

                            comparison_features = [c for c in df_comparison_common.columns if c not in ['duration', 'event']]
                            cph_comparison = CoxPHFitter(penalizer=self.penalizer).fit(df_comparison_common, 'duration', 'event')
                            c_index_comparison = 1 - concordance_index(df_comparison_common['duration'], cph_comparison.predict_partial_hazard(df_comparison_common[comparison_features]), df_comparison_common['event'])
                            
                            all_results.append({
                                'outcome': outcome,
                                'baseline_model': '_'.join(sorted(base_combo, key=lambda p: custom_sort_order.index(p))),
                                'comparison_model': '_'.join(comparison_combo),
                                'n_samples': len(common_eids),
                                'c_index_base': c_index_base,
                                'c_index_combo': c_index_comparison,
                                'delta_c_index': c_index_comparison - c_index_base
                            })
                            self.logger.info(f"C-Index for {comparison_combo}: {c_index_comparison:.4f} (Delta vs. {base_combo}: {c_index_comparison - c_index_base:+.4f})")

                        except Exception as e:
                            self.logger.error(f"Analysis failed for {comparison_combo} vs {base_combo}: {e}", exc_info=False)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.cindex_save_dir, "cindex_single_vs_multi_omics_summary.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Consolidated C-Index results saved to {results_filename}")
        else:
            self.logger.warning("No results were generated.")


if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Point_Estimates_Single_vs_Multi_Omics.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("MultiOmicsComparisonScript")
    
    evaluator = MultiOmicsComparisonEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")