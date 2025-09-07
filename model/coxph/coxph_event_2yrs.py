import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

class EventExclusionEvaluator:
    """
    Performs a point estimate evaluation of CoxPH models after excluding subjects
    with events occurring within the first 2 years of follow-up.
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
        Loads and prepares a merged DataFrame for a given predictor combination,
        excluding samples with events within 2 years.
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
            
            # --- Exclude samples with events within 2 years ---
            outcome_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
            initial_count = len(outcome_df)
            exclusion_condition = (outcome_df['event'] == 1) & (outcome_df['duration'] < 2)
            num_excluded = exclusion_condition.sum()
            
            if num_excluded > 0:
                self.logger.info(f"Excluding {num_excluded} of {initial_count} samples with events occurring within 2 years.")
                outcome_df = outcome_df[~exclusion_condition]

        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Could not load or process base y/e files. Error: {e}")
            return pd.DataFrame()

        # Start merging with the filtered outcome data
        merged_df = outcome_df

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
            self.logger.info(f"Loaded {len(merged_df)} common samples for {predictor_list} after event exclusion.")
        return merged_df

    def run_analysis(self):
        """Orchestrates the point estimate evaluation for the specified PANEL-based combinations."""
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        DATA_SPLIT = 'external_test'
        
        predictor_sets = [
            ['PANEL'], ['PANEL', 'Genomics'], ['PANEL', 'Metabolomics'], ['PANEL', 'Proteomics'],
            ['PANEL', 'Genomics', 'Metabolomics'], ['PANEL', 'Genomics', 'Proteomics'],
            ['PANEL', 'Metabolomics', 'Proteomics'], ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics'],
            ['Genomics'], ['Metabolomics'], ['Proteomics']
        ]
        all_results = []
        
        for outcome in outcomes_to_run:
            self.logger.info(f"\n{'='*20} PROCESSING OUTCOME: {outcome.upper()} {'='*20}")

            # Baseline is always the PANEL model
            baseline_combo = ['PANEL']
            df_baseline = self._load_model_data(baseline_combo, outcome, DATA_SPLIT)

            if df_baseline.empty:
                self.logger.warning(f"Baseline data for '{baseline_combo}' is empty for outcome {outcome}. Skipping.")
                continue

            for combo in predictor_sets:
                self.logger.info(f"\n----- Evaluating Combo: {combo} vs. Baseline: {baseline_combo} -----")
                
                df_combo = self._load_model_data(combo, outcome, DATA_SPLIT)
                if df_combo.empty: continue

                common_eids = pd.merge(df_baseline[['eid']], df_combo[['eid']], on='eid', how='inner')['eid']
                if len(common_eids) < 100:
                    self.logger.warning(f"Skipping {combo} due to small common sample size ({len(common_eids)}).")
                    continue

                df_base_common = df_baseline[df_baseline['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                df_combo_common = df_combo[df_combo['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                self.logger.info(f"Using {len(common_eids)} common samples for comparison.")
                
                try:
                    # Fit baseline model
                    base_features = [c for c in df_base_common.columns if c not in ['duration', 'event']]
                    cph_base = CoxPHFitter(penalizer=self.penalizer).fit(df_base_common, 'duration', 'event')
                    c_index_base = 1 - concordance_index(df_base_common['duration'], cph_base.predict_partial_hazard(df_base_common[base_features]), df_base_common['event'])

                    # Fit combination model
                    combo_features = [c for c in df_combo_common.columns if c not in ['duration', 'event']]
                    cph_combo = CoxPHFitter(penalizer=self.penalizer).fit(df_combo_common, 'duration', 'event')
                    c_index_combo = 1 - concordance_index(df_combo_common['duration'], cph_combo.predict_partial_hazard(df_combo_common[combo_features]), df_combo_common['event'])
                    
                    # Store C-Index results
                    all_results.append({
                        'outcome': outcome,
                        'baseline_model': '_'.join(baseline_combo),
                        'comparison_model': '_'.join(combo),
                        'n_samples': len(common_eids),
                        'c_index_base': c_index_base,
                        'c_index_combo': c_index_combo,
                        'delta_c_index': c_index_combo - c_index_base
                    })
                    self.logger.info(f"C-Index for {combo}: {c_index_combo:.4f} (Delta vs. {baseline_combo}: {c_index_combo - c_index_base:+.4f})")

                except Exception as e:
                    self.logger.error(f"Analysis failed for combo {combo}, outcome {outcome}: {e}", exc_info=False)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.cindex_save_dir, "cindex_event_2yrs_summary.csv")
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

    log_filename = os.path.join(LOG_DIR, "CoxPH/Point_Estimates_Event_2yrs.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("EventExclusionScript")
    
    evaluator = EventExclusionEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")