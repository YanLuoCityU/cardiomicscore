import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

class PointEstimateEvaluator:
    """
    A class to perform point estimate evaluation of CoxPH models for various predictor combinations.
    It calculates C-index and saves incident risk probabilities on a test set.
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
        
        self.prob_save_dir = os.path.join(self.results_dir, 'Incident_Probabilities')
        os.makedirs(self.prob_save_dir, exist_ok=True)

    def _load_model_data(self, predictor_list: list, outcome: str, data_split: str) -> pd.DataFrame:
        """
        Loads and prepares a merged DataFrame for a given predictor combination and data split.
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
            if duration_col not in y_df.columns or event_col not in e_df.columns:
                self.logger.error(f"Outcome columns '{duration_col}' or '{event_col}' not found.")
                return pd.DataFrame()
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
        except FileNotFoundError as e:
            self.logger.error(f"Could not load base y/e files. Error: {e}")
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
        """Orchestrates point estimate evaluation for all combinations and outcomes."""
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        DATA_SPLIT = 'external_test'
        predictor_sets = [
            ['AgeSex'], ['AgeSex', 'Genomics'], ['AgeSex', 'Metabolomics'], ['AgeSex', 'Proteomics'],
            ['AgeSex', 'Genomics', 'Metabolomics'], ['AgeSex', 'Genomics', 'Proteomics'],
            ['AgeSex', 'Metabolomics', 'Proteomics'], ['AgeSex', 'Genomics', 'Metabolomics', 'Proteomics'],
            ['Clinical'], ['Clinical', 'Genomics'], ['Clinical', 'Metabolomics'], ['Clinical', 'Proteomics'],
            ['Clinical', 'Genomics', 'Metabolomics'], ['Clinical', 'Genomics', 'Proteomics'],
            ['Clinical', 'Metabolomics', 'Proteomics'], ['Clinical', 'Genomics', 'Metabolomics', 'Proteomics'],
            ['PANEL'], ['PANEL', 'Genomics'], ['PANEL', 'Metabolomics'], ['PANEL', 'Proteomics'],
            ['PANEL', 'Genomics', 'Metabolomics'], ['PANEL', 'Genomics', 'Proteomics'],
            ['PANEL', 'Metabolomics', 'Proteomics'], ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics'],
            ['Genomics'], ['Metabolomics'], ['Proteomics']
        ]
        all_results = []
        
        for outcome in outcomes_to_run:
            self.logger.info(f"\n{'='*20} PROCESSING OUTCOME: {outcome.upper()} {'='*20}")
            
            outcome_prob_dir = os.path.join(self.prob_save_dir, outcome)
            os.makedirs(outcome_prob_dir, exist_ok=True)

            baseline_dataframes = {}
            for base in [['AgeSex'], ['Clinical'], ['PANEL']]:
                self.logger.info(f"Pre-loading baseline data for: {base}")
                baseline_dataframes['+'.join(base)] = self._load_model_data(base, outcome, DATA_SPLIT)

            for combo in predictor_sets:
                if 'PANEL' in combo: baseline_name = 'PANEL'
                elif 'Clinical' in combo: baseline_name = 'Clinical'
                else: baseline_name = 'AgeSex'
                
                baseline_combo = [baseline_name]
                df_baseline = baseline_dataframes[baseline_name]

                self.logger.info(f"\n----- Evaluating Combo: {combo} vs. Baseline: {baseline_combo} -----")
                if df_baseline.empty:
                    self.logger.error(f"Baseline data '{baseline_name}' is empty. Skipping.")
                    continue
                
                df_combo = self._load_model_data(combo, outcome, DATA_SPLIT)
                if df_combo.empty: continue

                common_eids = pd.merge(df_baseline[['eid']], df_combo[['eid']], on='eid', how='inner')['eid']
                
                # Keep eid for saving probabilities later
                df_base_common_w_eid = df_baseline[df_baseline['eid'].isin(common_eids)].reset_index(drop=True)
                df_combo_common_w_eid = df_combo[df_combo['eid'].isin(common_eids)].reset_index(drop=True)
                
                # Drop eid for model fitting
                df_base_common = df_base_common_w_eid.drop(columns=['eid'])
                df_combo_common = df_combo_common_w_eid.drop(columns=['eid'])

                self.logger.info(f"Using {len(common_eids)} common samples for comparison.")
                if len(common_eids) == 0: continue

                try:
                    base_features = [c for c in df_base_common.columns if c not in ['duration', 'event']]
                    cph_base = CoxPHFitter(penalizer=self.penalizer).fit(df_base_common, 'duration', 'event')
                    c_index_base = 1 - concordance_index(df_base_common['duration'], cph_base.predict_partial_hazard(df_base_common[base_features]), df_base_common['event'])

                    combo_features = [c for c in df_combo_common.columns if c not in ['duration', 'event']]
                    cph_combo = CoxPHFitter(penalizer=self.penalizer).fit(df_combo_common, 'duration', 'event')
                    c_index_combo = 1 - concordance_index(df_combo_common['duration'], cph_combo.predict_partial_hazard(df_combo_common[combo_features]), df_combo_common['event'])
                    
                    # --- Save incident probabilities for the combo model ---
                    risk_horizons = [5, 10, 15]
                    combo_risk = 1 - cph_combo.predict_survival_function(df_combo_common[combo_features], times=risk_horizons).T
                    
                    prob_df = df_combo_common_w_eid[['eid', 'event', 'duration']].copy()
                    prob_df['risk_5y'] = combo_risk[5].values
                    prob_df['risk_10y'] = combo_risk[10].values
                    prob_df['risk_15y'] = combo_risk[15].values
                    
                    prob_filename = os.path.join(outcome_prob_dir, f"incident_prob_{'_'.join(combo)}.csv")
                    prob_df.to_csv(prob_filename, index=False)
                    self.logger.info(f"Saved incident probabilities for {combo} to {prob_filename}")

                    # --- Store C-Index results ---
                    all_results.append({
                        'outcome': outcome, 'baseline_model': '_'.join(baseline_combo),
                        'comparison_model': '_'.join(combo), 'n_samples': len(common_eids),
                        'c_index_base': c_index_base, 'c_index_combo': c_index_combo,
                        'delta_c_index': c_index_combo - c_index_base
                    })
                    self.logger.info(f"C-Index for {combo}: {c_index_combo:.4f} (Delta vs. {baseline_name}: {c_index_combo - c_index_base:+.4f})")

                except Exception as e:
                    self.logger.error(f"Analysis failed for combo {combo}, outcome {outcome}: {e}", exc_info=False)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.cindex_save_dir, "cindex_summary.csv")
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

    log_filename = os.path.join(LOG_DIR, "CoxPH/Point_Estimates.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("PointEstimateScript")
    
    evaluator = PointEstimateEvaluator(
        data_dir=DATA_DIR, results_dir=RESULTS_DIR, seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")