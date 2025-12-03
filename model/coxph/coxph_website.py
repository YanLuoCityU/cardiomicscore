import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

class ModelDataExtractor:
    """
    A class to fit CoxPH models, extract model data for prediction,
    and calculate score percentile distributions.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, logger, penalizer: float = 0.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.logger = logger
        self.penalizer = penalizer

        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        self.scores_data_path = os.path.join(results_dir, 'Scores/OmicsNet/Final')
        
        self.website_save_dir = '/your path/cardiomicscore/saved/results/Website'
        os.makedirs(self.website_save_dir, exist_ok=True)

    def _load_model_data(self, predictor_list: list, outcome: str, data_split: str) -> pd.DataFrame:
        """
        Loads and prepares a merged DataFrame for model fitting.
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

    def calculate_and_save_percentiles(self):
        """
        Calculates and saves the percentile distributions for key scores.
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("Calculating percentile distributions for scores...")
        
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        data_split = 'external_test'
        all_percentiles_data = []

        def _calculate_bin_centers(data_series: pd.Series, bins=100):
            if data_series.empty or data_series.isnull().all():
                return np.full(bins, np.nan)
            bin_edges = np.linspace(data_series.min(), data_series.max(), bins + 1)
            return (bin_edges[:-1] + bin_edges[1:]) / 2

        # 1. Process Townsend score (same for all outcomes)
        try:
            panel_df = pd.read_feather(os.path.join(self.split_seed_path, f"X_{data_split}_PANEL.feather"))
            townsend_centers = _calculate_bin_centers(panel_df['townsend'])
            for outcome in outcomes_to_run:
                row = {'score': 'townsend', 'outcome': outcome}
                row.update({f'p{i+1}': center for i, center in enumerate(townsend_centers)})
                all_percentiles_data.append(row)
            self.logger.info("Successfully calculated percentiles for 'townsend'.")
        except Exception as e:
            self.logger.error(f"Failed to process 'townsend' score: {e}")
        
        # 2. Process PRS (outcome-specific)
        try:
            genomics_df = pd.read_feather(os.path.join(self.split_seed_path, f"X_{data_split}_Genomics.feather"))
            for outcome in outcomes_to_run:
                prs_col = f'{outcome}_prs'
                if prs_col in genomics_df.columns:
                    prs_centers = _calculate_bin_centers(genomics_df[prs_col])
                    row = {'score': 'prs', 'outcome': outcome}
                    row.update({f'p{i+1}': center for i, center in enumerate(prs_centers)})
                    all_percentiles_data.append(row)
            self.logger.info("Successfully calculated percentiles for 'prs'.")
        except Exception as e:
            self.logger.error(f"Failed to process 'prs' scores: {e}")

        # 3. Process MetScore and ProScore (outcome-specific and needs scaling)
        for score_name, predictor_name in [('metscore', 'Metabolomics'), ('proscore', 'Proteomics')]:
            try:
                scores_df = pd.read_csv(os.path.join(self.scores_data_path, f"{data_split}_scores_{predictor_name}.csv"))
                for outcome in outcomes_to_run:
                    if outcome in scores_df.columns:
                        scaler = StandardScaler()
                        # Scale the score first
                        scaled_scores = scaler.fit_transform(scores_df[[outcome]])
                        # Then calculate bin centers on the scaled data
                        score_centers = _calculate_bin_centers(pd.Series(scaled_scores.flatten()))
                        row = {'score': score_name, 'outcome': outcome}
                        row.update({f'p{i+1}': center for i, center in enumerate(score_centers)})
                        all_percentiles_data.append(row)
                self.logger.info(f"Successfully calculated percentiles for '{score_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to process '{score_name}' scores: {e}")

        # Save the final DataFrame
        if all_percentiles_data:
            percentiles_df = pd.DataFrame(all_percentiles_data)
            # Reorder columns to have score, outcome first
            cols = ['score', 'outcome'] + [f'p{i}' for i in range(1, 101)]
            percentiles_df = percentiles_df[cols]
            
            save_path = os.path.join(self.website_save_dir, 'percentiles.csv')
            percentiles_df.to_csv(save_path, index=False)
            self.logger.info(f"Successfully saved percentile data to {save_path}")
        else:
            self.logger.warning("No percentile data was generated.")
        self.logger.info("="*50 + "\n")


    def run_analysis(self):
        """Orchestrates fitting models and extracting all necessary data for prediction."""
        
        # --- Run the new percentile calculation first ---
        self.calculate_and_save_percentiles()
        
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        DATA_SPLIT = 'external_test'
        predictor_combo = ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics']
        
        all_coefficients = {}
        baseline_survival_dfs = []
        
        for outcome in outcomes_to_run:
            self.logger.info(f"\n{'='*20} PROCESSING OUTCOME: {outcome.upper()} {'='*20}")
            
            df_combo = self._load_model_data(predictor_combo, outcome, DATA_SPLIT)
            
            if df_combo.empty:
                self.logger.warning(f"Data for outcome {outcome} is empty. Skipping.")
                continue

            df_combo_no_eid = df_combo.drop(columns=['eid'])
            self.logger.info(f"Fitting model with {len(df_combo_no_eid.columns) - 2} predictors on {len(df_combo_no_eid)} samples.")

            try:
                cph = CoxPHFitter(penalizer=self.penalizer)
                cph.fit(df_combo_no_eid, 'duration', 'event')
                
                all_coefficients[outcome] = cph.params_
                self.logger.info(f"Successfully extracted coefficients for outcome: {outcome}")

                baseline_survival = cph.baseline_survival_
                baseline_survival = baseline_survival.rename(columns={'baseline survival': outcome})
                baseline_survival_dfs.append(baseline_survival)
                self.logger.info(f"Successfully extracted baseline survival for outcome: {outcome}")

            except Exception as e:
                self.logger.error(f"Analysis failed for outcome {outcome}: {e}", exc_info=False)
        
        if all_coefficients:
            coeffs_df = pd.DataFrame(all_coefficients)
            coeffs_df.index.name = 'Variable'
            # --- FILENAME CHANGED AS REQUESTED ---
            coeffs_filename = os.path.join(self.website_save_dir, "coefficients.csv")
            coeffs_df.to_csv(coeffs_filename)
            self.logger.info(f"\nConsolidated coefficients saved to {coeffs_filename}")
            
            if baseline_survival_dfs:
                combined_survival_df = pd.concat(baseline_survival_dfs, axis=1)
                combined_survival_df.index.name = 'Time'
                survival_filename = os.path.join(self.website_save_dir, "baseline_survivals.csv")
                combined_survival_df.to_csv(survival_filename)
                self.logger.info(f"Combined baseline survivals saved to {survival_filename}")
            
            self.logger.info("\n\nSUCCESS: Model data extraction complete.")
        else:
            self.logger.warning("No models were successfully fitted; no model data was saved.")


if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Model_Data_Extraction.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("ModelDataExtractionScript")
    
    extractor = ModelDataExtractor(
        data_dir=DATA_DIR, results_dir=RESULTS_DIR, seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, penalizer=PENALIZER
    )
    extractor.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")