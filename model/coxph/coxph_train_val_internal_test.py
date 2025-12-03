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
    Independently evaluates the performance of Metabolomics and Proteomics scores
    in their respective cohorts using a Cox model on the train, validation, and internal test sets.
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
        Loads the independent X, y, and e datasets for a single predictor (Metabolomics or Proteomics).
        """
        if len(predictor_list) != 1:
            self.logger.error(f"This script only supports single predictor evaluation, but received: {predictor_list}")
            return pd.DataFrame()
        
        predictor = predictor_list[0]
        self.logger.info(f"Loading '{data_split}' dataset for predictor: {predictor} (Outcome: {outcome})")

        try:
            # y and e paths now include the {predictor}
            y_path = os.path.join(self.split_seed_path, f"y_{data_split}_{predictor}.feather")
            e_path = os.path.join(self.split_seed_path, f"e_{data_split}_{predictor}.feather")
            y_df = pd.read_feather(y_path)
            e_df = pd.read_feather(e_path)
            
            duration_col, event_col = f'bl2{outcome}_yrs', outcome
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            
            # Load X (scores)
            x_path = os.path.join(self.scores_data_path, f"{data_split}_scores_{predictor}.csv")
            scores_df = pd.read_csv(x_path)
            final_name = 'metscore' if predictor == 'Metabolomics' else 'proscore'
            x_df = scores_df[['eid', outcome]].copy()
            scaler = StandardScaler()
            x_df[final_name] = scaler.fit_transform(x_df[[outcome]])
            x_df = x_df.drop(columns=[outcome])

            # Merge the independent X, y, e data
            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
            merged_df = pd.merge(merged_df, x_df, on='eid', how='inner')
            
            self.logger.info(f"Loaded {len(merged_df)} samples for {predictor}.")
            return merged_df

        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Failed to load data for predictor '{predictor}'. Error: {e}")
            return pd.DataFrame()

    def run_analysis(self):
        """
        Orchestrates the point estimate evaluation for all data splits, outcomes, and model combinations.
        """
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        # Evaluate on train, val, and internal_test
        data_splits_to_run = ['train', 'val', 'internal_test']
        
        # Predictor sets only include Metabolomics and Proteomics
        predictor_sets = [['Metabolomics'], ['Proteomics']]
        all_results = []
        
        # Outermost loop iterates through different datasets
        for data_split in data_splits_to_run:
            self.logger.info(f"\n{'#'*20} PROCESSING DATASET: {data_split.upper()} {'#'*20}")
            
            for outcome in outcomes_to_run:
                self.logger.info(f"\n{'='*20} Outcome: {outcome.upper()} | Dataset: {data_split.upper()} {'='*20}")

                for combo in predictor_sets:
                    model_name = combo[0]
                    self.logger.info(f"\n----- Evaluating model: {model_name} -----")
                    
                    df_model_data = self._load_model_data(combo, outcome, data_split)
                    if df_model_data.empty: 
                        self.logger.warning(f"Data is empty, skipping evaluation for {model_name} on {data_split}.")
                        continue

                    df_model_data = df_model_data.drop(columns=['eid']).reset_index(drop=True)
                    
                    try:
                        features = [c for c in df_model_data.columns if c not in ['duration', 'event']]
                        
                        # Fit and evaluate the model on the current dataset
                        cph = CoxPHFitter(penalizer=self.penalizer).fit(df_model_data, 'duration', 'event')
                        c_index = 1 - concordance_index(df_model_data['duration'], cph.predict_partial_hazard(df_model_data[features]), df_model_data['event'])
                        
                        all_results.append({
                            'data_split': data_split,
                            'outcome': outcome, 
                            'model': model_name, 
                            'n_samples': len(df_model_data),
                            'n_events': int(df_model_data['event'].sum()),
                            'c_index': c_index
                        })
                        self.logger.info(f"C-Index for model {model_name} on {data_split}: {c_index:.4f}")

                    except Exception as e:
                        self.logger.error(f"Analysis failed for model {model_name}, outcome {outcome}, dataset {data_split}: {e}", exc_info=False)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.cindex_save_dir, "cindex_train_val_internal_test_summary.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Consolidated results saved to {results_filename}")
        else:
            self.logger.warning("No results were generated.")

if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Point_Estimates_Train_Val_Interal_Test.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("PointEstimateScript_Omics")
    
    evaluator = PointEstimateEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")