import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from collections import defaultdict
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Bootstrap Calculation Function ---
# This version is simplified to evaluate a SINGLE model, not a comparison.

def run_single_model_bootstrap_iteration(iter_num, df_model, penalizer):
    """
    Logic for a single bootstrap iteration: resample, refit a model, and calculate its C-index.
    """
    try:
        # Resample from the provided data using indices
        indices = resample(df_model.index, random_state=iter_num)
        df_model_boot = df_model.loc[indices]

        if df_model_boot.shape[0] < 50 or df_model_boot['event'].sum() < 5: 
            return None

        # Fit a single model on the bootstrapped data
        features = [c for c in df_model_boot.columns if c not in ['duration', 'event']]
        cph_boot = CoxPHFitter(penalizer=penalizer).fit(df_model_boot, 'duration', 'event')
        
        true_events_boot = df_model_boot['event'].values
        
        # Calculate C-index metric for this iteration (using 1 - c_index as per last learned script)
        c_index = 1 - concordance_index(df_model_boot['duration'], cph_boot.predict_partial_hazard(df_model_boot[features]), true_events_boot)
        
        return {'c_index': c_index}
    except Exception:
        return None

class BootstrapPerformanceEvaluator:
    """
    Independently evaluates performance of omics scores by calculating bootstrapped C-index CIs
    on the train, validation, and internal test sets.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, logger, penalizer: float = 0.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.logger = logger
        self.penalizer = penalizer

        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        self.scores_data_path = os.path.join(self.results_dir, 'Scores/OmicsNet/Final')
        
        self.save_dir = os.path.join(self.results_dir, 'Cindex')
        os.makedirs(self.save_dir, exist_ok=True)

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
            y_path = os.path.join(self.split_seed_path, f"y_{data_split}_{predictor}.feather")
            e_path = os.path.join(self.split_seed_path, f"e_{data_split}_{predictor}.feather")
            y_df = pd.read_feather(y_path)
            e_df = pd.read_feather(e_path)
            
            duration_col, event_col = f'bl2{outcome}_yrs', outcome
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            
            x_path = os.path.join(self.scores_data_path, f"{data_split}_scores_{predictor}.csv")
            scores_df = pd.read_csv(x_path)
            final_name = 'metscore' if predictor == 'Metabolomics' else 'proscore'
            x_df = scores_df[['eid', outcome]].copy()
            scaler = StandardScaler()
            x_df[final_name] = scaler.fit_transform(x_df[[outcome]])
            x_df = x_df.drop(columns=[outcome])

            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
            merged_df = pd.merge(merged_df, x_df, on='eid', how='inner')
            
            self.logger.info(f"Loaded {len(merged_df)} samples for {predictor}.")
            return merged_df

        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Failed to load data for predictor '{predictor}'. Error: {e}")
            return pd.DataFrame()

    def _run_bootstrap(self, df_model) -> dict:
        """
        Manages the parallel execution of bootstrap iterations for a single model.
        """
        n_bootstraps = 1000
        max_workers = 94
        self.logger.info(f"Starting bootstrap with {n_bootstraps} iterations using {max_workers} parallel processes...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_model_bootstrap_iteration, i, df_model, self.penalizer) for i in range(n_bootstraps)]
            
            bootstrap_metrics = defaultdict(list)
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 100 == 0: self.logger.info(f"  Completed {i+1}/{n_bootstraps} bootstrap iterations...")
                result = future.result()
                if result:
                    for metric, value in result.items():
                        bootstrap_metrics[metric].append(value)
        
        ci_results = {}
        if not bootstrap_metrics:
            self.logger.warning("No valid bootstrap results were collected.")
            return ci_results
            
        self.logger.info(f"Collected {len(bootstrap_metrics['c_index'])} valid bootstrap results.")
        for metric, values in bootstrap_metrics.items():
            if len(values) > 20: # Ensure enough samples for stable CI
                ci_results[f'{metric}_mean'] = np.nanmean(values)
                ci_results[f'{metric}_ci_lower'] = np.nanpercentile(values, 2.5)
                ci_results[f'{metric}_ci_upper'] = np.nanpercentile(values, 97.5)
        
        return ci_results

    def run_analysis(self):
        """
        Orchestrates the bootstrap evaluation for all data splits, outcomes, and models.
        """
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        data_splits_to_run = ['train', 'val', 'internal_test']
        predictor_sets = [['Metabolomics'], ['Proteomics']]
        all_results = []
        
        for data_split in data_splits_to_run:
            self.logger.info(f"\n{'#'*20} PROCESSING DATASET: {data_split.upper()} {'#'*20}")
            
            for outcome in outcomes_to_run:
                self.logger.info(f"\n{'='*20} Outcome: {outcome.upper()} | Dataset: {data_split.upper()} {'='*20}")

                for combo in predictor_sets:
                    model_name = combo[0]
                    self.logger.info(f"\n----- Bootstrapping model: {model_name} -----")
                    
                    df_model_data = self._load_model_data(combo, outcome, data_split)
                    if df_model_data.empty: 
                        self.logger.warning(f"Data is empty, skipping evaluation for {model_name} on {data_split}.")
                        continue

                    df_to_bootstrap = df_model_data.drop(columns=['eid']).reset_index(drop=True)
                    
                    # Run the bootstrap process
                    metrics_ci = self._run_bootstrap(df_to_bootstrap)
                    
                    if metrics_ci:
                        metrics_ci['data_split'] = data_split
                        metrics_ci['outcome'] = outcome 
                        metrics_ci['model'] = model_name 
                        metrics_ci['n_samples'] = len(df_to_bootstrap)
                        metrics_ci['n_events'] = int(df_to_bootstrap['event'].sum())
                        all_results.append(metrics_ci)

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.save_dir, "cindex_train_val_internal_test_bootstrap_ci_summary.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Consolidated bootstrap results saved to {results_filename}")
        else:
            self.logger.warning("No results were generated.")

if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Bootstrap_CI_Train_Val_Interal_Test.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("BootstrapPerformanceScript")
    
    evaluator = BootstrapPerformanceEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")