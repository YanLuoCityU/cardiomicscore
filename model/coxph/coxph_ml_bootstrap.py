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
# This is a top-level function to be serializable for parallel processing.

def run_bootstrap_iteration(iter_num, df_base, df_combo, penalizer):
    """
    Logic for a single bootstrap iteration: resample, refit models, and calculate C-index metrics.
    """
    try:
        # Resample from the provided data using indices
        indices = resample(df_base.index, random_state=iter_num)
        df_base_boot = df_base.loc[indices]
        df_combo_boot = df_combo.loc[indices]

        if df_base_boot.shape[0] < 50 or df_base_boot['event'].sum() < 5: return None

        # Fit models on the bootstrapped data
        base_features = [c for c in df_base_boot.columns if c not in ['duration', 'event']]
        combo_features = [c for c in df_combo_boot.columns if c not in ['duration', 'event']]

        cph_base_boot = CoxPHFitter(penalizer=penalizer).fit(df_base_boot, 'duration', 'event')
        cph_combo_boot = CoxPHFitter(penalizer=penalizer).fit(df_combo_boot, 'duration', 'event')
        
        true_events_boot = df_base_boot['event'].values
        
        # Calculate C-index metrics for this iteration (using 1 - c_index as per last learned script)
        iteration_metrics = {}
        c_base_b = 1 - concordance_index(df_base_boot['duration'], cph_base_boot.predict_partial_hazard(df_base_boot[base_features]), true_events_boot)
        c_combo_b = 1 - concordance_index(df_combo_boot['duration'], cph_combo_boot.predict_partial_hazard(df_combo_boot[combo_features]), true_events_boot)
        
        iteration_metrics['c_index_base'] = c_base_b
        iteration_metrics['c_index_combo'] = c_combo_b
        iteration_metrics['delta_c_index'] = c_combo_b - c_base_b
        
        return iteration_metrics
    except Exception:
        return None

class MLBootstrapEvaluator:
    """
    Evaluates CoxPH models using pre-computed scores from various ML models as predictors,
    calculating bootstrapped confidence intervals for C-Index for each ML model's scores.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, model_name: str, logger, penalizer: float = 0.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.model_name = model_name  # e.g., 'LGBM', 'XGBoost'
        self.logger = logger
        self.penalizer = penalizer

        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        if self.model_name == 'OmicsNet_Unweighted':
            self.scores_data_path = os.path.join(self.results_dir, 'Scores', self.model_name, 'Final')
            self.logger.info(f"Using special scores path for {self.model_name}: {self.scores_data_path}")
        else:
            self.scores_data_path = os.path.join(self.results_dir, 'Scores', self.model_name)
        
        self.save_dir = os.path.join(self.results_dir, 'Cindex')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _load_model_data(self, predictor_list: list, outcome: str, data_split: str) -> pd.DataFrame:
        """
        Loads and prepares a merged DataFrame using the modern data loading flow.
        """
        self.logger.info(f"Loading '{data_split}' data for {predictor_list} (Outcome: {outcome})")
        if not predictor_list: return pd.DataFrame()
        try:
            y_path = os.path.join(self.split_seed_path, f"y_{data_split}.feather")
            e_path = os.path.join(self.split_seed_path, f"e_{data_split}.feather")
            y_df, e_df = pd.read_feather(y_path), pd.read_feather(e_path)
            duration_col, event_col = f'bl2{outcome}_yrs', outcome
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Could not load base y/e files. Error: {e}")
            return pd.DataFrame()
        for predictor in predictor_list:
            x_df, path = pd.DataFrame(), "N/A"
            try:
                if predictor in ['PANEL', 'Genomics']:
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_{predictor}.feather")
                    if predictor == 'Genomics':
                        x_df = pd.read_feather(path)[['eid', f'{outcome}_prs']].rename(columns={f'{outcome}_prs': 'prs'})
                    else:
                        x_df = pd.read_feather(path)
                elif predictor in ['Metabolomics', 'Proteomics']:
                    path = os.path.join(self.scores_data_path, f"{data_split}_scores_{predictor}.csv")
                    scores_df = pd.read_csv(path)
                    final_name = 'metscore' if predictor == 'Metabolomics' else 'proscore'
                    x_df = scores_df[['eid', outcome]].copy()
                    scaler = StandardScaler()
                    x_df[final_name] = scaler.fit_transform(x_df[[outcome]])
                    x_df = x_df.drop(columns=[outcome])
                else: continue
                if not x_df.empty: merged_df = pd.merge(merged_df, x_df, on='eid', how='inner')
                else: return pd.DataFrame()
            except (FileNotFoundError, KeyError) as e:
                self.logger.error(f"Failed to load data for '{predictor}'. Path: {path}. Error: {e}")
                return pd.DataFrame()
        return merged_df

    def _run_bootstrap_on_test_set(self, df_base, df_combo) -> dict:
        """
        Manages the parallel execution of bootstrap iterations.
        """
        n_bootstraps = 1000
        max_workers = 95
        self.logger.info(f"Starting bootstrap with {n_bootstraps} iterations using {max_workers} parallel processes...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_bootstrap_iteration, i, df_base, df_combo, self.penalizer) for i in range(n_bootstraps)]
            
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
            
        self.logger.info(f"Collected {len(bootstrap_metrics['delta_c_index'])} valid bootstrap results.")
        for metric, values in bootstrap_metrics.items():
            if len(values) > 20: # Ensure enough samples for stable CI
                ci_results[f'{metric}_mean'] = np.nanmean(values)
                ci_results[f'{metric}_ci_lower'] = np.nanpercentile(values, 2.5)
                ci_results[f'{metric}_ci_upper'] = np.nanpercentile(values, 97.5)
        
        return ci_results

    def run_analysis(self):
        """
        Orchestrates the bootstrap evaluation for all combinations and outcomes.
        """
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
            self.logger.info(f"\n{'#'*20} PROCESSING OUTCOME: {outcome.upper()} {'#'*20}")
            outcome_start_time = time.time()

            baseline_combo = ['PANEL']
            df_baseline_full = self._load_model_data(baseline_combo, outcome, DATA_SPLIT)

            if df_baseline_full is None or df_baseline_full.empty:
                self.logger.warning(f"Baseline data for {baseline_combo} is empty. Skipping outcome {outcome}.")
                continue

            total_predictor_sets = len(predictor_sets)
            for i, combo in enumerate(predictor_sets):
                self.logger.info(f"\n----- [Outcome: {outcome.upper()} | Set: {i+1}/{total_predictor_sets}] -----")
                self.logger.info(f"Bootstrapping Combo: {combo} vs. Base: {baseline_combo}")

                df_combo_full = self._load_model_data(combo, outcome, DATA_SPLIT)
                if df_combo_full.empty: continue

                common_eids = pd.merge(df_baseline_full[['eid']], df_combo_full[['eid']], on='eid', how='inner')['eid']
                if len(common_eids) < 100:
                    self.logger.warning(f"Skipping combo {combo} due to small sample size ({len(common_eids)}).")
                    continue
                
                df_base_common = df_baseline_full[df_baseline_full['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                df_combo_common = df_combo_full[df_combo_full['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                metrics_ci = self._run_bootstrap_on_test_set(df_base_common, df_combo_common)
                
                if metrics_ci:
                    metrics_ci['outcome'] = outcome
                    metrics_ci['baseline_model'] = '_'.join(baseline_combo)
                    metrics_ci['comparison_model'] = '_'.join(combo)
                    metrics_ci['n_samples'] = len(common_eids)
                    all_results.append(metrics_ci)

            outcome_end_time = time.time()
            outcome_elapsed_time = outcome_end_time - outcome_start_time
            self.logger.info(f"\n{'#'*20} COMPLETED OUTCOME: {outcome.upper()} {'#'*20}")
            self.logger.info(f"Time taken for this outcome: {outcome_elapsed_time / 60:.2f} minutes.")

        if all_results:
            results_df = pd.DataFrame(all_results)
            cols_to_drop = [
                'c_index_base_ci_lower',
                'c_index_base_ci_upper',
                'c_index_base_mean'
            ]
            results_df = results_df.drop(columns=cols_to_drop, errors='ignore')
            id_cols = ['outcome', 'baseline_model', 'comparison_model', 'n_samples']
            metric_cols = sorted([col for col in results_df.columns if col not in id_cols])
            results_df = results_df[id_cols + metric_cols]
            
            filename_map = {
                'LGBM': 'lgbm',
                'XGBoost': 'xgb',
                'RandomForest': 'rf',
                'LogisticRegression': 'glm',
                'OmicsNet_Unweighted': 'unweighted'
            }
            file_suffix = filename_map.get(self.model_name, self.model_name.lower())
            results_filename = os.path.join(self.save_dir, f"cindex_{file_suffix}_bootstrap_ci_summary.csv")
            
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Bootstrap CI results for '{self.model_name}' saved to {results_filename}")


if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    score_models_to_run = [
        'LGBM',
        'XGBoost',
        'RandomForest',
        'LogisticRegression',
        'OmicsNet_Unweighted'
    ]

    for model_name in score_models_to_run:
        print(f"\n{'#'*30}\n# Running Bootstrap Analysis for ML Model: {model_name}\n{'#'*30}")

        log_suffix_map = {
            'LGBM': 'LGBM',
            'XGBoost': 'XGBoost',
            'RandomForest': 'RandomForest',
            'LogisticRegression': 'LogisticRegression',
            'OmicsNet_Unweighted': 'Unweighted'
        }
        log_suffix = log_suffix_map.get(model_name, model_name)
        log_filename = os.path.join(LOG_DIR, f"CoxPH/Bootstrap_CI_{log_suffix}.log")
        
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        logger_main = logging.getLogger(f"BootstrapScript_{model_name}")
        logger_main.setLevel(logging.INFO)
        if logger_main.hasHandlers(): logger_main.handlers.clear()
            
        file_handler = logging.FileHandler(log_filename, mode='w')
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger_main.addHandler(file_handler)
        logger_main.addHandler(stream_handler)
        
        evaluator = MLBootstrapEvaluator(
            data_dir=DATA_DIR, 
            results_dir=RESULTS_DIR, 
            seed_to_split=SEED_TO_SPLIT,
            model_name=model_name,
            logger=logger_main, 
            penalizer=PENALIZER
        )
        evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    print(f"\n--- Total Script Execution Time for all models: {total_elapsed_time / 60:.2f} minutes ---")