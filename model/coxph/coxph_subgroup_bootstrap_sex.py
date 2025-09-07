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
def run_bootstrap_iteration(iter_num, df_base, df_combo, penalizer, cols_to_drop):
    """
    Logic for a single bootstrap iteration: resample, refit models (excluding subgroup cols),
    and calculate C-index metrics.
    """
    try:
        indices = resample(df_base.index, random_state=iter_num)
        df_base_boot = df_base.loc[indices]
        df_combo_boot = df_combo.loc[indices]

        if df_base_boot.shape[0] < 50 or df_base_boot['event'].sum() < 5: return None

        base_features = [c for c in df_base_boot.columns if c not in ['duration', 'event'] + cols_to_drop]
        combo_features = [c for c in df_combo_boot.columns if c not in ['duration', 'event'] + cols_to_drop]
        if not base_features or not combo_features: return None

        cph_base_boot = CoxPHFitter(penalizer=penalizer).fit(df_base_boot, 'duration', 'event', formula=" + ".join(base_features))
        cph_combo_boot = CoxPHFitter(penalizer=penalizer).fit(df_combo_boot, 'duration', 'event', formula=" + ".join(combo_features))
        
        true_events_boot = df_base_boot['event'].values
        
        iteration_metrics = {}
        c_base_b = 1 - concordance_index(df_base_boot['duration'], cph_base_boot.predict_partial_hazard(df_base_boot[base_features]), true_events_boot)
        c_combo_b = 1 - concordance_index(df_combo_boot['duration'], cph_combo_boot.predict_partial_hazard(df_combo_boot[combo_features]), true_events_boot)
        
        iteration_metrics['c_index_base'] = c_base_b
        iteration_metrics['c_index_combo'] = c_combo_b
        iteration_metrics['delta_c_index'] = c_combo_b - c_base_b
        
        return iteration_metrics
    except Exception:
        return None

class SubgroupBootstrapEvaluator:
    """
    Performs subgroup analysis by calculating bootstrapped confidence intervals
    for C-Index within each subgroup.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, logger, penalizer: float = 0.0, file_suffix: str = ""):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.logger = logger
        self.penalizer = penalizer
        self.file_suffix = file_suffix

        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        self.scores_data_path = os.path.join(self.results_dir, 'Scores/OmicsNet/Final')
        
        self.save_dir = os.path.join(self.results_dir, 'Cindex')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _load_model_data(self, predictor_list: list, outcome: str, data_split: str) -> pd.DataFrame:
        """
        Loads and prepares a merged DataFrame using the modern data loading flow.
        """
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
                if predictor in ['AgeSex', 'Clinical', 'PANEL']:
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_{predictor}.feather")
                    x_df = pd.read_feather(path)
                elif predictor == 'Genomics':
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_Genomics.feather")
                    x_df = pd.read_feather(path)[['eid', f'{outcome}_prs']].rename(columns={f'{outcome}_prs': 'prs'})
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

    def _run_bootstrap_on_test_set(self, df_base, df_combo, cols_to_drop) -> dict:
        """
        Manages the parallel execution of bootstrap iterations for a given subgroup.
        """
        n_bootstraps = 1000
        max_workers = 92
        self.logger.info(f"Starting bootstrap with {n_bootstraps} iterations...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_bootstrap_iteration, i, df_base, df_combo, self.penalizer, cols_to_drop) for i in range(n_bootstraps)]
            
            bootstrap_metrics = defaultdict(list)
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 200 == 0: self.logger.info(f"  Completed {i+1}/{n_bootstraps} iterations...")
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
            if len(values) > 20:
                ci_results[f'{metric}_mean'] = np.nanmean(values)
                ci_results[f'{metric}_ci_lower'] = np.nanpercentile(values, 2.5)
                ci_results[f'{metric}_ci_upper'] = np.nanpercentile(values, 97.5)
        
        return ci_results

    def run_analysis(self):
        """
        Orchestrates the subgroup bootstrap evaluation.
        """
        outcomes_to_run = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        DATA_SPLIT = 'external_test'
        
        predictor_sets = [
            ['PANEL'], ['PANEL', 'Genomics'], ['PANEL', 'Metabolomics'], ['PANEL', 'Proteomics'],
            ['PANEL', 'Genomics', 'Metabolomics'], ['PANEL', 'Genomics', 'Proteomics'],
            ['PANEL', 'Metabolomics', 'Proteomics'], ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics'],
            ['Genomics'], ['Metabolomics'], ['Proteomics']
        ]
        
        try:
            scaler_params_df = pd.read_csv(os.path.join(self.split_seed_path, "Clinical_scaler_params.csv"))
            age_params = scaler_params_df[scaler_params_df['feature'] == 'age'].iloc[0]
            age_mean, age_std = age_params['mean'], np.sqrt(age_params['variance'])
        except (FileNotFoundError, IndexError) as e:
            self.logger.error(f"Could not load age scaler parameters. Error: {e}. Aborting.")
            return

        subgroups = {
            'Male': {'filter_lambda': lambda df: df['male_1.0'] == 1, 'drop_cols': ['male_1.0']},
            'Female': {'filter_lambda': lambda df: df['male_1.0'] == 0, 'drop_cols': ['male_1.0']}
        }
        
        try:
            self.logger.info("Loading PANEL data to create the master dataframe for subgroup filtering...")
            df_filter_master = pd.read_feather(os.path.join(self.split_seed_path, f"X_{DATA_SPLIT}_PANEL.feather"))
        except FileNotFoundError as e:
            self.logger.error(f"Critical error: PANEL data for subgrouping not found. Error: {e}. Aborting.")
            return

        total_predictor_sets = len(predictor_sets)
        all_results = []
        for subgroup_name, subgroup_info in subgroups.items():
            subgroup_start_time = time.time()
            self.logger.info(f"\n{'#'*20} PROCESSING SUBGROUP: {subgroup_name} {'#'*20}")
            
            subgroup_eids = df_filter_master.loc[subgroup_info['filter_lambda'](df_filter_master), 'eid']
            if subgroup_eids.empty:
                self.logger.warning(f"Subgroup '{subgroup_name}' is empty. Skipping.")
                continue

            for outcome in outcomes_to_run:
                outcome_start_time = time.time()
                self.logger.info(f"\n{'='*20} Outcome: {outcome.upper()} | Subgroup: {subgroup_name} {'='*20}")
                
                self.logger.info("Pre-loading baseline models for this subgroup...")
                baseline_dataframes = {}
                for base in [['AgeSex'], ['Clinical'], ['PANEL']]:
                    base_name = '+'.join(base)
                    df_full = self._load_model_data(base, outcome, DATA_SPLIT)
                    if not df_full.empty:
                        baseline_dataframes[base_name] = df_full[df_full['eid'].isin(subgroup_eids)]
                        self.logger.info(f"Loaded baseline '{base_name}' with {len(baseline_dataframes[base_name])} samples in subgroup.")
                    else:
                        self.logger.warning(f"Could not load baseline '{base_name}'.")

                for i, combo in enumerate(predictor_sets):
                    if 'PANEL' in combo:
                        baseline_name = 'PANEL'
                    elif 'Clinical' in combo:
                        baseline_name = 'Clinical'
                    else:
                        baseline_name = 'AgeSex'

                    df_baseline = baseline_dataframes.get(baseline_name)

                    if df_baseline is None or df_baseline.empty:
                        self.logger.warning(f"Baseline '{baseline_name}' not available or empty for this subgroup. Skipping combo: {combo}")
                        continue
                    
                    self.logger.info(f"\n----- [Subgroup: {subgroup_name} | Outcome: {outcome.upper()} | Set: {i+1}/{total_predictor_sets}] -----")
                    self.logger.info(f"Bootstrapping Combo: {combo} vs. Base: {[baseline_name]}")
                    
                    df_combo_full = self._load_model_data(combo, outcome, DATA_SPLIT)
                    if df_combo_full.empty: continue
                    df_combo = df_combo_full[df_combo_full['eid'].isin(subgroup_eids)]
                    
                    common_eids = pd.merge(df_baseline[['eid']], df_combo[['eid']], on='eid', how='inner')['eid']
                    if len(common_eids) < 100:
                        self.logger.warning(f"Skipping combo {combo} due to small sample size ({len(common_eids)}).")
                        continue
                        
                    df_base_common = df_baseline[df_baseline['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                    df_combo_common = df_combo[df_combo['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                    metrics_ci = self._run_bootstrap_on_test_set(df_base_common, df_combo_common, subgroup_info['drop_cols'])
                    
                    if metrics_ci:
                        metrics_ci['subgroup'] = subgroup_name
                        metrics_ci['outcome'] = outcome
                        metrics_ci['baseline_model'] = baseline_name
                        metrics_ci['comparison_model'] = '_'.join(combo)
                        metrics_ci['n_samples'] = len(common_eids)
                        all_results.append(metrics_ci)

                outcome_end_time = time.time()
                self.logger.info(f"Time for outcome '{outcome}' in subgroup '{subgroup_name}': {(outcome_end_time - outcome_start_time) / 60:.2f} min.")
            
            subgroup_end_time = time.time()
            self.logger.info(f"\n{'#'*20} COMPLETED SUBGROUP: {subgroup_name} | Total time: {(subgroup_end_time - subgroup_start_time) / 60:.2f} minutes {'#'*20}")

        if all_results:
            results_df = pd.DataFrame(all_results)
            cols_to_drop = ['c_index_base_ci_lower', 'c_index_base_ci_upper', 'c_index_base_mean']
            results_df = results_df.drop(columns=cols_to_drop, errors='ignore')
            id_cols = ['subgroup', 'outcome', 'baseline_model', 'comparison_model', 'n_samples']
            metric_cols = sorted([col for col in results_df.columns if col not in id_cols])
            results_df = results_df[id_cols + metric_cols]
            
            results_filename = os.path.join(self.save_dir, f"cindex_subgroup_bootstrap_ci_{self.file_suffix}.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Results saved to {results_filename}")

if __name__ == '__main__':
    script_start_time = time.time()
    
    FILE_SUFFIX = "Sex"
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, f"CoxPH/Bootstrap_CI_Subgroup_{FILE_SUFFIX}.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("SubgroupBootstrapScript")
    
    evaluator = SubgroupBootstrapEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER,
        file_suffix=FILE_SUFFIX
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")