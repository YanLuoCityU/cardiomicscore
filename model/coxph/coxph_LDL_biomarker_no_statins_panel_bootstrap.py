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
        
        # Calculate C-index metrics for this iteration
        iteration_metrics = {}
        c_base_b = 1 - concordance_index(df_base_boot['duration'], cph_base_boot.predict_partial_hazard(df_base_boot[base_features]), true_events_boot)
        c_combo_b = 1 - concordance_index(df_combo_boot['duration'], cph_combo_boot.predict_partial_hazard(df_combo_boot[combo_features]), true_events_boot)
        
        iteration_metrics['c_index_base'] = c_base_b
        iteration_metrics['c_index_combo'] = c_combo_b
        iteration_metrics['delta_c_index'] = c_combo_b - c_base_b
        
        return iteration_metrics
    except Exception:
        return None

class BootstrapEvaluator:
    """
    Evaluates CoxPH models by calculating bootstrapped confidence intervals for C-Index.
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
        Loads and prepares a merged DataFrame using the modern data loading flow,
        now including logic for single biomarkers.
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

        metabolite_biomarkers = [
            "Total_C", "non_HDL_C", "Remnant_C", "Total_TG", "Total_PL", "Total_CE", "Total_FC", "Total_L", # Total Lipids / Cholesterol
            "Clinical_LDL_C", "LDL_C", "LDL_TG", "LDL_PL", "LDL_CE", "LDL_FC", "LDL_L", "LDL_P", "LDL_size" # Overall LDL Measurements
        ]
        
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
                elif predictor in metabolite_biomarkers:
                    path = os.path.join(self.split_seed_path, f"X_{data_split}_Metabolomics_no_statins.feather")
                    metabolomics_full = pd.read_feather(path)
                    x_df = metabolomics_full[['eid', predictor]]
                else: 
                    self.logger.warning(f"Predictor '{predictor}' has no defined loading logic. Skipping.")
                    continue
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
        
        predictor_sets = [["PANEL"]]
        
        total_predictor_sets = len(predictor_sets)
        all_results = []
        
        # --- NEW: Load lipid treatment status to filter the dataset ---
        panel_path = os.path.join(self.split_seed_path, f"X_{DATA_SPLIT}_PANEL.feather")
        try:
            self.logger.info("Loading lipid treatment status for filtering.")
            panel_df = pd.read_feather(panel_path)
            eids_no_treatment = panel_df[panel_df['lipidlower_1.0'] == 0]['eid']
            self.logger.info(f"Identified {len(eids_no_treatment)} individuals not on lipid-lowering therapy.")
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"CRITICAL: Could not load lipid treatment status from {panel_path}. Error: {e}")
            return # Exit if the filtering file isn't available

        for outcome in outcomes_to_run:
            outcome_start_time = time.time()
            self.logger.info(f"\n{'#'*20} PROCESSING OUTCOME: {outcome.upper()} {'#'*20}")
            
            baseline_dataframes = {}
            for base in [['AgeSex'], ['Clinical'], ['PANEL']]:
                self.logger.info(f"Pre-loading baseline data for: {base}")
                df = self._load_model_data(base, outcome, DATA_SPLIT)
                # --- MODIFIED: Filter pre-loaded baseline dataframes ---
                baseline_dataframes['+'.join(base)] = df[df['eid'].isin(eids_no_treatment)].copy()

            for i, combo in enumerate(predictor_sets):
                self.logger.info(f"\n----- [Outcome: {outcome.upper()} | Set: {i+1}/{total_predictor_sets}] -----")
                if 'PANEL' in combo: baseline_name = 'PANEL'
                elif 'Clinical' in combo: baseline_name = 'Clinical'
                else: baseline_name = 'AgeSex'
                
                df_baseline_full = baseline_dataframes.get(baseline_name)
                if df_baseline_full is None or df_baseline_full.empty:
                    self.logger.warning(f"Baseline data '{baseline_name}' is empty after filtering. Skipping.")
                    continue

                self.logger.info(f"Bootstrapping Combo: {combo} vs. Base: {[baseline_name]}")
                df_combo_full = self._load_model_data(combo, outcome, DATA_SPLIT)
                if df_combo_full.empty: continue
                
                # --- MODIFIED: Filter the combination dataframe ---
                df_combo_full = df_combo_full[df_combo_full['eid'].isin(eids_no_treatment)]
                if df_combo_full.empty:
                    self.logger.warning(f"Combo data for {combo} is empty after filtering. Skipping.")
                    continue

                common_eids = pd.merge(df_baseline_full[['eid']], df_combo_full[['eid']], on='eid', how='inner')['eid']
                if len(common_eids) < 100:
                    self.logger.warning(f"Skipping combo {combo} due to small sample size ({len(common_eids)}) post-filtering.")
                    continue
                
                df_base_common = df_baseline_full[df_baseline_full['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                df_combo_common = df_combo_full[df_combo_full['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                # --- NEW: Remove the treatment variable before sending data to bootstrap ---
                if 'lipidlower_1.0' in df_base_common.columns:
                    df_base_common = df_base_common.drop(columns=['lipidlower_1.0'])
                if 'lipidlower_1.0' in df_combo_common.columns:
                    df_combo_common = df_combo_common.drop(columns=['lipidlower_1.0'])

                metrics_ci = self._run_bootstrap_on_test_set(df_base_common, df_combo_common)
                
                if metrics_ci:
                    metrics_ci['outcome'] = outcome
                    metrics_ci['baseline_model'] = baseline_name
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
            
            results_filename = os.path.join(self.save_dir, "cindex_LDL_biomarker_no_statins_panel_bootstrap_ci_summary.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Bootstrap CI results saved to {results_filename}")

if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Bootstrap_CI_LDL_Biomarker_No_Statins_PANEL.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("BootstrapScript")
    
    evaluator = BootstrapEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")