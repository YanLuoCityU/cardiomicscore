import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

class SubgroupAnalysisEvaluator:
    """
    Performs subgroup analysis using CoxPH models on the external test set.
    It calculates the Concordance Index (C-index) for various predictor combinations
    across multiple predefined patient subgroups.
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
        Loads and prepares a merged DataFrame for a given predictor combination and data split.
        This is the modern data loading function from coxph.py.
        """
        self.logger.info(f"Loading '{data_split}' data for {predictor_list} (Outcome: {outcome})")
        if not predictor_list: return pd.DataFrame()
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

    def run_analysis(self):
        """
        Orchestrates the subgroup analysis evaluation.
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
            scaler_params_df = pd.read_csv(os.path.join(self.split_seed_path, "AgeSex_scaler_params.csv"))
            age_params = scaler_params_df[scaler_params_df['feature'] == 'age'].iloc[0]
            age_mean, age_std = age_params['mean'], np.sqrt(age_params['variance'])
            self.logger.info(f"Loaded age scaler parameters: mean={age_mean:.4f}, std={age_std:.4f}")
        except (FileNotFoundError, IndexError) as e:
            self.logger.error(f"Could not load age scaler parameters. Error: {e}. Aborting.")
            return

        subgroups = {
            'Age_under_60': {'filter_lambda': lambda df: (df['age'] * age_std + age_mean) < 60, 'drop_cols': ['age']},
            'Age_over_60': {'filter_lambda': lambda df: (df['age'] * age_std + age_mean) >= 60, 'drop_cols': ['age']},
            'Male': {'filter_lambda': lambda df: df['male_1.0'] == 1, 'drop_cols': ['male_1.0']},
            'Female': {'filter_lambda': lambda df: df['male_1.0'] == 0, 'drop_cols': ['male_1.0']},
            'Lipid_lowering_medication': {'filter_lambda': lambda df: df['lipidlower_1.0'] == 1, 'drop_cols': ['lipidlower_1.0']},
            'No_lipid_lowering_medication': {'filter_lambda': lambda df: df['lipidlower_1.0'] == 0, 'drop_cols': ['lipidlower_1.0']},
            'Antihypertensive_medication': {'filter_lambda': lambda df: df['antihypt_1.0'] == 1, 'drop_cols': ['antihypt_1.0']},
            'No_antihypertensive_medication': {'filter_lambda': lambda df: df['antihypt_1.0'] == 0, 'drop_cols': ['antihypt_1.0']}
        }
        
        try:
            self.logger.info("Loading PANEL data to create the master dataframe for subgroup filtering...")
            df_filter_master = pd.read_feather(os.path.join(self.split_seed_path, f"X_{DATA_SPLIT}_PANEL.feather"))
            
        except FileNotFoundError as e:
            self.logger.error(f"Critical error: PANEL data for subgrouping not found. Error: {e}. Aborting.")
            return

        all_results = []
        for subgroup_name, subgroup_info in subgroups.items():
            self.logger.info(f"\n{'#'*20} PROCESSING SUBGROUP: {subgroup_name} {'#'*20}")
            
            subgroup_eids = df_filter_master.loc[subgroup_info['filter_lambda'](df_filter_master), 'eid']
            if subgroup_eids.empty:
                self.logger.warning(f"Subgroup '{subgroup_name}' is empty. Skipping.")
                continue

            for outcome in outcomes_to_run:
                self.logger.info(f"\n{'='*20} Outcome: {outcome.upper()} | Subgroup: {subgroup_name} {'='*20}")
                
                baseline_dataframes = {}
                for base in [['AgeSex'], ['Clinical'], ['PANEL']]:
                    df_full = self._load_model_data(base, outcome, DATA_SPLIT)
                    if not df_full.empty:
                        baseline_dataframes['+'.join(base)] = df_full[df_full['eid'].isin(subgroup_eids)]

                for combo in predictor_sets:
                    if 'PANEL' in combo: baseline_name = 'PANEL'
                    elif 'Clinical' in combo: baseline_name = 'Clinical'
                    else: baseline_name = 'AgeSex'
                    
                    df_baseline = baseline_dataframes.get(baseline_name)
                    if df_baseline is None or df_baseline.empty: continue

                    self.logger.info(f"\n-- Combo: {combo} vs. Base: {[baseline_name]} --")
                    df_combo_full = self._load_model_data(combo, outcome, DATA_SPLIT)
                    if df_combo_full.empty: continue
                    df_combo = df_combo_full[df_combo_full['eid'].isin(subgroup_eids)]

                    common_eids = pd.merge(df_baseline[['eid']], df_combo[['eid']], on='eid', how='inner')['eid']
                    if len(common_eids) < 50: 
                        self.logger.warning(f"Skipping combo {combo} due to small sample size ({len(common_eids)}) in subgroup.")
                        continue
                        
                    df_base_common = df_baseline[df_baseline['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)
                    df_combo_common = df_combo[df_combo['eid'].isin(common_eids)].drop(columns=['eid']).reset_index(drop=True)

                    try:
                        cols_to_drop = subgroup_info['drop_cols']
                        base_features = [c for c in df_base_common.columns if c not in ['duration', 'event'] + cols_to_drop]
                        combo_features = [c for c in df_combo_common.columns if c not in ['duration', 'event'] + cols_to_drop]
                        if not base_features or not combo_features: continue

                        cph_base = CoxPHFitter(penalizer=self.penalizer).fit(df_base_common, 'duration', 'event', formula=" + ".join(base_features))
                        c_index_base = 1 - concordance_index(df_base_common['duration'], cph_base.predict_partial_hazard(df_base_common[base_features]), df_base_common['event'])

                        cph_combo = CoxPHFitter(penalizer=self.penalizer).fit(df_combo_common, 'duration', 'event', formula=" + ".join(combo_features))
                        c_index_combo = 1 - concordance_index(df_combo_common['duration'], cph_combo.predict_partial_hazard(df_combo_common[combo_features]), df_combo_common['event'])
                        
                        all_results.append({
                            'subgroup': subgroup_name, 'outcome': outcome, 'baseline_model': baseline_name,
                            'comparison_model': '_'.join(combo), 'n_samples': len(common_eids),
                            'n_events': int(df_combo_common['event'].sum()),
                            'c_index_base': c_index_base, 'c_index_combo': c_index_combo,
                            'delta_c_index': c_index_combo - c_index_base
                        })
                        self.logger.info(f"C-Index for {combo}: {c_index_combo:.4f} (Delta: {c_index_combo - c_index_base:+.4f})")

                    except Exception as e:
                        self.logger.error(f"Analysis failed for {combo}, outcome {outcome}, subgroup {subgroup_name}: {e}", exc_info=False)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_filename = os.path.join(self.cindex_save_dir, "cindex_subgroup_summary.csv")
            results_df.to_csv(results_filename, index=False)
            self.logger.info(f"\n\nSUCCESS: All analyses complete. Subgroup results saved to {results_filename}")

if __name__ == '__main__':
    script_start_time = time.time()
    
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/'
    SEED_TO_SPLIT = 250901
    PENALIZER = 0.03

    log_filename = os.path.join(LOG_DIR, "CoxPH/Point_Estimates_Subgroup.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger("SubgroupAnalysisScript")
    
    evaluator = SubgroupAnalysisEvaluator(
        data_dir=DATA_DIR, 
        results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, 
        logger=logger_main, 
        penalizer=PENALIZER
    )
    evaluator.run_analysis()

    total_elapsed_time = time.time() - script_start_time
    logger_main.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")