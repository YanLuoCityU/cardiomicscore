import os
import time
import pandas as pd
import numpy as np
import logging
import sys
from lifelines.utils import concordance_index
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_bootstrap_iteration(iter_num: int, df: pd.DataFrame, score_col_name: str) -> float:
    """
    Performs a single bootstrap iteration on the given dataframe.
    It resamples the data and calculates the C-index.
    """
    try:
        # Resample using indices
        indices = resample(df.index, random_state=iter_num)
        df_boot = df.loc[indices]

        if df_boot.shape[0] < 50 or df_boot['event'].sum() < 5:
            return np.nan

        # Calculate C-index on the bootstrap sample
        c_index = 1 - concordance_index(
            event_times=df_boot['duration'],
            predicted_scores=df_boot[score_col_name],
            event_observed=df_boot['event']
        )
        return c_index
    except Exception:
        return np.nan

class ClinicalScoreEvaluator:
    """
    A base class for evaluating clinical risk scores on an external test set.
    It calculates a point estimate C-index and a bootstrapped confidence interval.
    """
    def __init__(self, data_dir: str, results_dir: str, seed_to_split: int, logger):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed_to_split = seed_to_split
        self.logger = logger
        self.data_split = 'external_test'
        
        self.split_seed_path = os.path.join(self.data_dir, f"split_seed-{self.seed_to_split}")
        os.makedirs(self.results_dir, exist_ok=True)

        # Subclasses are expected to define these
        self.predictor_name = "BasePredictor"
        self.score_col_name = "score"

    def _load_y_e_data(self, outcome: str) -> pd.DataFrame:
        """Loads outcome data (y and e) based on the logic from coxph.py."""
        self.logger.info(f"Loading 'duration' and 'event' data for outcome '{outcome}'.")
        try:
            y_path = os.path.join(self.split_seed_path, f"y_{self.data_split}.feather")
            e_path = os.path.join(self.split_seed_path, f"e_{self.data_split}.feather")
            y_df = pd.read_feather(y_path)
            e_df = pd.read_feather(e_path)
            
            duration_col, event_col = f'bl2{outcome}_yrs', outcome
            if duration_col not in y_df.columns or event_col not in e_df.columns:
                self.logger.error(f"Outcome columns '{duration_col}' or '{event_col}' not found.")
                return pd.DataFrame()
                
            y_df_renamed = y_df[['eid', duration_col]].rename(columns={duration_col: 'duration'})
            e_df_renamed = e_df[['eid', event_col]].rename(columns={event_col: 'event'})
            merged_df = pd.merge(y_df_renamed, e_df_renamed, on='eid')
            return merged_df
        except FileNotFoundError as e:
            self.logger.error(f"Could not load base y/e files. Error: {e}")
            return pd.DataFrame()

    def _load_predictor_data(self) -> pd.DataFrame:
        """(Placeholder) Subclasses must implement this to load their specific predictor data."""
        raise NotImplementedError("Subclasses must implement _load_predictor_data.")

    def _calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """(Placeholder) Subclasses must implement this to calculate the risk score."""
        raise NotImplementedError("Subclasses must implement _calculate_score.")

    def _run_bootstrap(self, df: pd.DataFrame) -> dict:
        """Manages the parallel execution of bootstrap iterations."""
        n_bootstraps = 1000
        # Adjust max_workers based on your machine's configuration
        max_workers = 68
        self.logger.info(f"Starting {n_bootstraps} bootstrap iterations using {max_workers} parallel processes...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_bootstrap_iteration, i, df, self.score_col_name) for i in range(n_bootstraps)]
            
            bootstrap_cindices = []
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 100 == 0:
                    self.logger.info(f"  Completed {i+1}/{n_bootstraps} bootstrap iterations...")
                result = future.result()
                if not np.isnan(result):
                    bootstrap_cindices.append(result)
        
        self.logger.info(f"Collected {len(bootstrap_cindices)} valid bootstrap results.")
        if len(bootstrap_cindices) < 50:
            self.logger.warning("Not enough valid bootstrap results to calculate CI.")
            return {'cindex_ci_lower': np.nan, 'cindex_ci_upper': np.nan}

        # Calculate confidence intervals
        ci_lower = np.nanpercentile(bootstrap_cindices, 2.5)
        ci_upper = np.nanpercentile(bootstrap_cindices, 97.5)
        
        return {'cindex_ci_lower': ci_lower, 'cindex_ci_upper': ci_upper}

    def run_analysis(self, outcomes_to_run: list):
        """
        Orchestrates the point estimate and bootstrap evaluation for all specified outcomes.
        """
        all_results = []
        self.logger.info(f"\n{'='*20} Starting Evaluation: {self.predictor_name.upper()} {'='*20}")

        # 1. Load predictor data (only once)
        X_df = self._load_predictor_data()
        if X_df.empty:
            self.logger.error(f"Failed to load predictor data for {self.predictor_name}. Aborting.")
            return pd.DataFrame()

        for outcome in outcomes_to_run:
            self.logger.info(f"\n----- Processing Outcome: {outcome.upper()} -----")
            
            # 2. Load outcome data
            y_e_df = self._load_y_e_data(outcome)
            if y_e_df.empty:
                self.logger.warning(f"Skipping outcome {outcome} because y/e data could not be loaded.")
                continue

            # 3. Merge data
            df_test = pd.merge(X_df, y_e_df, on='eid').dropna()
            self.logger.info(f"Found {len(df_test)} common samples for {outcome}.")
            if len(df_test) < 100:
                self.logger.warning(f"Sample size too small ({len(df_test)}), skipping.")
                continue

            # 4. Calculate score
            df_test = self._calculate_score(df_test)

            # 5. Calculate point estimate C-index
            cindex_point_estimate = 1 - concordance_index(
                event_times=df_test['duration'],
                predicted_scores=df_test[self.score_col_name],
                event_observed=df_test['event']
            )
            self.logger.info(f"Point Estimate C-index: {cindex_point_estimate:.4f}")

            # 6. Run Bootstrap Analysis
            ci_results = self._run_bootstrap(df_test)
            self.logger.info(f"Bootstrap 95% CI: [{ci_results['cindex_ci_lower']:.4f} - {ci_results['cindex_ci_upper']:.4f}]")
            
            # 7. Store results
            all_results.append({
                'outcome': outcome,
                'predictor': self.predictor_name,
                'cindex': cindex_point_estimate,
                'cindex_ci_lower': ci_results['cindex_ci_lower'],
                'cindex_ci_upper': ci_results['cindex_ci_upper']
            })

        return pd.DataFrame(all_results)


class ASCVD_Evaluator(ClinicalScoreEvaluator):
    """Specific implementation for the ASCVD score."""
    def __init__(self, data_dir, results_dir, seed_to_split, logger):
        super().__init__(data_dir, results_dir, seed_to_split, logger)
        self.predictor_name = "ascvd"
        self.score_col_name = "ascvd"

    def _load_predictor_data(self) -> pd.DataFrame:
        # MODIFIED: Path now uses self.split_seed_path
        path = os.path.join(self.split_seed_path, f"X_{self.data_split}_ASCVD.feather")
        self.logger.info(f"Loading ASCVD predictors from {path}.")
        try:
            return pd.read_feather(path)
        except FileNotFoundError:
            self.logger.error(f"ASCVD predictor file not found: {path}")
            return pd.DataFrame()

    def _calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the ASCVD score using the logic from clinical_scores.py.

        Coefficients are from the following paper:
        
        Yadlowsky S, Hayward RA, Sussman JB, McClelland RL, Min YI, Basu S. 
        Clinical Implications of Revised Pooled Cohort Equations for Estimating Atherosclerotic Cardiovascular Disease Risk. 
        Ann Intern Med. 2018;169(1):20-29. doi:10.7326/M17-3011
        
        Appendix Table. Example Calculation for Model Set 2, the Proposed Revision of the PCEs for Estimating ASCVD Risk
        https://www.msdmanuals.com/professional/multimedia/clinical-calculator/cardiovascular-risk-assessment-10-year-revised-pooled-cohort-equations-2018
        """
        df = df.copy()
        coefficients = {
            'male': {'age': 0.064200, 'sbp': 0.038950, 'current_smoking': 0.895589, 'diab_hist': 0.842209, 'antihypt': 2.055533, 'black': 0.482835,'totalcl_hdlcl': 0.193307, 'sbp2': -0.000061, 'sbp_antihypt': -0.014207, 'black_age': 0, 'black_sbp': 0.011609, 'black_antihypt': 0.119460, 'black_diab_hist': -0.077214, 'black_current_smoking': -0.226771,'black_totalcl_hdlcl': -0.117749, 'black_sbp_antihypt': 0.004190, 'black_age_sbp': -0.000199, 'age_sbp': 0.000025},
            'female': {'age': 0.106501, 'sbp': 0.017666, 'current_smoking': 1.009790, 'diab_hist': 0.943970, 'antihypt': 0.731678, 'black': 0.432440,'totalcl_hdlcl': 0.151318, 'sbp2': 0.000056, 'sbp_antihypt': -0.003647, 'black_age': -0.008580, 'black_sbp': 0.006208, 'black_antihypt': 0.152968, 'black_diab_hist': 0.115232, 'black_current_smoking': -0.092231,'black_totalcl_hdlcl': 0.070498, 'black_sbp_antihypt': -0.000173, 'black_age_sbp': -0.000094, 'age_sbp': -0.000153}
        }
        
        male_indices = df[df['male']==1].index
        female_indices = df[df['male']==0].index
        
        df[self.score_col_name] = 0.0
        # Males
        lp_male = -11.679980
        for feature, coef in coefficients['male'].items():
            lp_male += coef * df.loc[male_indices, feature]
        df.loc[male_indices, self.score_col_name] = lp_male
        
        # Females
        lp_female = -12.823110
        for feature, coef in coefficients['female'].items():
            lp_female += coef * df.loc[female_indices, feature]
        df.loc[female_indices, self.score_col_name] = lp_female
        
        df[self.score_col_name] = 1 / (1 + np.exp(-df[self.score_col_name]))
        return df


class SCORE2_Evaluator(ClinicalScoreEvaluator):
    """Specific implementation for the SCORE2 score."""
    def __init__(self, data_dir, results_dir, seed_to_split, logger):
        super().__init__(data_dir, results_dir, seed_to_split, logger)
        self.predictor_name = "score2"
        self.score_col_name = "score2"

    def _load_predictor_data(self) -> pd.DataFrame:
        # MODIFIED: Path now uses self.split_seed_path
        path = os.path.join(self.split_seed_path, f"X_{self.data_split}_SCORE2.feather")
        self.logger.info(f"Loading SCORE2 predictors from {path}.")
        try:
            return pd.read_feather(path)
        except FileNotFoundError:
            self.logger.error(f"SCORE2 predictor file not found: {path}")
            return pd.DataFrame()
            
    def _calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the SCORE2 score using the logic from clinical_scores.py.

        Coefficients are from the following paper:
        
        SCORE2 working group and ESC Cardiovascular risk collaboration. 
        SCORE2 risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in Europe. 
        Eur Heart J. 2021;42(25):2439-2454. doi:10.1093/eurheartj/ehab309
        
        Supplementary methods Table 2: Model coefficients and baseline survival of the SCORE2 algorithm
        Supplementary Table 7: Summary of subdistribution hazard ratios for predictor variables in the SCORE2 risk models
        """
        df = df.copy()
        coefficients = {
            'male': {'age': 0.3742, 'current_smoking': 0.6012, 'sbp': 0.2777, 'diab_hist': 0.6457, 'total_cl': 0.1458, 'hdl_cl': -0.2698, 'age_current_smoking': -0.0755, 'age_sbp': -0.0255, 'age_total_cl': -0.0281, 'age_hdl_cl': 0.0426, 'age_diab_hist': -0.0983},
            'female': {'age': 0.4648, 'current_smoking': 0.7744, 'sbp': 0.3131, 'diab_hist': 0.8096, 'total_cl': 0.1002, 'hdl_cl': -0.2606, 'age_current_smoking': -0.1088, 'age_sbp': -0.0277, 'age_total_cl': -0.0226, 'age_hdl_cl': 0.0613, 'age_diab_hist': -0.1272}
        }
        
        male_indices = df[df['male']==1].index
        female_indices = df[df['male']==0].index
        
        df[self.score_col_name] = 0.0
        # Males
        lp_male = 0.0
        for feature, coef in coefficients['male'].items():
            lp_male += coef * df.loc[male_indices, feature]
        df.loc[male_indices, self.score_col_name] = 1 - (0.9605 ** np.exp(lp_male))

        # Females
        lp_female = 0.0
        for feature, coef in coefficients['female'].items():
            lp_female += coef * df.loc[female_indices, feature]
        df.loc[female_indices, self.score_col_name] = 1 - (0.9776 ** np.exp(lp_female))
        
        return df

if __name__ == '__main__':
    script_start_time = time.time()
    
    # --- Configuration ---
    DATA_DIR = '/your path/cardiomicscore/data/'
    RESULTS_DIR = '/your path/cardiomicscore/saved/results/Cindex/'
    LOG_DIR = '/your path/cardiomicscore/saved/log/ClinicalScores/'
    SEED_TO_SPLIT = 250901
    OUTCOMES = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']

    # --- Logger Setup ---
    log_filename = os.path.join(LOG_DIR, "ClinicalScores_Bootstrap_Evaluation.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger("ClinicalScoreScript")
    
    # --- Run Evaluations ---
    final_results_df = pd.DataFrame()
    
    # ASCVD Evaluation
    ascvd_evaluator = ASCVD_Evaluator(
        data_dir=DATA_DIR, results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, logger=main_logger
    )
    ascvd_results = ascvd_evaluator.run_analysis(outcomes_to_run=OUTCOMES)
    final_results_df = pd.concat([final_results_df, ascvd_results], ignore_index=True)

    # SCORE2 Evaluation
    score2_evaluator = SCORE2_Evaluator(
        data_dir=DATA_DIR, results_dir=RESULTS_DIR, 
        seed_to_split=SEED_TO_SPLIT, logger=main_logger
    )
    score2_results = score2_evaluator.run_analysis(outcomes_to_run=OUTCOMES)
    final_results_df = pd.concat([final_results_df, score2_results], ignore_index=True)

    # --- Save Final Results ---
    if not final_results_df.empty:
        results_filename = os.path.join(RESULTS_DIR, "clinical_scores_cindex_summary.csv")
        final_results_df.to_csv(results_filename, index=False, float_format='%.4f')
        main_logger.info(f"\n\nSUCCESS: All analyses complete. Consolidated C-Index results saved to {results_filename}")
    else:
        main_logger.warning("No results were generated.")
    
    total_elapsed_time = time.time() - script_start_time
    main_logger.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")