import os
import time
import pandas as pd
import numpy as np
import logging
from os.path import join, exists
from sklearn.model_selection import train_test_split

class UKBiobankData():
    def __init__(self,
                 data_dir,
                 log_dir,
                 val_size=0.1,
                 shuffle=True,
                 seed=12345):

        class_init_start_time = time.time()
        self.data_dir = data_dir
        self.val_size = val_size
        self.shuffle = shuffle
        self.seed = seed

        log_prep_dir = join(log_dir, 'DataPreparation')
        log_filename = join(log_prep_dir, f'UKBiobankData_raw_{self.seed}.log')
        self.logger = self._setup_logger(log_filename)
        
        self.split_seed_filename = f"split_seed_raw-{self.seed}"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        if not exists(self.split_seed_dir):
            os.makedirs(self.split_seed_dir)
        
        self.logger.info(f"Class instantiated. Log file at: {log_filename}")
        self.logger.info(f"Output will be saved to: {self.split_seed_dir}")

        load_data_start_time = time.time()
        self.load_data()
        self.logger.info(f"load_data() executed in {time.time() - load_data_start_time:.2f} seconds.")

        self.disease_list = ['cad', 'stroke', 'hf', 'af', 'pad', 'vte']
        self.disease_mapping = {
            'cad': 'Coronary artery disease', 'stroke': 'Stroke', 'vhd': 'Valvular heart disease', 'hf': 'Heart failure', 
            'af': 'Atrial fibrillation', 'pad': 'Peripheral artery disease',  'vte': 'Venous thromboembolism'
        }

        self.clinical_continuous_list = ['age', 'townsend', 'sbp', 'dbp', 'height', 'weight', 'waist_cir', 'waist_hip_ratio', 'bmi']
        self.blood_count_continuous_list = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        self.blood_biochem_continuous_list = [
            'apoA', 'apoB', 'total_cl', 'ldl_cl', 'hdl_cl', 'lpa', 'tg', 'glucose', 'hba1c', 'crt', 'cysc', 'urate', 'urea',
            'alb', 'tp', 'alt', 'ast', 'ggt', 'alp', 'dbil', 'tbil', 'crp', 'pho', 'ca', 'vd', 'igf1', 'shbg', 'trt'
        ]
        self.clinical_categorical_list = [
            'male', 'ethnicity', 'current_smoking', 'daily_drinking', 'healthy_sleep', 'physical_act', 'healthy_diet', 'social_active',
            'family_heart_hist', 'family_stroke_hist', 'family_hypt_hist', 'family_diab_hist',
            'diab_hist', 'hypt_hist', 'lipidlower', 'antihypt'
        ]
        
        self.dict_predictor = self.get_predictors_dict()
        self.predictor_set_mapping = {
            'Clinical': self.dict_predictor['clinical'],
            'Genomics': self.dict_predictor['genomics'], 
            'Metabolomics': self.dict_predictor['metabolomics'],
            'Proteomics': self.dict_predictor['proteomics']
        }
        
        self.logger.info("Adding clinical predictors to all omics sets.")
        clinical_predictors = self.dict_predictor['clinical']
        for p_set in ['Genomics', 'Metabolomics', 'Proteomics']:
            self.predictor_set_mapping[p_set] = list(dict.fromkeys(self.predictor_set_mapping[p_set] + clinical_predictors))
        
        merge_start_time = time.time()
        self.merge_clinical_predictors()
        self.merge_genomics()
        self.merge_metabolomics()
        self.merge_all_predictors()
        self.logger.info(f"All merge operations executed in {time.time() - merge_start_time:.2f} seconds.\n")

        exclusion_start_time = time.time()
        history_cols = [f'{disease}_hist' for disease in self.disease_list if f'{disease}_hist' in self.diseases_df.columns]
        preexisting_disease_mask = self.diseases_df[history_cols].sum(axis=1) > 0
        all_disease_eid_to_exclude = self.diseases_df[preexisting_disease_mask]['eid'].unique()
        
        self.excluded_df = self.exclusion(
            predictors_df=self.merge_df.copy(), 
            preexisting_disease_eids=all_disease_eid_to_exclude
        )
        self.main_dataset_eid = self.merge_df['eid'].unique()
        
        self.excluded_proteomics_pool = self.exclusion_subcohort(predictors_df=self.proteomics_df, main_test_set_eids=self.main_dataset_eid, preexisting_disease_eids=all_disease_eid_to_exclude, predictor_set='Proteomics')
        self.excluded_metabolomics_pool = self.exclusion_subcohort(predictors_df=self.merge_metabolomics_df, main_test_set_eids=self.main_dataset_eid, preexisting_disease_eids=all_disease_eid_to_exclude, predictor_set='Metabolomics')
        self.excluded_genomics_pool = self.exclusion_subcohort(predictors_df=self.merge_genomics_df, main_test_set_eids=self.main_dataset_eid, preexisting_disease_eids=all_disease_eid_to_exclude, predictor_set='Genomics')
        self.excluded_clinical_pool = self.exclusion_subcohort(predictors_df=self.merge_clinical_df, main_test_set_eids=self.main_dataset_eid, preexisting_disease_eids=all_disease_eid_to_exclude, predictor_set='Clinical_PANELs')
        
        self.logger.info(f"Exclusion operations executed in {time.time() - exclusion_start_time:.2f} seconds.\n")

        split_prepare_start_time = time.time()
        self._split_and_prepare_data()
        self.logger.info(f"_split_and_prepare_data() method executed in {time.time() - split_prepare_start_time:.2f} seconds.\n")
        
        process_save_start_time = time.time()
        self._process_and_save_datasets()
        self.logger.info(f"Data processing and saving completed in {time.time() - process_save_start_time:.2f} seconds.\n")

        self.logger.info(f"UKBiobankData class initialization complete in {time.time() - class_init_start_time:.2f} seconds.\n")

    def _setup_logger(self, log_filename):
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logger = logging.getLogger(log_filename)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler(log_filename, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self):
        processed_data_dir = os.path.join(self.data_dir, 'processed/')
        self.logger.info(f'Loading processed predictors and cardiovascular diseases from {processed_data_dir}.\n')
        self.demographic_df = pd.read_csv(join(processed_data_dir, 'covariates/DemographicInfo.csv'), low_memory=False)
        self.lifestyle_df = pd.read_csv(join(processed_data_dir, 'covariates/Lifestyle.csv'), low_memory=False)
        self.familyhist_df = pd.read_csv(join(processed_data_dir, 'covariates/FamilyHistory.csv'), low_memory=False)
        self.phymeasure_df = pd.read_csv(join(processed_data_dir, 'covariates/PhysicalMeasurements.csv'), low_memory=False)
        self.biofluids_df = pd.read_csv(join(processed_data_dir, 'covariates/Biofluids.csv'), low_memory=False)
        self.medication_df = pd.read_csv(join(processed_data_dir, 'covariates/MedicationHistory.csv'), low_memory=False)
        self.diseases_df = pd.read_csv(join(processed_data_dir, 'covariates/DiseaseHistory.csv'), low_memory=False)
        self.genomics_df = pd.read_csv(join(processed_data_dir, 'omics/PolygenicScores.csv'), low_memory=False)
        self.proteomics_df = pd.read_csv(join(processed_data_dir, 'omics/Proteomics.csv'), low_memory=False)
        self.outcomes_df = pd.read_csv(join(processed_data_dir, 'outcomes/Outcomes.csv'), low_memory=False)
        self.genotype_data_qc_df = pd.read_csv(join(processed_data_dir, 'covariates/GenotypeDataQC.csv'), low_memory=False)
        
        self.logger.info("Loading and filtering Metabolomics data...")
        metabolomics_path = join(processed_data_dir, 'omics/Metabolomics.csv')
        self.metabolomics_df = pd.read_csv(metabolomics_path, low_memory=False)
        self.logger.info(f"  ..loaded raw metabolomics data with shape: {self.metabolomics_df.shape}")

        nmr_info_path = '/home/ukb/data/resources/NMR/nmr_info.csv'
        ukb_nmr_info_df = pd.read_csv(nmr_info_path)
        
        filtered_df = ukb_nmr_info_df[
            (ukb_nmr_info_df['Nightingale'] == True) & 
            (ukb_nmr_info_df['Type'].isin(['Non-derived', 'Composite'])) &
            (~ukb_nmr_info_df['Description'].isin(['Spectrometer-corrected alanine', 'Glucose-lactate']))
        ]
        original_nmr_list = filtered_df['Biomarker'].tolist()
        
        self.metabolomics_df = self.metabolomics_df.filter(items=original_nmr_list + ['eid'])
        self.logger.info(f"  ..filtered metabolomics data to shape: {self.metabolomics_df.shape}")

    def get_predictors_dict(self):
        dict_predictor = {}
        dict_predictor['demographic'] = ['age', 'male', 'ethnicity', 'townsend']
        dict_predictor['lifestyle'] = ['current_smoking', 'daily_drinking', 'healthy_sleep', 'physical_act', 'healthy_diet', 'social_active']
        dict_predictor['familyhist'] = ['family_heart_hist', 'family_stroke_hist', 'family_hypt_hist', 'family_diab_hist']
        dict_predictor['phymeasure'] = ['sbp', 'dbp', 'height', 'weight', 'waist_cir', 'waist_hip_ratio', 'bmi']
        dict_predictor['medication'] = ['lipidlower', 'antihypt']
        dict_predictor['diseases'] = ['diab_hist', 'hypt_hist']
        dict_predictor['biofluids'] = self.blood_count_continuous_list + self.blood_biochem_continuous_list
        dict_predictor['genomics'] = self.genomics_df.columns.drop('eid').tolist()
        dict_predictor['metabolomics'] = self.metabolomics_df.columns.drop('eid').tolist()
        dict_predictor['proteomics'] = self.proteomics_df.columns.drop('eid').tolist()
        dict_predictor['clinical'] = (dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases'] + dict_predictor['biofluids'])
        return dict_predictor

    def merge_clinical_predictors(self):
        self.logger.info('Merging clinical predictors.')
        self.merge_clinical_df = self.demographic_df.copy()
        self.logger.info(f"Start with demographic_df, shape: {self.merge_clinical_df.shape}")
        dfs_to_merge = [('lifestyle_df', self.lifestyle_df), ('familyhist_df', self.familyhist_df), ('phymeasure_df', self.phymeasure_df), ('biofluids_df', self.biofluids_df), ('medication_df', self.medication_df), ('diseases_df', self.diseases_df)]
        for name, df in dfs_to_merge:
            self.merge_clinical_df = pd.merge(self.merge_clinical_df, df, on='eid', how='inner')
            self.logger.info(f"  ..merged with {name}, new shape: {self.merge_clinical_df.shape}")
        
        all_clinical_related_cols = list(dict.fromkeys(self.dict_predictor['clinical'] + ['male', 'age']))
        cols_to_keep = ['eid'] + [col for col in all_clinical_related_cols if col in self.merge_clinical_df.columns]
        self.merge_clinical_df = self.merge_clinical_df[cols_to_keep]
        self.logger.info(f"Final merged clinical df shape after column selection: {self.merge_clinical_df.shape}")

    def merge_genomics(self):
        self.logger.info('Merging genomics predictors.')
        self.merge_genomics_df = pd.merge(self.genomics_df, self.genotype_data_qc_df, on='eid', how='inner')
        self.logger.info(f"  ..merged with genotype_data_qc_df, new shape: {self.merge_genomics_df.shape}")
        self.merge_genomics_df = pd.merge(self.merge_genomics_df, self.demographic_df[['eid', 'male']], on='eid', how='inner')
        self.logger.info(f"  ..merged with demographic_df, new shape: {self.merge_genomics_df.shape}")
        self.logger.info(f"Final merged genomics df shape: {self.merge_genomics_df.shape}")

    def merge_metabolomics(self):
        self.logger.info('Merging metabolomics with outcomes to align participants.')
        self.merge_metabolomics_df = pd.merge(self.metabolomics_df, self.outcomes_df[['eid']], on='eid', how='inner')
        self.logger.info(f"Shape after merging metabolomics with outcomes: {self.merge_metabolomics_df.shape}")
        
    def merge_all_predictors(self):
        self.logger.info('Merging all predictors for external test set definition.')
        self.merge_df = self.proteomics_df.copy()
        self.logger.info(f"Start with proteomics_df, shape: {self.merge_df.shape}")
        dataframes_to_merge = [('metabolomics_df', self.metabolomics_df), ('genomics_df', self.genomics_df), ('demographic_df', self.demographic_df), ('lifestyle_df', self.lifestyle_df), ('familyhist_df', self.familyhist_df), ('phymeasure_df', self.phymeasure_df), ('biofluids_df', self.biofluids_df), ('medication_df', self.medication_df), ('diseases_df', self.diseases_df)]
        for name, df_to_merge in dataframes_to_merge:
            self.merge_df = pd.merge(self.merge_df, df_to_merge, on='eid', how='inner')
            self.logger.info(f"  ..merged with {name}, new shape: {self.merge_df.shape}")
        self.logger.info(f"Final merged df shape: {self.merge_df.shape} with {self.merge_df['eid'].nunique()} unique participants.")

    def exclusion(self, predictors_df, preexisting_disease_eids):
        self.logger.info(f"Starting main exclusion criteria to define EXTERNAL TEST set. Initial shape: {predictors_df.shape}")
        excluded_df = predictors_df[~predictors_df['eid'].isin(preexisting_disease_eids)].copy()
        self.logger.info(f'Excluded participants with any pre-existing CVD at baseline: {len(excluded_df["eid"].unique())} remaining.')
        proteomics_cols = self.dict_predictor['proteomics']
        missing_percentage = excluded_df[proteomics_cols].isnull().mean(axis=1) * 100
        excluded_df = excluded_df[missing_percentage <= 50]
        self.logger.info(f'Excluded participants missing >50% proteomics data: {len(excluded_df["eid"].unique())} remaining.')
        self.logger.info("Applying genetic QC exclusions...")
        excluded_df = pd.merge(excluded_df, self.genotype_data_qc_df, on='eid', how='left')
        original_count = len(excluded_df)
        excluded_df = excluded_df[excluded_df['male'] == excluded_df['genetic_sex']]
        self.logger.info(f'Excluded {original_count - len(excluded_df)} participants with discordant sex and genetic sex. {len(excluded_df)} remaining.')
        original_count = len(excluded_df)
        excluded_df = excluded_df[excluded_df['sex_chromosome_aneuploidy'] != 1]
        self.logger.info(f'Excluded {original_count - len(excluded_df)} participants for sex chromosome aneuploidy. {len(excluded_df)} remaining.')
        original_count = len(excluded_df)
        excluded_df = excluded_df[excluded_df['genetic_kinship'] != 10]
        self.logger.info(f'Excluded {original_count - len(excluded_df)} participants with 10+ third-degree relatives. {len(excluded_df)} remaining.')
        original_count = len(excluded_df)
        excluded_df = excluded_df[excluded_df['outliers_heterozygosity_missing_rate'] != 1]
        self.logger.info(f'Excluded {original_count - len(excluded_df)} participants for heterozygosity/missing rate outliers. {len(excluded_df)} remaining.\n')
        return excluded_df

    def exclusion_subcohort(self, predictors_df, main_test_set_eids, preexisting_disease_eids, predictor_set=None):
        self.logger.info(f"Creating TRAIN/VAL/INTERNAL TEST pool for predictor set: {predictor_set}. Initial shape: {predictors_df.shape}")
        
        pool_df = predictors_df[~predictors_df['eid'].isin(main_test_set_eids)].copy()
        self.logger.info(f"  ..Excluded {len(predictors_df) - len(pool_df)} participants found in the external test set. {len(pool_df)} remaining.")
        
        original_count = len(pool_df)
        pool_df = pool_df[~pool_df['eid'].isin(preexisting_disease_eids)].copy()
        self.logger.info(f"  ..Excluded {original_count - len(pool_df)} participants with pre-existing disease. {len(pool_df)} remaining.")
        
        if predictor_set == 'Genomics':
            self.logger.info("Applying genetic QC exclusions to Genomics pool...")
            original_count = len(pool_df)
            pool_df = pool_df[pool_df['male'] == pool_df['genetic_sex']]
            self.logger.info(f'  ..Excluded {original_count - len(pool_df)} participants with discordant sex. {len(pool_df)} remaining.')
            original_count = len(pool_df)
            pool_df = pool_df[pool_df['sex_chromosome_aneuploidy'] != 1]
            self.logger.info(f'  ..Excluded {original_count - len(pool_df)} participants for sex chromosome aneuploidy. {len(pool_df)} remaining.')
            original_count = len(pool_df)
            pool_df = pool_df[pool_df['genetic_kinship'] != 10]
            self.logger.info(f'  ..Excluded {original_count - len(pool_df)} participants with 10+ third-degree relatives. {len(pool_df)} remaining.')
            original_count = len(pool_df)
            pool_df = pool_df[pool_df['outliers_heterozygosity_missing_rate'] != 1]
            self.logger.info(f'  ..Excluded {original_count - len(pool_df)} participants for heterozygosity/missing rate outliers. {len(pool_df)} remaining.')
        
        elif predictor_set == 'Proteomics':
            self.logger.info("Applying proteomics QC exclusions to Proteomics pool...")
            proteomics_cols = self.dict_predictor['proteomics']
            original_count = len(pool_df)
            missing_percentage = pool_df[proteomics_cols].isnull().mean(axis=1) * 100
            pool_df = pool_df[missing_percentage <= 50]
            self.logger.info(f'  ..Excluded {original_count - len(pool_df)} participants missing >50% proteomics data. {len(pool_df)} remaining.')
            
        self.logger.info(f"Finished creating pool for {predictor_set}. Final shape: {pool_df.shape}.\n")
        return pool_df

    def _log_split_stats(self, df, set_name):
        if df.empty:
            self.logger.info(f"{set_name}: 0 participants.")
            return
        
        outcomes_subset = self.outcomes_df[self.outcomes_df['eid'].isin(df['eid'])]
        event_counts = outcomes_subset[self.disease_list].sum()
        
        self.logger.info(f"{set_name}: {len(df)} participants.")
        for disease, count in event_counts.items():
            self.logger.info(f"  - {self.disease_mapping[disease]} ({disease}): {count} cases")

    def _create_splits(self, pool_df, pool_name):
        self.logger.info("\n")
        self.logger.info(f"--- Splitting {pool_name} cohort ---")
        
        pool_df_region = pd.merge(pool_df, self.demographic_df[['eid', 'region']], on='eid', how='left')
        internal_test_df = pool_df_region[pool_df_region['region'] == 9].drop(columns=['region'])
        training_pool_df = pool_df_region[pool_df_region['region'] != 9].drop(columns=['region'])
        
        train_df, val_df = train_test_split(training_pool_df, 
                                            test_size=self.val_size, 
                                            shuffle=self.shuffle, 
                                            random_state=self.seed)
        
        self._log_split_stats(train_df, f"{pool_name} Training Set")
        self._log_split_stats(val_df, f"{pool_name} Validation Set")
        self._log_split_stats(internal_test_df, f"{pool_name} Internal Test Set (Scotland)")
        
        return train_df, val_df, internal_test_df

    def _split_and_prepare_data(self):
        self.X_train_clinical_df, self.X_val_clinical_df, self.X_internal_test_clinical_df = self._create_splits(self.excluded_clinical_pool, "Clinical")
        self.X_train_genomics_df, self.X_val_genomics_df, self.X_internal_test_genomics_df = self._create_splits(self.excluded_genomics_pool, "Genomics")
        self.X_train_proteomics_df, self.X_val_proteomics_df, self.X_internal_test_proteomics_df = self._create_splits(self.excluded_proteomics_pool, "Proteomics")
        self.X_train_metabolomics_df, self.X_val_metabolomics_df, self.X_internal_test_metabolomics_df = self._create_splits(self.excluded_metabolomics_pool, "Metabolomics")

        self.X_external_test_main_df = self.excluded_df.copy()
        self.logger.info("\n")
        self.logger.info("--- External Test Set ---")
        self._log_split_stats(self.X_external_test_main_df, "External Test Set")

    def _get_y_e_sets(self, eids_series):
        if eids_series.empty:
            return pd.DataFrame(), pd.DataFrame()
        y_cols = ['eid'] + [f'bl2{outcome}_yrs' for outcome in self.disease_list]
        e_cols = ['eid'] + [f'{outcome}' for outcome in self.disease_list]
        y_df = self.outcomes_df[self.outcomes_df['eid'].isin(eids_series)][y_cols].reset_index(drop=True)
        e_df = self.outcomes_df[self.outcomes_df['eid'].isin(eids_series)][e_cols].reset_index(drop=True)
        return y_df, e_df

    def _merge_and_save_split(self, df_features, predictor_set_name, split_name):
        """Merges features (X), outcome times (y), and outcome events (e) and saves to a single CSV file."""
        if df_features.empty:
            self.logger.warning(f"Skipping saving for {split_name}_{predictor_set_name}.csv because the input dataframe is empty.")
            return

        filename = f"{split_name}_{predictor_set_name}.csv"
        self.logger.info(f"Merging and saving data for: {filename}")

        # 1. Get X data (select relevant feature columns from the input dataframe)
        relevant_cols = self.predictor_set_mapping.get(predictor_set_name, [])
        cols_to_select = ['eid'] + [col for col in relevant_cols if col in df_features.columns]
        X_df = df_features[list(dict.fromkeys(cols_to_select))]

        # 2. Get y and e data
        y_df, e_df = self._get_y_e_sets(df_features['eid'])

        # 3. Merge X, y, and e DataFrames
        if y_df.empty or e_df.empty:
            self.logger.warning(f"y or e dataframe is empty for {filename}. Saving features only.")
            merged_df = X_df
        else:
            merged_df = pd.merge(X_df, y_df, on='eid', how='inner')
            merged_df = pd.merge(merged_df, e_df, on='eid', how='inner')

        # 4. Save the final merged dataframe to a CSV file
        file_path = join(self.split_seed_dir, filename)
        merged_df.to_csv(file_path, index=False)
        self.logger.info(f"Saved merged dataset to {file_path}")
        
    def _process_and_save_datasets(self):
        self.logger.info("\n" + "="*80)
        self.logger.info("Starting Final Data Saving Stage (Merged Raw CSVs)")
        self.logger.info("="*80)

        for p_set in self.predictor_set_mapping.keys():
            self.logger.info(f"\n--- Processing and saving data for: {p_set} ---")

            if p_set in ['Clinical', 'Genomics']:
                pool_df = self.excluded_clinical_pool if p_set == 'Clinical' else self.excluded_genomics_pool
                self.logger.info(f"Saving entire data pool for {p_set}...")
                self._merge_and_save_split(pool_df, p_set, 'cohort')
                
                self.logger.info(f"Saving external_test split for {p_set}...")
                external_test_df = self.X_external_test_main_df
                if p_set == 'Genomics':
                     self.logger.info(f"Merging clinical data into external_test split for {p_set}...")
                     external_test_df = pd.merge(self.X_external_test_main_df.copy(), self.merge_clinical_df, on='eid', how='left')
                self._merge_and_save_split(external_test_df, p_set, 'external_test')
                
                continue
            
            self.logger.info(f"Processing train/val/internal_test/external_test splits for {p_set}...")
            df_train_source = getattr(self, f"X_train_{p_set.lower()}_df")
            df_val_source = getattr(self, f"X_val_{p_set.lower()}_df")
            df_internal_test_source = getattr(self, f"X_internal_test_{p_set.lower()}_df")
            df_external_test_source = self.X_external_test_main_df

            splits_to_process = {
                'train': df_train_source,
                'val': df_val_source,
                'internal_test': df_internal_test_source,
                'external_test': df_external_test_source
            }

            for split_name, df_split in splits_to_process.items():
                self.logger.info(f"Merging clinical data into {split_name} split for {p_set}...")
                df_for_saving = pd.merge(df_split, self.merge_clinical_df, on='eid', how='left')
                self._merge_and_save_split(df_for_saving, p_set, split_name)


if __name__ == '__main__':
    data_dir = '/your path/cardiomicscore/data/'
    log_dir = '/your path/cardiomicscore/saved/log'
    
    val_size = 0.1
    shuffle = True
    seed = 250901

    try:
        data_preparer = UKBiobankData(data_dir=data_dir, log_dir=log_dir, val_size=val_size, shuffle=shuffle, seed=seed)
        data_preparer.logger.info('------------------ All UKBiobank Datasets have been processed and saved --------------------')
    except Exception as e:
        print(f"An unhandled error occurred during data preparation: {e}")
        if 'data_preparer' in locals() and hasattr(data_preparer, 'logger'):
             data_preparer.logger.error(f"An unhandled error occurred: {e}", exc_info=True)
        print('------------------ UKBiobank Data Preparation FAILED --------------------')