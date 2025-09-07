import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
import logging

class UKBiobankDataset(Dataset):
    def __init__(self, data_dir, dataset_type, predictor_set, outcomes_list, logger=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.predictor_set = predictor_set 
        self.outcomes_list = outcomes_list
        self.logger = logger
        
        x_file, y_file, e_file = self._get_filenames()
        
        if not exists(y_file) or not exists(e_file):
            raise FileNotFoundError(f"y/e files not found for {dataset_type}/{predictor_set} at {y_file} or {e_file}")
        
        self.logger.info(f"Loading y data from: {y_file}")
        self.logger.info(f"Loading e data from: {e_file}")
        y_df = pd.read_feather(y_file).set_index('eid')
        e_df = pd.read_feather(e_file).set_index('eid')
        self.master_eids = y_df.index

        y_cols = [f'bl2{outcome}_yrs' for outcome in self.outcomes_list]
        e_cols = [outcome for outcome in self.outcomes_list]
        self.y_data = y_df.loc[self.master_eids, y_cols].values
        self.e_data = e_df.loc[self.master_eids, e_cols].values

        if not exists(x_file):
            raise FileNotFoundError(f"X file not found at: {x_file}")
        
        self.logger.info(f"Loading X data from: {x_file}...")
        x_df_current = pd.read_feather(x_file).set_index('eid')
        self.X_data = x_df_current.reindex(self.master_eids).values
        
        # --- MODIFICATION: Store feature names to avoid re-reading the file ---
        self.feature_names = x_df_current.columns.tolist()
        
        self.logger.info(f"Dataset '{self.dataset_type}' (Predictor: {self.predictor_set}): X:{self.X_data.shape}, y:{self.y_data.shape}, e:{self.e_data.shape}")

    def _get_filenames(self):
        """Helper to determine the correct filenames based on dataset type and predictor set."""
        if self.dataset_type in ['train', 'val', 'internal_test']:
            x_file = join(self.data_dir, f"X_{self.dataset_type}_{self.predictor_set}.feather")
            y_file = join(self.data_dir, f"y_{self.dataset_type}_{self.predictor_set}.feather")
            e_file = join(self.data_dir, f"e_{self.dataset_type}_{self.predictor_set}.feather")
        elif self.dataset_type == 'external_test':
            x_file = join(self.data_dir, f"X_{self.dataset_type}_{self.predictor_set}.feather")
            y_file = join(self.data_dir, "y_external_test.feather")
            e_file = join(self.data_dir, "e_external_test.feather")
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        return x_file, y_file, e_file

    def __len__(self):
        return len(self.master_eids)

    def __getitem__(self, idx):
        eid_sample = self.master_eids[idx]
        x_sample = torch.tensor(self.X_data[idx, :], dtype=torch.float32)
        y_sample = torch.tensor(self.y_data[idx, :], dtype=torch.float32) 
        e_sample = torch.tensor(self.e_data[idx, :], dtype=torch.float32)
        return eid_sample, x_sample, y_sample, e_sample


class UKBiobankDataLoader():
    def __init__(self, 
                 data_dir=None,
                 seed_to_split=None,
                 batch_size=32, 
                 shuffle_train=True,
                 num_workers=0,
                 logger=None):
        self.data_dir = data_dir
        self.seed_to_split = seed_to_split
        self.batch_size_config = batch_size 
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.logger = logger if logger else self._setup_default_logger()
    
        self.split_seed_folder_name = f"split_seed-{self.seed_to_split}"
        self.split_data_path = join(self.data_dir, self.split_seed_folder_name)

        if not exists(self.split_data_path):
            msg = f"Data directory for the specified split seed not found: {self.split_data_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
            
        self.logger.info(f"UKBiobankDataLoader initialized. Using data from: {self.split_data_path}.")

    def _setup_default_logger(self):
        logger = logging.getLogger("UKBiobankDataLoaderDefault")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_dataloader(self, dataset_type, predictor_set, outcomes_list, pin_memory=True):
        self.logger.info(f"\nAttempting to create DataLoader for: type='{dataset_type}', predictor='{predictor_set}'")

        dataset = UKBiobankDataset(
            data_dir=self.split_data_path,
            dataset_type=dataset_type,
            predictor_set=predictor_set,
            outcomes_list=outcomes_list,
            logger=self.logger
        )
        
        resolved_batch_size = 1
        if isinstance(self.batch_size_config, dict):
            if dataset_type in ['train', 'val']:
                resolved_batch_size = self.batch_size_config.get('train', 1)
            # --- MODIFICATION: Catches 'test', 'internal_test', and 'external_test' ---
            elif 'test' in dataset_type:
                test_bs_setting = self.batch_size_config.get('test')
                if test_bs_setting is None: 
                    resolved_batch_size = len(dataset) if len(dataset) > 0 else 1
                else:
                    resolved_batch_size = test_bs_setting
        elif isinstance(self.batch_size_config, int):
            resolved_batch_size = self.batch_size_config
        
        # Ensure batch size is at least 1
        resolved_batch_size = max(1, resolved_batch_size)
        
        current_shuffle_flag = self.shuffle_train if dataset_type == 'train' else False
        
        self.logger.info(f"Creating DataLoader with batch size: {resolved_batch_size}, shuffle: {current_shuffle_flag}")

        dataloader = DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=current_shuffle_flag,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        
        self.logger.info(f"Successfully created DataLoader for '{dataset_type}' ('{predictor_set}') with {len(dataset)} samples.")
        return dataloader
    
    def get_feature_names(self, predictor_set):
        self.logger.info(f"Retrieving feature names for predictor set '{predictor_set}'.")
        try:
            temp_dataset = UKBiobankDataset(
                data_dir=self.split_data_path,
                dataset_type='train',
                predictor_set=predictor_set,
                outcomes_list=['cad'],
                logger=self.logger
            )
            feature_names = temp_dataset.feature_names
            if 'eid' in feature_names:
                feature_names.remove('eid')
            self.logger.info(f"Retrieved {len(feature_names)} feature names for '{predictor_set}'.")
            return feature_names
        except Exception as e:
            self.logger.error(f"Could not retrieve feature names for '{predictor_set}': {e}")
            return []