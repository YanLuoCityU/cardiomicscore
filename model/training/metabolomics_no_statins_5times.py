import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import time
import shap
import copy # 1. Import the copy module
from tensorboardX import SummaryWriter

# --- Project Setup ---
project_path = '/your path/cardiomicscore'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, 'data_loader'))

from data_loader import UKBiobankDataLoader
from model.model import OmicsNet
from model.trainer import WeightedMLPTrainer

# ==========================================================================================
# --- Helper Classes & Functions ---
# ==========================================================================================

class OutcomeSpecificNet(nn.Module):
    """Wrapper to make the multi-task OmicsNet compatible with SHAP for a single outcome."""
    def __init__(self, base_model, outcome):
        super(OutcomeSpecificNet, self).__init__()
        self.shared_mlp = base_model.shared_mlp
        self.outcome_mlp = base_model.output_layers[outcome]

    def forward(self, omics_data=None):
        shared_fts = self.shared_mlp(omics_data)
        outcome_output = self.outcome_mlp(features=shared_fts, covariates=omics_data)
        return outcome_output

def get_pos_weights(e_df, outcomes_list):
    pos_weights = {}
    for outcome in outcomes_list:
        if outcome in e_df.columns:
            neg, pos = (e_df[outcome] == 0).sum(), (e_df[outcome] == 1).sum()
            pos_weights[outcome] = neg / pos if pos > 0 else 1.0
    return pos_weights

def generate_background_samples(dataloader, num_samples=200):
    """Generates a background dataset for SHAP from the training loader."""
    background_data = []
    count = 0
    for _, x_data, _, _ in dataloader:
        background_data.append(x_data)
        count += x_data.shape[0]
        if count >= num_samples:
            break
    return torch.cat(background_data, dim=0)[:num_samples].to('cpu') if background_data else torch.tensor([])


# ==========================================================================================
# --- 2. Encapsulate the single run logic into a function ---
# ==========================================================================================
def run_experiment(config, run_seed):
    """
    Encapsulates the logic for a single training and evaluation run.
    """
    # 2a. Setup Environment for the current run using run_seed
    config['seed'] = run_seed # Update config for this specific run
    SEED = run_seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create a unique log directory for each seed
    log_dir = os.path.join(config['log_dir'], config['name'], config['predictor_set'], str(SEED))
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging to file and console
    # To prevent multiple handlers being added in a loop, we get a logger with a unique name
    logger = logging.getLogger(f"run_{SEED}")
    logger.setLevel(logging.INFO)
    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    log_filename = os.path.join(log_dir, f"training_log.log")
    file_handler = logging.FileHandler(log_filename, mode='w')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    writer = SummaryWriter(log_dir)
    start_time = time.time()
    logger.info(f"--- Experiment starting with SEED: {SEED} ---")
    
    # 2b. Load Data
    logger.info("--- Loading Data ---")
    dataloader_manager = UKBiobankDataLoader(
        data_dir=config['data_dir'], seed_to_split=config['data_loader']['seed_to_split'],
        batch_size=config['data_loader']['batch_size'], shuffle_train=config['data_loader']['shuffle_train'],
        num_workers=config['data_loader']['num_workers'], logger=logger
    )
    
    feature_names = dataloader_manager.get_feature_names(config['predictor_set'])
    if not feature_names: raise ValueError("Could not get feature names.")
    config['model']['omics_input_dim'] = len(feature_names)
    logger.info(f"Set model input_dim to {len(feature_names)} for {config['predictor_set']}")

    train_loader = dataloader_manager.get_dataloader('train', config['predictor_set'], config['outcomes_list'])
    val_loader = dataloader_manager.get_dataloader('val', config['predictor_set'], config['outcomes_list'])
    internal_test_loader = dataloader_manager.get_dataloader('internal_test', config['predictor_set'], config['outcomes_list'])
    external_test_loader = dataloader_manager.get_dataloader('external_test', config['predictor_set'], config['outcomes_list'])
    
    e_train_file = os.path.join(dataloader_manager.split_data_path, f"e_train_{config['predictor_set']}.feather")
    pos_weights = get_pos_weights(pd.read_feather(e_train_file), config['outcomes_list'])
    positive_counts = pd.read_feather(e_train_file)[config['outcomes_list']].sum()
    initial_loss_weights = {outcome: 1.0 / count if count > 0 else 1.0 for outcome, count in positive_counts.items()}
    logger.info(f"Initial (un-normalized) Loss Weights: {initial_loss_weights}")
    total_weight = sum(initial_loss_weights.values())
    logger.info(f"Sum of all weights: {total_weight}")
    normalized_loss_weights = {outcome: weight / total_weight for outcome, weight in initial_loss_weights.items()}
    logger.info(f"Calculated Normalized Loss Weights: {normalized_loss_weights}")
    logger.info(f"Sum of normalized weights: {sum(normalized_loss_weights.values())}")
    outcome_indices = {outcome: i for i, outcome in enumerate(config['outcomes_list'])}

    # 2c. Initialize Model, Optimizer, and Trainer
    logger.info("--- Initializing Model and Trainer ---")
    model = OmicsNet(
        omics_input_dim=config['model']['omics_input_dim'],
        outcomes_list=config['outcomes_list'],
        shared_mlp_kwargs=config['model']['shared_mlp_kwargs'],
        skip_connection_mlp_kwargs_default=config['model']['skip_connection_mlp_kwargs_default'],
        predictor_mlp_kwargs_default=config['model']['predictor_mlp_kwargs_default']
    )
    logger.info(f"Model architecture: {model}\n")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['trainer']['lr'], weight_decay=config['trainer']['weight_decay'])

    trainer = WeightedMLPTrainer(
        config=config, model=model, optimizer=optimizer, lr_scheduler=None,
        train_dataloader=train_loader, validate_dataloader=val_loader,
        test_dataloader_internal=internal_test_loader, test_dataloader_external=external_test_loader,
        logger=logger, writer=writer, outcome_column_indices=outcome_indices,
        pos_weights=pos_weights, loss_weights=normalized_loss_weights
    )

    # 2d. Train the Model
    logger.info("\n" + "="*80 + "\n--- Starting Final Model Training ---\n" + "="*80)
    trainer.train()
    logger.info("--- Training Complete ---")

    # 2e. Final Evaluation
    trainer.test()
    
    # 2f. Generate SHAP values for EXTERNAL TEST SET
    logger.info("\n" + "="*80 + "\n--- Generating Scores & SHAP Values ---\n" + "="*80)
    
    checkpoint = torch.load(trainer.checkpoint_path, map_location=trainer.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    results_dir = config['results_dir']

    logger.info("\n--- Preparing for SHAP analysis on External Test Set ---")
    background_data = generate_background_samples(train_loader, num_samples=500)
    
    all_test_samples_unfiltered = [x_data for _, x_data, _, _ in external_test_loader]
    test_data_tensor_unfiltered = torch.cat(all_test_samples_unfiltered, dim=0).to('cpu')
    external_test_eids_unfiltered = torch.cat([eid for eid, _, _, _ in external_test_loader]).numpy()

    shap_test_data_tensor = None
    shap_test_eids = None

    panel_path = os.path.join(dataloader_manager.split_data_path, "X_external_test_PANEL.feather")
    try:
        logger.info(f"Loading lipid treatment status for filtering from: {panel_path}")
        panel_df = pd.read_feather(panel_path)
        if 'eid' not in panel_df.columns or 'lipidlower_1.0' not in panel_df.columns:
            raise KeyError("'eid' or 'lipidlower_1.0' not found in the panel DataFrame.")
        eids_no_treatment = panel_df[panel_df['lipidlower_1.0'] == 0]['eid']
        logger.info(f"Identified {len(eids_no_treatment)} individuals in the external test panel not on lipid-lowering therapy.")
        filter_mask = np.isin(external_test_eids_unfiltered, eids_no_treatment)
        shap_test_data_tensor = test_data_tensor_unfiltered[filter_mask]
        shap_test_eids = external_test_eids_unfiltered[filter_mask]
        original_count = len(external_test_eids_unfiltered)
        filtered_count = len(shap_test_eids)
        if filtered_count == 0:
            logger.warning("No samples left for SHAP analysis after filtering. SKIPPING SHAP CALCULATION.")
            shap_test_data_tensor = None
        else:
            logger.info(f"SHAP analysis will proceed on {filtered_count} of {original_count} samples (those without lipid-lowering therapy).")

    except (FileNotFoundError, KeyError) as e:
        logger.error(f"CRITICAL: Could not load or parse lipid treatment status from {panel_path}. Error: {e}")
        logger.error("SKIPPING SHAP analysis due to filtering failure.")

    if shap_test_data_tensor is not None and shap_test_eids is not None:
        shap_dir = os.path.join(results_dir, "SHAP", config['name'], config['predictor_set'], str(SEED))
        os.makedirs(shap_dir, exist_ok=True)

        for outcome in config['outcomes_list']:
            logger.info(f"Calculating SHAP values for outcome: {outcome}...")
            model_outcome = OutcomeSpecificNet(base_model=model, outcome=outcome).to('cpu')
            explainer = shap.DeepExplainer(model_outcome, background_data)
            shap_values = explainer.shap_values(shap_test_data_tensor, check_additivity=False)
            shap_values_2d = np.squeeze(shap_values, axis=-1)
            shap_values_32bit = shap_values_2d.astype('float32')
            shap_df = pd.DataFrame(shap_values_32bit, columns=feature_names)
            if len(shap_test_eids) == len(shap_df):
                shap_df.insert(0, 'eid', shap_test_eids)
                logger.info(f"Successfully added 'eid' column to SHAP DataFrame for {outcome}.")
            else:
                logger.warning(f"Could not add 'eid' column for {outcome}: Mismatch in length between eids ({len(shap_test_eids)}) and SHAP rows ({len(shap_df)}).")
            
            shap_filename = os.path.join(shap_dir, f"shap_{outcome}.parquet")
            shap_df.to_parquet(shap_filename, engine='pyarrow', compression='snappy')
            logger.info(f"Saved SHAP values for {outcome} to {shap_filename}")
    else:
        logger.info("Skipping SHAP value generation as no valid data was available after filtering.")


    end_time = time.time()
    logger.info(f"\nTotal script execution time for SEED {SEED}: {(end_time - start_time)/60:.2f} minutes")
    logger.info(f"--- SCRIPT FINISHED FOR SEED {SEED} ---")

# ==========================================================================================
# --- 3. Main block, responsible for the loop execution ---
# ==========================================================================================
if __name__ == '__main__':
    # 3a. Load base configuration from JSON
    config_path = '/your path/cardiomicscore/config/Metabolomics_no_statins/final_param.json'
    try:
        with open(config_path, 'r') as f:
            base_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
        
    # 3b. Get the base SEED, exit if it doesn't exist
    base_seed = base_config.get('seed')
    if base_seed is None:
        print("Error: 'seed' not found in config file. Please define a base seed.")
        sys.exit(1)

    # 3c. Loop for 5 runs
    for i in range(5):
        run_seed = base_seed + i
        print("\n" + "="*100)
        print(f"--- STARTING RUN {i+1}/5 WITH SEED: {run_seed} ---")
        print("="*100 + "\n")

        # Use deepcopy to ensure each run's configuration is independent
        run_config = copy.deepcopy(base_config)
        
        # Call the experiment function
        run_experiment(config=run_config, run_seed=run_seed)

        print(f"\n--- FINISHED RUN {i+1}/5 WITH SEED: {run_seed} ---\n")