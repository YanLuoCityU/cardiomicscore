import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import logging
import time
from collections import defaultdict
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix

# --- Project Setup ---
project_path = '/your path/cardiomicscore'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, 'data_loader'))

from data_loader import UKBiobankDataLoader
from model.model import OmicsNet


# ==========================================================================================
# --- Trainer Classes (Self-Contained for this Script with Full Logging) ---
# ==========================================================================================

class BaseTrainer:
    def __init__(self, config, model, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, 
                 logger, writer, outcome_column_indices):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        
        self.logger = logger
        self.writer = writer
        
        self.outcome_column_indices = outcome_column_indices
        self.epochs = self.config['trainer']['epochs']
        self.early_stop = self.config['trainer']['early_stop']

    def train(self, trial: optuna.trial.Trial):
        best_cindex = -1
        not_improved_count = 0

        for epoch in range(1, self.epochs + 1):
            avg_train_loss, train_outputs, train_y, train_e = self._run_epoch(self.train_dataloader, is_training=True)
            avg_val_loss, val_outputs, val_y, val_e = self._run_epoch(self.validate_dataloader, is_training=False)

            self._log_losses(avg_train_loss, avg_val_loss, epoch)
            
            train_cindex, mean_train_cindex = self.get_cindex_from_outputs(train_outputs, train_y, train_e)
            val_cindex, mean_val_cindex = self.get_cindex_from_outputs(val_outputs, val_y, val_e)
            
            self._log_cindex(train_cindex, val_cindex, mean_train_cindex, mean_val_cindex, epoch)
            
            if epoch % 5 == 0 or self.early_stop - not_improved_count <= 1:
                self._log_confusion_matrices(val_outputs, val_e, epoch)
                
            trial.report(mean_val_cindex, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if mean_val_cindex > best_cindex:
                best_cindex = mean_val_cindex
                not_improved_count = 0
            else:
                not_improved_count += 1
                if not_improved_count >= self.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}. Best Val C-index: {best_cindex:.4f}.")
                    break
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if epoch % 1 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train C-idx={mean_train_cindex:.4f}, Val C-idx={mean_val_cindex:.4f}")
        
        return best_cindex

    def _run_epoch(self, dataloader, is_training):
        if is_training: self.model.train()
        else: self.model.eval()

        total_loss = 0
        all_outputs_dict, all_y_list, all_e_list = defaultdict(list), [], []
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch in dataloader:
                _, outputs, y, e = self._model_batch(batch, self.model)
                loss_dict = self._calculate_loss_dict(outputs, y, e)
                loss = self._calculate_loss(loss_dict)
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                for key, value in outputs.items(): all_outputs_dict[key].append(value.detach().cpu())
                all_y_list.append(y.cpu())
                all_e_list.append(e.cpu())
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        final_outputs = {key: torch.cat(val) for key, val in all_outputs_dict.items()}
        final_y = torch.cat(all_y_list)
        final_e = torch.cat(all_e_list)
        return avg_loss, final_outputs, final_y, final_e

    def _log_confusion_matrices(self, outputs_dict, true_events, epoch):
        self.logger.info(f"--- Confusion Matrices at Epoch {epoch} (Validation Set) ---")
        threshold = 0.0
        for outcome in self.config['outcomes_list']:
            logits, labels = outputs_dict[outcome], true_events[:, self.outcome_column_indices[outcome]]
            predictions = (logits > threshold).int()
            try:
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                cm_str = (f"Outcome: {outcome}\n           Predicted 0 | Predicted 1\n         ---------------------------\n  True 0 |   {tn:^7d}   |   {fp:^7d}   \n  True 1 |   {fn:^7d}   |   {tp:^7d}   \n")
                self.logger.info(cm_str)
            except Exception as e:
                self.logger.warning(f"Could not compute confusion matrix for '{outcome}': {e}")
        self.logger.info("--- End of Confusion Matrices ---")

    def get_cindex_from_outputs(self, outputs_dict, y, e):
        cindex_per_outcome = {}
        for outcome, index in self.outcome_column_indices.items():
            predictions = outputs_dict[outcome].numpy().flatten()
            durations, events = y[:, index].numpy(), e[:, index].numpy()
            cindex = 1 - concordance_index(event_times=durations, predicted_scores=predictions, event_observed=events)
            cindex_per_outcome[outcome] = cindex
        mean_cindex = np.mean(list(cindex_per_outcome.values()))
        return cindex_per_outcome, mean_cindex

    def _log_losses(self, avg_train_loss, avg_val_loss, epoch):
        if self.writer: self.writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)

    def _log_cindex(self, train_cindex, val_cindex, mean_train_cindex, mean_val_cindex, epoch):
        if self.writer:
            self.writer.add_scalars('Cindex_Mean', {'Train': mean_train_cindex, 'Validation': mean_val_cindex}, epoch)
            for outcome in self.config['outcomes_list']:
                self.writer.add_scalars(f'Cindex/{outcome}', {'Train': train_cindex[outcome], 'Validation': val_cindex[outcome]}, epoch)
    
    def _save_checkpoint(self, epoch, best_cindex):
        state = {'epoch': epoch, 'model': self.model.state_dict(), 'best_cindex': best_cindex}
        torch.save(state, self.checkpoint_path)

    def _calculate_loss_dict(self, outputs, y, e): raise NotImplementedError
    def _calculate_loss(self, loss_dict): raise NotImplementedError
    def _model_batch(self, batch, model): raise NotImplementedError

class WeightedMLPTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, 
                 logger, writer, outcome_column_indices,
                 pos_weights, loss_weights):
        super().__init__(config, model, optimizer, lr_scheduler,
                         train_dataloader, validate_dataloader, 
                         logger, writer, outcome_column_indices)
        self.criterions = {o: torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=self.device)) for o, w in pos_weights.items()}
        self.loss_weights = {o: torch.tensor(w, device=self.device) for o, w in loss_weights.items()}

    def _calculate_loss_dict(self, outputs, y, e):
        loss_dict = {}
        for outcome, criterion in self.criterions.items():
            predicted_risk = outputs[outcome]
            event = e[:, self.outcome_column_indices[outcome]].unsqueeze(1)
            loss_dict[outcome] = criterion(predicted_risk, event)
        return loss_dict

    def _calculate_loss(self, loss_dict):
        return sum(self.loss_weights[outcome] * loss for outcome, loss in loss_dict.items())

    def _process_batch(self, batch):
        eid, x_data, y, e = batch
        return (eid, x_data.to(self.device).float(), y.to(self.device).float(), e.to(self.device).float())
            
    def _model_batch(self, batch, model):
        eid, x_data, y, e = self._process_batch(batch)
        outputs = model(omics_data=x_data)
        return eid, outputs, y, e

# ==========================================================================================
# --- Helper Functions & Main Script ---
# ==========================================================================================
def get_pos_weights(e_df, outcomes_list):
    pos_weights = {}
    for outcome in outcomes_list:
        if outcome in e_df.columns:
            neg, pos = (e_df[outcome] == 0).sum(), (e_df[outcome] == 1).sum()
            pos_weights[outcome] = neg / pos if pos > 0 else 1.0
    return pos_weights

config = {
    'name': 'OmicsNet_Tune',
    'data_dir': '/your path/cardiomicscore/data/',
    'model_dir': '/your path/cardiomicscore/saved/models/',
    'log_dir': '/your path/cardiomicscore/saved/log',
    'predictor_set': 'Proteomics',
    'outcomes_list': ['cad', 'stroke', 'hf', 'af', 'pad', 'vte'],
    'model': {
        'omics_input_dim': 2920, 
        'shared_mlp_kwargs': {
            'snn_init': False, 'dropout_fn': 'nn.Dropout', 
            'norm_fn': 'nn.BatchNorm1d', 'norm_layer': 'all', 'input_norm': False,
            'final_norm': True, 'final_dropout': True,
        },
        'skip_connection_mlp_kwargs_default': {
            'norm_fn': 'nn.BatchNorm1d', 'norm_layer': 'all', 'input_norm': False,
            'dropout_fn': 'nn.Dropout', 'final_norm': True, 'final_dropout': True
        },
        'predictor_mlp_kwargs_default': {
            'output_dim': 1, 'dropout_fn': 'nn.Dropout', 'final_activation': None,
            'final_norm': False, 'final_dropout': False,
            'norm_fn': 'nn.BatchNorm1d', 'norm_layer': 'all', 'input_norm': False
        }
    },
    'data_loader': {
        'batch_size': {'train': 128, 'val': 128}, 'shuffle_train': True,
        'num_workers': 8, 'seed_to_split': 250901,
    },
    'trainer': {'epochs': 50, 'early_stop': 50, 'lr': 1e-5, 'weight_decay': 0},
    'loss': {'loss_weights': {outcome: 1.0 for outcome in ['cad', 'stroke', 'hf', 'af',  'pad', 'vte']}},
    'seed': 25090203
}

# --- 1. Define the dictionary of possible dimension sequences ---
all_dim_sequences = {
    "s0_o32": [32],
    "s1_o32": [64, 32],
    "s2_o32": [128, 64, 32],
    "s3_o32": [256, 128, 64, 32],
    "s4_o32": [512, 256, 128, 64, 32],

    "s0_o64": [64],
    "s1_o64": [128, 64],
    "s2_o64": [256, 128, 64],
    "s3_o64": [512, 256, 128, 64],

    "s0_o128": [128],
    "s1_o128": [256, 128],
    "s2_o128": [512, 256, 128],

    "s0_o256": [256],
    "s1_o256": [512, 256]
}

# --- 2. Define a helper function to parse the sequence ---
def parse_dim_sequence(full_dims_list: list):
    """Splits a list into hidden_dims (all but last) and output_dim (last)."""
    if not full_dims_list:
        if len(full_dims_list) == 1:
            return full_dims_list[0], []
        else:
            raise ValueError("Sequence for MLP must have at least 1 element.")
    if len(full_dims_list) >= 2:
        output_d = full_dims_list[-1]
        hidden_ds = full_dims_list[:-1]
        return output_d, hidden_ds
    else:
        raise ValueError("Invalid sequence format.")
    
def objective(trial: optuna.trial.Trial) -> float:
    trial_config = copy.deepcopy(config)
    trial_config['trial_number'] = trial.number
    
    # Create a filtered list of keys for MLPs that must have at least 2 hidden layers ---
    dim_sequences_keys_for_shared_and_skip = [
        key for key in all_dim_sequences.keys() if key.startswith('s2_') or key.startswith('s3_') or key.startswith('s4_')
    ]
    all_dim_sequences_keys_for_predictor = list(all_dim_sequences.keys())

    # --- 3. Suggest hyperparameters using the new constrained approach ---
    activation = trial.suggest_categorical('activation', ['nn.ReLU', 'nn.LeakyReLU', 'nn.ELU', 'nn.Tanh'])
    
    # Shared MLP parameters (must have >= 2 hidden layers)
    shared_mlp_dims_key = trial.suggest_categorical('shared_mlp_dims_choice', dim_sequences_keys_for_shared_and_skip)
    shared_mlp_output_dim, shared_mlp_hidden_dims = parse_dim_sequence(all_dim_sequences[shared_mlp_dims_key])
    
    trial_config['model']['shared_mlp_kwargs'].update({
        'output_dim': shared_mlp_output_dim,
        'hidden_dims': shared_mlp_hidden_dims,
        'dropout': trial.suggest_float('shared_mlp_dropout', 0.1, 0.7, step=0.1),
        'activation': activation, 
        'final_activation': activation
    })
    
    # Skip Connection MLP parameters (must have >= 2 hidden layers)
    skip_mlp_dims_key = trial.suggest_categorical('skip_mlp_dims_choice', dim_sequences_keys_for_shared_and_skip)
    skip_mlp_output_dim, skip_mlp_hidden_dims = parse_dim_sequence(all_dim_sequences[skip_mlp_dims_key])

    trial_config['model']['skip_connection_mlp_kwargs_default'].update({
        'output_dim': skip_mlp_output_dim,
        'hidden_dims': skip_mlp_hidden_dims,
        'dropout': trial.suggest_float('skip_connection_mlp_dropout', 0.1, 0.7, step=0.1),
        'activation': activation, 
        'final_activation': activation
    })

    # Predictor MLP parameters (can be any structure)
    predictor_mlp_dims_key = trial.suggest_categorical('predictor_mlp_dims_choice', all_dim_sequences_keys_for_predictor)
    # For the predictor, the whole sequence is used as hidden layers, as the final output_dim is fixed to 1.
    predictor_mlp_hidden_dims = all_dim_sequences[predictor_mlp_dims_key]

    trial_config['model']['predictor_mlp_kwargs_default'].update({
        'hidden_dims': predictor_mlp_hidden_dims,
        'dropout': trial.suggest_float('predictor_mlp_dropout', 0.1, 0.7, step=0.1),
        'activation': activation
    })

    # You can still tune learning rate and weight decay if you wish
    # trial_config['trainer'].update({
    #     'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
    #     'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    # })
    
    logger.info(f"\n--- Starting Trial {trial.number} ---")
    logger.info(f"Params: {trial.params}")
    
    model = OmicsNet(
        omics_input_dim=trial_config['model']['omics_input_dim'],
        outcomes_list=trial_config['outcomes_list'],
        shared_mlp_kwargs=trial_config['model']['shared_mlp_kwargs'],
        skip_connection_mlp_kwargs_default=trial_config['model']['skip_connection_mlp_kwargs_default'],
        predictor_mlp_kwargs_default=trial_config['model']['predictor_mlp_kwargs_default']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_config['trainer']['lr'], weight_decay=trial_config['trainer']['weight_decay'])
    trainer = WeightedMLPTrainer(
        config=trial_config, model=model, optimizer=optimizer, lr_scheduler=None,
        train_dataloader=train_dataloader, validate_dataloader=validate_dataloader,
        logger=logger, writer=None, outcome_column_indices=outcome_column_indices,
        pos_weights=pos_weights, loss_weights=trial_config['loss']['loss_weights']
    )
    
    best_val_cindex = trainer.train(trial)
    return best_val_cindex

if __name__ == '__main__':
    script_start_time = time.time()
    
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    log_dir = os.path.join(config['log_dir'], "HyperparameterTuning")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"tune_{config['predictor_set']}_{SEED}.log")
    
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    logger.info("Loading data once for all trials...")
    dataloader_manager = UKBiobankDataLoader(
        data_dir=config['data_dir'], seed_to_split=config['data_loader']['seed_to_split'],
        batch_size=config['data_loader']['batch_size'], shuffle_train=config['data_loader']['shuffle_train'],
        num_workers=config['data_loader']['num_workers'], logger=logger
    )
    
    train_dataloader = dataloader_manager.get_dataloader('train', config['predictor_set'], config['outcomes_list'])
    validate_dataloader = dataloader_manager.get_dataloader('val', config['predictor_set'], config['outcomes_list'])

    e_train_file = os.path.join(dataloader_manager.split_data_path, f"e_train_{config['predictor_set']}.feather")
    e_train_df = pd.read_feather(e_train_file)
    pos_weights = get_pos_weights(e_train_df, config['outcomes_list'])
    outcome_column_indices = {outcome: i for i, outcome in enumerate(config['outcomes_list'])}
    
    logger.info("Calculating loss weights as the reciprocal of positive sample counts to prioritize rarer outcomes.")
    positive_counts = e_train_df[config['outcomes_list']].sum()
    initial_loss_weights = {outcome: 1.0 / count if count > 0 else 1.0 for outcome, count in positive_counts.items()}
    logger.info(f"Initial (un-normalized) Loss Weights: {initial_loss_weights}")
    total_weight = sum(initial_loss_weights.values())
    logger.info(f"Sum of all weights: {total_weight}")
    normalized_loss_weights = {outcome: weight / total_weight for outcome, weight in initial_loss_weights.items()}
    config['loss']['loss_weights'] = normalized_loss_weights
    logger.info(f"Calculated Normalized Loss Weights: {config['loss']['loss_weights']}")
    logger.info(f"Sum of normalized weights: {sum(config['loss']['loss_weights'].values())}")
    
    study_name = f"OmicsNet-Tuning-{config['predictor_set']}-{SEED}"
    storage_path = f"sqlite:///{log_dir}/{study_name}.db"
    
    study = optuna.create_study(
        direction='maximize', study_name=study_name, storage=storage_path,
        load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Database stored at: {storage_path}")
    study.optimize(objective, n_trials=100)

    logger.info("--- Tuning Complete ---")
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"  Value (Val C-index): {study.best_trial.value}")
    logger.info("  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    - {key}: {value}")
        
    logger.info("\n--- Building and Displaying Best Model Architecture ---")

    # 1. Retrieve the dictionary of best hyperparameters
    best_params = study.best_trial.params
    
    # Shared MLP kwargs
    shared_mlp_dims_key = best_params['shared_mlp_dims_choice']
    shared_mlp_output_dim, shared_mlp_hidden_dims = parse_dim_sequence(all_dim_sequences[shared_mlp_dims_key])
    best_shared_mlp_kwargs = copy.deepcopy(config['model']['shared_mlp_kwargs'])
    best_shared_mlp_kwargs.update({
        'output_dim': shared_mlp_output_dim,
        'hidden_dims': shared_mlp_hidden_dims,
        'dropout': best_params['shared_mlp_dropout'],
        'activation': best_params['activation'],
        'final_activation': best_params['activation']
    })
    
    # Skip Connection MLP kwargs
    skip_mlp_dims_key = best_params['skip_mlp_dims_choice']
    skip_mlp_output_dim, skip_mlp_hidden_dims = parse_dim_sequence(all_dim_sequences[skip_mlp_dims_key])
    best_skip_connection_mlp_kwargs = copy.deepcopy(config['model']['skip_connection_mlp_kwargs_default'])
    best_skip_connection_mlp_kwargs.update({
        'output_dim': skip_mlp_output_dim,
        'hidden_dims': skip_mlp_hidden_dims,
        'dropout': best_params['skip_connection_mlp_dropout'],
        'activation': best_params['activation'],
        'final_activation': best_params['activation']
    })

    # Predictor MLP kwargs
    predictor_mlp_dims_key = best_params['predictor_mlp_dims_choice']
    predictor_mlp_hidden_dims = all_dim_sequences[predictor_mlp_dims_key]
    best_predictor_mlp_kwargs = copy.deepcopy(config['model']['predictor_mlp_kwargs_default'])
    best_predictor_mlp_kwargs.update({
        'hidden_dims': predictor_mlp_hidden_dims,
        'dropout': best_params['predictor_mlp_dropout'],
        'activation': best_params['activation']
    })

    best_model = OmicsNet(
        omics_input_dim=config['model']['omics_input_dim'],
        outcomes_list=config['outcomes_list'],
        shared_mlp_kwargs=best_shared_mlp_kwargs,
        skip_connection_mlp_kwargs_default=best_skip_connection_mlp_kwargs,
        predictor_mlp_kwargs_default=best_predictor_mlp_kwargs
    )
    logger.info(f"Best model architecture:\n{best_model}")

    # --- End of the script ---
    total_elapsed_time = time.time() - script_start_time
    logger.info(f"\n--- Total Script Execution Time: {total_elapsed_time / 60:.2f} minutes ---")