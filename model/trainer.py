import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score

class BaseTrainer:
    def __init__(self, config, model, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, test_dataloader_internal, test_dataloader_external, 
                 logger, writer, outcome_column_indices):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader_internal = test_dataloader_internal
        self.test_dataloader_external = test_dataloader_external
        
        self.logger = logger
        self.writer = writer
        
        self.outcome_column_indices = outcome_column_indices
        self.epochs = self.config['trainer']['epochs']
        self.early_stop = self.config['trainer']['early_stop']
        
        model_dir = f"{self.config['model_dir']}/{self.config['name']}/{self.config['predictor_set']}"
        os.makedirs(model_dir, exist_ok=True)
        self.checkpoint_path = f"{model_dir}/{self.config['predictor_set']}_model_{self.config['seed']}.pth"

    def train(self):
        best_cindex = -1
        not_improved_count = 0
        for epoch in range(1, self.epochs + 1):
            avg_train_loss, avg_train_loss_dict, train_outputs, train_y, train_e = self._run_epoch(self.train_dataloader, is_training=True)
            avg_val_loss, avg_val_loss_dict, val_outputs, val_y, val_e = self._run_epoch(self.validate_dataloader, is_training=False)

            train_metrics = self.calculate_all_metrics(train_outputs, train_y, train_e)
            val_metrics = self.calculate_all_metrics(val_outputs, val_y, val_e)

            self._log_losses(avg_train_loss, avg_val_loss, avg_train_loss_dict, avg_val_loss_dict, epoch)
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Log confusion matrix every 10 epochs
            if epoch % 5 == 0:
                self._log_confusion_matrices(val_outputs, val_e, epoch)

            # Early stopping logic
            mean_val_cindex = val_metrics['C-Index_mean']
            if mean_val_cindex > best_cindex:
                best_cindex = mean_val_cindex
                not_improved_count = 0
                self._save_checkpoint(epoch, best_cindex)
            else:
                not_improved_count += 1
                if not_improved_count >= self.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}. Best Val C-index: {best_cindex:.4f}.")
                    break
            
            # Updated epoch summary log
            if epoch % 1 == 0:
                 self.logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.3e}, Val Loss={avg_val_loss:.3e}, "
                                  f"Train AUC={train_metrics['AUC_mean']:.4f}, Val AUC={val_metrics['AUC_mean']:.4f}, "
                                  f"Train C-idx={train_metrics['C-Index_mean']:.4f}, Val C-idx={mean_val_cindex:.4f}")
        
        return best_cindex

    def test(self):
        self.logger.info("\n" + "="*80 + "\n--- Starting Final Test Evaluation ---\n" + "="*80)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with Val C-index {checkpoint['best_cindex']:.4f}.")
        except FileNotFoundError:
            self.logger.error(f"Could not find checkpoint at {self.checkpoint_path}!")
            return None, None
            
        self.logger.info("\n--- Internal Test Set (Scotland) Results ---")
        internal_metrics = self._evaluate_on_test_set(self.test_dataloader_internal)

        self.logger.info("\n--- External Test Set Results ---")
        external_metrics = self._evaluate_on_test_set(self.test_dataloader_external)
        
        return internal_metrics, external_metrics

    def _evaluate_on_test_set(self, dataloader):
        if not dataloader:
            self.logger.warning("Test dataloader is None, skipping evaluation.")
            return None
        
        avg_test_loss, _, test_outputs, test_y, test_e = self._run_epoch(dataloader, is_training=False)
        test_metrics = self.calculate_all_metrics(test_outputs, test_y, test_e)
        
        self.logger.info(f"  - Average Loss: {avg_test_loss:.3e}")
        self.logger.info(f"  - Mean AUC: {test_metrics['AUC_mean']:.4f}")
        self.logger.info(f"  - Mean C-index: {test_metrics['C-Index_mean']:.4f}")
        self.logger.info("\n  - Metrics per outcome:")
        
        header = f"{'Outcome':<10} | {'AUC':^10} | {'C-Index':^10} | {'PRAUC':^10} | {'Precision':^10} | {'Recall':^10}"
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for outcome in self.config['outcomes_list']:
            row = (f"{outcome:<10} | "
                   f"{test_metrics[outcome]['AUC']:.4f}{'':^4} | "
                   f"{test_metrics[outcome]['C-Index']:.4f}{'':^3} | "
                   f"{test_metrics[outcome]['PRAUC']:.4f}{'':^3} | "
                   f"{test_metrics[outcome]['Precision']:.4f}{'':^1} | "
                   f"{test_metrics[outcome]['Recall']:.4f}{'':^4}")
            self.logger.info(row)
        
        self.logger.info("\n  - Confusion Matrices:")
        self._log_confusion_matrices(test_outputs, test_e)
        return test_metrics

    def _run_epoch(self, dataloader, is_training):
        if is_training: self.model.train()
        else: self.model.eval()
        total_loss, all_outputs_dict, all_y_list, all_e_list = 0, defaultdict(list), [], []
        cumulative_loss_dict = {outcome: 0.0 for outcome in self.config['outcomes_list']}
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
                for outcome, l in loss_dict.items():
                    cumulative_loss_dict[outcome] += l.item()
                for k, v in outputs.items(): all_outputs_dict[k].append(v.detach().cpu())
                all_y_list.append(y.cpu())
                all_e_list.append(e.cpu())
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_dict = {o: l / num_batches for o, l in cumulative_loss_dict.items()} if num_batches > 0 else cumulative_loss_dict
        
        final_outputs = {k: torch.cat(v) for k, v in all_outputs_dict.items()}
        final_y = torch.cat(all_y_list)
        final_e = torch.cat(all_e_list)
        return avg_loss, avg_loss_dict, final_outputs, final_y, final_e
        
    def _log_confusion_matrices(self, outputs_dict, true_events, epoch=None):
        log_header = f"--- Confusion Matrices"
        log_header += f" at Epoch {epoch} (Validation Set) ---" if epoch else " (Test Set) ---"
        self.logger.info(log_header)
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
    
    def calculate_all_metrics(self, outputs_dict, y, e):
        metrics = {}
        all_aucs, all_cindices = [], []
        threshold = 0.0
        for outcome, index in self.outcome_column_indices.items():
            scores = outputs_dict[outcome].numpy().flatten()
            durations, events = y[:, index].numpy(), e[:, index].numpy()
            binary_preds = (scores > threshold).astype(int)
            
            metrics[outcome] = {
                'C-Index': 1 - concordance_index(durations, scores, events),
                'AUC': roc_auc_score(events, scores),
                'PRAUC': average_precision_score(events, scores),
                'Precision': precision_score(events, binary_preds, zero_division=0),
                'Recall': recall_score(events, binary_preds, zero_division=0)
            }
            all_aucs.append(metrics[outcome]['AUC'])
            all_cindices.append(metrics[outcome]['C-Index'])
        
        metrics['AUC_mean'] = np.mean(all_aucs)
        metrics['C-Index_mean'] = np.mean(all_cindices)
        return metrics

    def _save_checkpoint(self, epoch, best_cindex):
        state = {'epoch': epoch, 'model': self.model.state_dict(), 'best_cindex': best_cindex}
        torch.save(state, self.checkpoint_path)
        self.logger.info(f"Saving new best model at epoch {epoch} to {self.checkpoint_path}")

    def _log_losses(self, avg_train_loss, avg_val_loss, avg_train_loss_dict, avg_val_loss_dict, epoch):
        if self.writer:
            self.writer.add_scalars('Loss/Mean_Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
            for outcome in self.config['outcomes_list']:
                self.writer.add_scalars(f'Loss/{outcome}', {'Train': avg_train_loss_dict[outcome], 'Validation': avg_val_loss_dict[outcome]}, epoch)

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        if self.writer:
            self.writer.add_scalars('AUC/Mean_AUC', {'Train': train_metrics['AUC_mean'], 'Validation': val_metrics['AUC_mean']}, epoch)
            self.writer.add_scalars('C-Index/Mean_C-Index', {'Train': train_metrics['C-Index_mean'], 'Validation': val_metrics['C-Index_mean']}, epoch)
            for outcome in self.config['outcomes_list']:
                self.writer.add_scalars(f'AUC/{outcome}', {'Train': train_metrics[outcome]['AUC'], 'Validation': val_metrics[outcome]['AUC']}, epoch)
                self.writer.add_scalars(f'C-Index/{outcome}', {'Train': train_metrics[outcome]['C-Index'], 'Validation': val_metrics[outcome]['C-Index']}, epoch)
                
    def _calculate_loss_dict(self, outputs, y, e): raise NotImplementedError
    def _calculate_loss(self, loss_dict): raise NotImplementedError
    def _model_batch(self, batch, model): raise NotImplementedError

class WeightedMLPTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, test_dataloader_internal, test_dataloader_external,
                 logger, writer, outcome_column_indices, pos_weights, loss_weights):
        super().__init__(config, model, optimizer, lr_scheduler,
                         train_dataloader, validate_dataloader, test_dataloader_internal, test_dataloader_external,
                         logger, writer, outcome_column_indices)
        self.criterions = {o: torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=self.device)) for o, w in pos_weights.items()}
        self.loss_weights = {o: torch.tensor(w, device=self.device) for o, w in loss_weights.items()}

    def _calculate_loss_dict(self, outputs, y, e):
        loss_dict = {}
        for outcome, criterion in self.criterions.items():
            loss_dict[outcome] = criterion(outputs[outcome], e[:, self.outcome_column_indices[outcome]].unsqueeze(1))
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
    
class MLPTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, test_dataloader_internal, test_dataloader_external,
                 logger, writer, outcome_column_indices, loss_weights):
        
        super().__init__(config, model, optimizer, lr_scheduler,
                         train_dataloader, validate_dataloader, test_dataloader_internal, test_dataloader_external,
                         logger, writer, outcome_column_indices)
        
        self.criterions = {outcome: torch.nn.BCEWithLogitsLoss() for outcome in self.config['outcomes_list']}
        self.loss_weights = {o: torch.tensor(w, device=self.device) for o, w in loss_weights.items()}

    def _calculate_loss_dict(self, outputs, y, e):
        loss_dict = {}
        for outcome, criterion in self.criterions.items():
            loss_dict[outcome] = criterion(outputs[outcome], e[:, self.outcome_column_indices[outcome]].unsqueeze(1))
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