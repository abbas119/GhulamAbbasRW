# NMT_EEGPT_Project/monitors.py
import torch
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, balanced_accuracy_score)
import numpy as np
import logging # Ensure logging is configured in the main scripts that use this
from glob import glob # For finding latest checkpoint

logger_monitor = logging.getLogger(__name__) # Use a named logger for this module
if not logger_monitor.handlers:
    logger_monitor.setLevel(logging.INFO)
    ch_monitor = logging.StreamHandler()
    ch_monitor.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'))
    logger_monitor.addHandler(ch_monitor)


class TrainingMonitor:
    def __init__(self, model_path, checkpoint_dir, patience, 
                 monitor_metric_name='val_f1_score',
                 metric_mode='max'):
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.monitor_metric_name = monitor_metric_name
        self.metric_mode = metric_mode
        
        if self.metric_mode == 'max':
            self.best_metric_val = -float('inf')
        else:
            self.best_metric_val = float('inf')
            
        self.counter = 0 # Patience counter
        self.best_epoch = 0 # Epoch where best_metric_val was achieved
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger_monitor.info(f"TrainingMonitor initialized. Monitoring '{self.monitor_metric_name}' for '{self.metric_mode}'. Patience: {self.patience}.")
        logger_monitor.info(f"  Best model will be saved to: {self.model_path}")
        logger_monitor.info(f"  Checkpoints will be saved in: {self.checkpoint_dir}")

    def step(self, epoch, model, optimizer, current_metrics_dict, checkpoint_interval=5):
        current_monitored_metric = current_metrics_dict.get(self.monitor_metric_name)
        
        if current_monitored_metric is None:
            logger_monitor.error(f"Monitored metric '{self.monitor_metric_name}' not found in current_metrics_dict. Available: {list(current_metrics_dict.keys())}")
            if 'val_loss' in current_metrics_dict and self.monitor_metric_name != 'val_loss':
                 logger_monitor.warning(f"Falling back to monitor 'val_loss'. Ensure metric_mode is 'min'.")
                 current_monitored_metric = current_metrics_dict['val_loss']
            else:
                 logger_monitor.warning("Cannot proceed with monitoring step without a valid metric.")
                 return False # Cannot determine early stopping or best model

        # Log current epoch's performance (using the logger from the main training script)
        # The main training script's logger should handle this.
        # logger_monitor.info(f"Epoch {epoch+1} results: {current_metrics_dict}")


        # --- Save Periodic Checkpoint ---
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_data = {
                'epoch': epoch + 1, # Save as next epoch to start from if resuming
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_metrics': current_metrics_dict, # Save all current metrics
                'monitor_best_metric_val': self.best_metric_val,
                'monitor_best_epoch': self.best_epoch,
                'monitor_counter': self.counter,
                'monitor_metric_name': self.monitor_metric_name, # Save what was being monitored
                'monitor_metric_mode': self.metric_mode
            }
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                torch.save(checkpoint_data, checkpoint_path)
                logger_monitor.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            except Exception as e:
                logger_monitor.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}", exc_info=True)

        # --- Check for Improvement and Save Best Model ---
        stop_early = False
        improved = False
        if self.metric_mode == 'max':
            if current_monitored_metric > self.best_metric_val:
                improved = True
        else: # min mode
            if current_monitored_metric < self.best_metric_val:
                improved = True
        
        if improved:
            old_best = self.best_metric_val
            self.best_metric_val = current_monitored_metric
            self.best_epoch = epoch # Current epoch (0-indexed)
            self.counter = 0
            try:
                torch.save(model.state_dict(), self.model_path)
                logger_monitor.info(f"Metric improved from {old_best:.4f} to {self.best_metric_val:.4f}. Saved best model at epoch {epoch+1} to {self.model_path}")
            except Exception as e:
                logger_monitor.error(f"Failed to save best model at epoch {epoch+1}: {e}", exc_info=True)
        else:
            self.counter += 1
            logger_monitor.info(f"No improvement for {self.counter} epochs. Best {self.monitor_metric_name} remains {self.best_metric_val:.4f} from epoch {self.best_epoch+1}.")
            
        if self.counter >= self.patience:
            logger_monitor.info(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {self.best_epoch+1} with {self.monitor_metric_name}: {self.best_metric_val:.4f}")
            stop_early = True
            
        return stop_early

    def load_latest_checkpoint(self, model, optimizer):
        """
        Loads the latest checkpoint from self.checkpoint_dir.
        Restores model and optimizer states, and monitor's internal state.
        Returns:
            tuple: (model, optimizer, start_epoch, best_metric_val, counter)
                   Returns (model, optimizer, 0, initial_best_metric, 0) if no checkpoint or error.
        """
        initial_best_metric = -float('inf') if self.metric_mode == 'max' else float('inf')
        
        if not os.path.isdir(self.checkpoint_dir):
            logger_monitor.info(f"Checkpoint directory {self.checkpoint_dir} does not exist. Starting from scratch.")
            return model, optimizer, 0, initial_best_metric, 0

        checkpoint_files = sorted(glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')), key=os.path.getmtime, reverse=True)
        
        if not checkpoint_files:
            logger_monitor.info(f"No checkpoints found in {self.checkpoint_dir}. Starting from scratch.")
            return model, optimizer, 0, initial_best_metric, 0
        
        latest_checkpoint_path = checkpoint_files[0]
        logger_monitor.info(f"Attempting to load latest checkpoint: {latest_checkpoint_path}")
        
        try:
            # Load to CPU first to avoid GPU mismatches if resuming on different device setup
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu') 
            
            model.load_state_dict(checkpoint['model_state_dict'])
            # Move model to device *after* loading state_dict, if device is known by calling script
            # device = next(model.parameters()).device # Get device if model was already moved
            # model.to(device) # Ensure model is on correct device

            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # If optimizer was on GPU, its state needs to be moved too if model is moved
                # for state in optimizer.state.values():
                #     for k, v in state.items():
                #         if isinstance(v, torch.Tensor):
                #             state[k] = v.to(device)
                logger_monitor.info(f"Loaded optimizer state from checkpoint.")
            elif optimizer is None:
                 logger_monitor.warning("No optimizer provided to load_latest_checkpoint, optimizer state not loaded.")
            else:
                logger_monitor.warning("Optimizer state not found in checkpoint.")

            start_epoch = checkpoint.get('epoch', 0) # Epoch saved is the next one to start
            
            # Restore monitor's state
            # Check if the saved monitor metric matches the current one
            if checkpoint.get('monitor_metric_name') == self.monitor_metric_name and \
               checkpoint.get('monitor_metric_mode') == self.metric_mode:
                self.best_metric_val = checkpoint.get('monitor_best_metric_val', initial_best_metric)
                self.best_epoch = checkpoint.get('monitor_best_epoch', start_epoch -1 if start_epoch > 0 else 0)
                self.counter = checkpoint.get('monitor_counter', 0)
                logger_monitor.info(f"Restored monitor state: Best Metric ({self.monitor_metric_name}) = {self.best_metric_val:.4f} at epoch {self.best_epoch+1}, Counter = {self.counter}")
            else:
                logger_monitor.warning(f"Monitor metric in checkpoint ({checkpoint.get('monitor_metric_name')}) "
                               f"differs from current ({self.monitor_metric_name}). Monitor state not fully restored.")
                # Keep self.best_metric_val and self.counter as initialized if metric differs
                # Or decide on a strategy, e.g., reset if metric name changed.

            logger_monitor.info(f"Successfully resumed model and optimizer from checkpoint. Next epoch to run: {start_epoch}.")
            return model, optimizer, start_epoch, self.best_metric_val, self.counter

        except Exception as e:
            logger_monitor.error(f"Error loading checkpoint {latest_checkpoint_path}: {e}", exc_info=True)
            return model, optimizer, 0, initial_best_metric, 0


class PerformanceMetricsMonitor:
    def __init__(self, model, data_loader, device, criterion=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion 

    @torch.no_grad() 
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs_class1 = [] 
        running_loss = 0.0

        for batch_x, batch_y in self.data_loader: 
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            outputs = self.model(batch_x) 
            
            if self.criterion:
                loss = self.criterion(outputs, batch_y)
                running_loss += loss.item() * batch_x.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            if probs.shape[1] > 1: 
                all_probs_class1.extend(probs[:, 1].cpu().numpy()) 
            else: 
                all_probs_class1.extend(probs[:, 0].cpu().numpy())


        metrics = {}
        if self.criterion and len(self.data_loader.dataset) > 0:
            metrics['loss'] = running_loss / len(self.data_loader.dataset)
        elif self.criterion:
             metrics['loss'] = 0

        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        all_probs_class1_np = np.array(all_probs_class1) if all_probs_class1 else np.array([0.0])

        metrics['accuracy'] = accuracy_score(all_labels_np, all_preds_np)
        metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels_np, all_preds_np)
        
        if len(np.unique(all_labels_np)) > 1 and len(all_probs_class1_np) == len(all_labels_np): 
            metrics['auc'] = roc_auc_score(all_labels_np, all_probs_class1_np)
            metrics['f1_score'] = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
            metrics['precision'] = precision_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
            metrics['recall'] = recall_score(all_labels_np, all_preds_np, average='binary', zero_division=0)
            cm = confusion_matrix(all_labels_np, all_preds_np)
            metrics['confusion_matrix'] = cm
            if cm.shape == (2,2) : 
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                metrics['specificity'] = "N/A (CM not 2x2)"
        else: 
            metrics['auc'] = 0.0
            metrics['f1_score'] = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0) if len(all_labels_np)>0 else 0.0
            metrics['precision'] = precision_score(all_labels_np, all_preds_np, average='binary', zero_division=0) if len(all_labels_np)>0 else 0.0
            metrics['recall'] = recall_score(all_labels_np, all_preds_np, average='binary', zero_division=0) if len(all_labels_np)>0 else 0.0
            metrics['specificity'] = 0.0
            metrics['confusion_matrix'] = confusion_matrix(all_labels_np, all_preds_np) 

        metrics['predictions_list'] = all_preds_np.tolist()
        metrics['labels_list'] = all_labels_np.tolist()
        metrics['probabilities_class1_list'] = all_probs_class1_np.tolist()
        
        log_str_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list) and not isinstance(v, np.ndarray)}
        logger_monitor.info(f"Evaluation Metrics: {log_str_metrics}")
        return metrics