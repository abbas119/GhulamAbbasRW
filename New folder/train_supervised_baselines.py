# NMT_EEGPT_Project/train_supervised_baselines.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

# Import configurations, datasets, models, and monitors
import config_supervised as cfg
from dataset_supervised import SupervisedNMTDataset
from models.hybrid_cnn_transformer import HybridCNNTransformer
from models.ctnet_model import CTNet
from models.eeg_conformer_model import EEGConformer
from monitors import TrainingMonitor # Uses the enhanced monitor
import torch.nn.functional as F

# Setup logger uniquely for this module
logger_train_sup = logging.getLogger('train_supervised_logger')
if not logger_train_sup.handlers:
    logger_train_sup.setLevel(logging.INFO)
    ch_train_sup = logging.StreamHandler()
    ch_train_sup.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s'))
    logger_train_sup.addHandler(ch_train_sup)

def setup_logging(log_dir, model_name_str): # File logger for each specific run
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{model_name_str}.txt')
    
    file_logger = logging.getLogger(f'{model_name_str}_file_logger')
    if file_logger.hasHandlers(): file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a') # Append to log if resuming
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    file_logger.addHandler(fh)
    return file_logger

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', n_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction
        self.n_classes = n_classes
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        alpha_factor = torch.ones_like(targets, dtype=torch.float).to(inputs.device)
        if isinstance(self.alpha, (float, int)): # Binary case, alpha for positive class
            alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        elif isinstance(self.alpha, (list, tuple, torch.Tensor)): # Multi-class alpha
            if not isinstance(self.alpha, torch.Tensor): 
                self.alpha = torch.tensor(self.alpha, device=inputs.device, dtype=torch.float)
            alpha_factor = self.alpha[targets.data.view(-1)]
        
        F_loss = alpha_factor * (1-pt)**self.gamma * CE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        return torch.sum(F_loss) if self.reduction == 'sum' else F_loss

def train_model(model_name_str):
    current_run_logger = setup_logging(cfg.LOG_DIR_SUPERVISED, model_name_str)
    current_run_logger.info(f"--- Initializing Supervised Training for: {model_name_str} ---")
    current_run_logger.info(f"Using device: {'cuda' if cfg.CUDA else 'cpu'}")

    # --- Data Loading ---
    current_run_logger.info("Loading training dataset...")
    train_dataset = SupervisedNMTDataset(
        data_dir=cfg.PROCESSED_DATA_SUPERVISED_DIR, split_type='train', augment=True,
        segment_duration_sec=cfg.SEGMENT_DURATION_SEC, target_sfreq=cfg.TARGET_SFREQ,
        test_mode_reduce_data=cfg.TEST_MODE_REDUCE_DATA,
        n_recordings_test_mode=10 if cfg.TEST_MODE_REDUCE_DATA else None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE_TRAIN_LOADER, num_workers=4, pin_memory=True, drop_last=True)
    
    current_run_logger.info("Loading validation dataset...")
    val_dataset = SupervisedNMTDataset(
        data_dir=cfg.PROCESSED_DATA_SUPERVISED_DIR, split_type='eval', augment=False,
        segment_duration_sec=cfg.SEGMENT_DURATION_SEC, target_sfreq=cfg.TARGET_SFREQ,
        test_mode_reduce_data=cfg.TEST_MODE_REDUCE_DATA,
        n_recordings_test_mode=10 if cfg.TEST_MODE_REDUCE_DATA else None)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    current_run_logger.info(f"Train segments: {len(train_dataset)}, Validation segments: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        current_run_logger.error("One of the datasets is empty. Exiting.")
        return

    # --- Model Initialization ---
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    model = None
    if model_name_str == 'HybridCNNTransformer':
        model = HybridCNNTransformer(
            n_channels=cfg.N_CHANNELS_SELECTED, n_start_chans=cfg.HYBRID_N_START_CHANS,
            n_layers_transformer=cfg.HYBRID_N_LAYERS_TRANSFORMER, n_heads=cfg.HYBRID_N_HEADS,
            hidden_dim=cfg.HYBRID_HIDDEN_DIM, ff_dim=cfg.HYBRID_FF_DIM,
            dropout=cfg.HYBRID_DROPOUT, input_time_length=cfg.INPUT_TIME_LENGTH,
            n_classes=cfg.N_CLASSES)
    elif model_name_str == 'CTNet':
        model = CTNet(
            n_channels=cfg.N_CHANNELS_SELECTED, n_classes=cfg.N_CLASSES,
            input_time_length=cfg.INPUT_TIME_LENGTH, target_sfreq=cfg.TARGET_SFREQ,
            f1=cfg.CTNET_F1, d_multiplier=cfg.CTNET_D, f2=cfg.CTNET_F2,
            kc1_divisor=4, pool1_size=cfg.CTNET_P1, k2_kernel_length=cfg.CTNET_K2, pool2_size=cfg.CTNET_P2,
            transformer_depth=cfg.CTNET_TRANSFORMER_DEPTH, transformer_heads=cfg.CTNET_TRANSFORMER_HEADS,
            dropout_cnn=cfg.CTNET_DROPOUT_CNN, 
            dropout_transformer_p1=cfg.CTNET_DROPOUT_TRANSFORMER,
            dropout_classifier_p2=cfg.CTNET_DROPOUT_CLASSIFIER)
    elif model_name_str == 'EEGConformer':
        model = EEGConformer(
            n_channels=cfg.N_CHANNELS_SELECTED, 
            n_classes=cfg.N_CLASSES,
            input_time_length=cfg.INPUT_TIME_LENGTH, 
            target_sfreq=cfg.TARGET_SFREQ,
            n_filters_time=cfg.CONFORMER_N_FILTERS_TIME, 
            filter_time_length_ms=getattr(cfg, 'CONFORMER_FILTER_TIME_LENGTH_MS', 100), 
            n_filters_spat=cfg.CONFORMER_N_FILTERS_SPAT, 
            pool_time_length_ms=getattr(cfg, 'CONFORMER_POOL_TIME_LENGTH_MS', 300), 
            pool_time_stride_ms=getattr(cfg, 'CONFORMER_POOL_TIME_STRIDE_MS', 60),  
            cnn_drop_prob=cfg.CONFORMER_DROPOUT, 
            transformer_d_model_explicit=getattr(cfg, 'CONFORMER_TRANSFORMER_D_MODEL_EXPLICIT', None), # Corrected argument
            transformer_depth=cfg.CONFORMER_TRANSFORMER_DEPTH, 
            transformer_n_heads=cfg.CONFORMER_TRANSFORMER_HEADS,
            transformer_ff_dim_factor=getattr(cfg, 'CONFORMER_TRANSFORMER_FF_DIM_FACTOR', 2),
            transformer_drop_prob=getattr(cfg, 'CONFORMER_TRANSFORMER_DROP_PROB', cfg.CONFORMER_DROPOUT), # Use specific or general dropout
            classifier_hidden_dim=getattr(cfg, 'CONFORMER_CLASSIFIER_HIDDEN_DIM', 128),
            classifier_drop_prob=getattr(cfg, 'CONFORMER_CLASSIFIER_DROP_PROB', cfg.CONFORMER_DROPOUT) # Use specific or general dropout
        )
    else:
        current_run_logger.error(f"Unknown model_name_str: {model_name_str}")
        raise ValueError(f"Unknown model_name_str: {model_name_str}")

    model.to(device) 
    current_run_logger.info(f"Model: {model_name_str} initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Loss Function & Optimizer ---
    if hasattr(cfg, 'CLASS_WEIGHTS') and cfg.CLASS_WEIGHTS:
        class_weights_tensor = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        current_run_logger.info(f"Using weighted CrossEntropyLoss with weights: {cfg.CLASS_WEIGHTS}")
    else: 
        try: 
            criterion = FocalLoss(alpha=getattr(cfg, 'FOCAL_LOSS_ALPHA', 0.25), 
                                  gamma=getattr(cfg, 'FOCAL_LOSS_GAMMA', 2.0), 
                                  n_classes=cfg.N_CLASSES)
            current_run_logger.info(f"Using FocalLoss (alpha={criterion.alpha}, gamma={criterion.gamma}).")
        except AttributeError: 
            criterion = nn.CrossEntropyLoss()
            current_run_logger.info("Using standard CrossEntropyLoss (no class weights, FocalLoss params not found in config).")
            
    optimizer = optim.AdamW(model.parameters(), lr=cfg.INIT_LR, weight_decay=getattr(cfg, 'WEIGHT_DECAY', 1e-4))
    scheduler = None
    if cfg.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=cfg.SCHEDULER_FACTOR, 
            patience=cfg.SCHEDULER_PATIENCE, min_lr=cfg.MIN_LR)

    # --- Training Monitor & Checkpointing Setup ---
    monitor_metric_to_use = cfg.MONITOR_METRIC if hasattr(cfg, 'MONITOR_METRIC') else 'val_f1_score'
    metric_mode_to_use = 'max' if 'loss' not in monitor_metric_to_use.lower() else 'min'
    model_save_path = os.path.join(cfg.MODEL_SAVE_DIR_SUPERVISED, f'{model_name_str}_best.pt')
    checkpoint_dir_path = os.path.join(cfg.MODEL_SAVE_DIR_SUPERVISED, f'{model_name_str}_checkpoints/')
    
    monitor = TrainingMonitor(
        model_path=model_save_path, 
        checkpoint_dir=checkpoint_dir_path, 
        patience=cfg.PATIENCE_EARLY_STOPPING,
        monitor_metric_name=monitor_metric_to_use,
        metric_mode=metric_mode_to_use
    )
    
    start_epoch = 0 

    # <<< --- ATTEMPT TO LOAD LATEST CHECKPOINT --- >>>
    if os.path.isdir(checkpoint_dir_path) and any(f.startswith('checkpoint_epoch_') for f in os.listdir(checkpoint_dir_path)):
        current_run_logger.info(f"Attempting to load latest checkpoint from {checkpoint_dir_path}...")
        resume_data = monitor.load_latest_checkpoint(model, optimizer) 
        if resume_data:
            _model_loaded, _optimizer_loaded, loaded_epoch_num, best_metric_val_resumed, counter_resumed = resume_data
            
            if loaded_epoch_num > 0: 
                model = _model_loaded.to(device) 
                optimizer = _optimizer_loaded 
                start_epoch = loaded_epoch_num 
                
                monitor.best_metric_val = best_metric_val_resumed
                monitor.counter = counter_resumed
                monitor.best_epoch = start_epoch -1 
                
                current_run_logger.info(f"Resumed training from epoch {start_epoch}. Monitor state restored: Best {monitor.monitor_metric_name}={monitor.best_metric_val:.4f}, Counter={monitor.counter}.")
            else: 
                 current_run_logger.info("No suitable checkpoint loaded. Starting from scratch.")
                 start_epoch = 0 
        else: 
            current_run_logger.info("No checkpoint found by monitor or error in loading. Starting from scratch.")
            start_epoch = 0
    else:
        current_run_logger.info("No checkpoint directory or checkpoints. Starting from scratch.")
        start_epoch = 0
    # <<< --- END OF CHECKPOINT LOADING --- >>>

    current_run_logger.info(f"Starting training loop from epoch {start_epoch}...")
    training_start_time = time.time()

    for epoch in range(start_epoch, cfg.MAX_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.MAX_EPOCHS} [T]", leave=False, disable=None):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch_x_val, batch_y_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.MAX_EPOCHS} [V]", leave=False, disable=None):
                batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_x_val)
                loss_val = criterion(outputs_val, batch_y_val)
                running_val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                all_val_preds.extend(predicted_val.cpu().numpy())
                all_val_labels.extend(batch_y_val.cpu().numpy())
        
        epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) if len(all_val_labels) > 0 else 0
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0) if len(all_val_labels) > 0 else 0
        val_recall = recall_score(all_val_labels, all_val_preds, average='binary', zero_division=0) if len(all_val_labels) > 0 else 0
        
        epoch_duration = time.time() - epoch_start_time
        
        metrics_for_monitor = {
            'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss,
            'val_accuracy': val_accuracy, 'val_f1_score': val_f1, 'val_recall': val_recall
        }
        
        log_msg_epoch = (f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                         f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, "
                         f"LR: {optimizer.param_groups[0]['lr']:.2e}, Time: {epoch_duration:.2f}s")
        current_run_logger.info(log_msg_epoch)
        # print(log_msg_epoch) # tqdm provides progress, this might be redundant unless tqdm is disabled

        if cfg.USE_SCHEDULER and scheduler is not None:
            metric_for_scheduler = metrics_for_monitor.get(monitor_metric_to_use, epoch_val_loss)
            scheduler.step(metric_for_scheduler)

        if monitor.step(epoch, model, optimizer, metrics_for_monitor, checkpoint_interval=getattr(cfg, 'CHECKPOINT_INTERVAL', 10)): # Use getattr for CHECKPOINT_INTERVAL
            current_run_logger.info(f"Early stopping decision from monitor for epoch {epoch+1}.")
            break
            
    total_training_time = time.time() - training_start_time
    current_run_logger.info(f"--- Training finished for {model_name_str}. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s ---")
    if monitor.best_epoch is not None and monitor.best_metric_val != (-float('inf') if metric_mode_to_use == 'max' else float('inf')):
         current_run_logger.info(f"Best model for {model_name_str} saved at: {model_save_path} (Achieved at Epoch: {monitor.best_epoch+1}, {monitor.monitor_metric_name}: {monitor.best_metric_val:.4f})")
    else:
         current_run_logger.info(f"No best model saved for {model_name_str}. Check logs and patience settings.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train supervised EEG classification models.")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['HybridCNNTransformer', 'CTNet', 'EEGConformer'],
                        help="Name of the model to train.")
    args = parser.parse_args()
    
    if hasattr(cfg, 'LOG_DIR_SUPERVISED'): os.makedirs(cfg.LOG_DIR_SUPERVISED, exist_ok=True)
    if hasattr(cfg, 'MODEL_SAVE_DIR_SUPERVISED'): os.makedirs(cfg.MODEL_SAVE_DIR_SUPERVISED, exist_ok=True)
    
    train_model(args.model)