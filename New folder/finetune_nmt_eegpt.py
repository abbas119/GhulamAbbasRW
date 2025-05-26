# NMT_EEGPT_Project/finetune_nmt_eegpt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm

import config_ssl_finetune as cfg_ft
import config_ssl_pretrain as cfg_ssl_model_structure # For NMT-EEGPT structure
from dataset_supervised import SupervisedNMTDataset
from models.nmt_eegpt_downstream_model import NMT_EEGPT_Classifier
from monitors import TrainingMonitor
import torch.nn.functional as F

# Setup logger uniquely for this module
logger_ssl_finetune = logging.getLogger('ssl_finetune_logger')
# ... (similar logger setup as in train_supervised_baselines.py) ...
if not logger_ssl_finetune.handlers:
    logger_ssl_finetune.setLevel(logging.INFO)
    ch_ssl_finetune = logging.StreamHandler()
    ch_ssl_finetune.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s'))
    logger_ssl_finetune.addHandler(ch_ssl_finetune)

def setup_logging_finetune(log_dir, finetune_mode_str): # File logger
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'finetuning_log_{finetune_mode_str}.txt')
    
    file_logger = logging.getLogger(f'{finetune_mode_str}_file_logger')
    if file_logger.hasHandlers(): file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a') # Append if resuming
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    file_logger.addHandler(fh)
    return file_logger

# ... (FocalLoss class definition remains the same) ...
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
        if isinstance(self.alpha, (float, int)):
            alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        elif isinstance(self.alpha, (list, tuple, torch.Tensor)):
            if not isinstance(self.alpha, torch.Tensor): self.alpha = torch.tensor(self.alpha, device=inputs.device)
            alpha_factor = self.alpha[targets.data.view(-1)]
        F_loss = alpha_factor * (1-pt)**self.gamma * CE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        return torch.sum(F_loss) if self.reduction == 'sum' else F_loss

def finetune_or_probe(mode):
    current_run_logger = setup_logging_finetune(cfg_ft.LOG_DIR_SSL_FINETUNE, mode)
    current_run_logger.info(f"--- Initializing NMT-EEGPT Downstream Task: {mode} ---")
    current_run_logger.info(f"Using device: {'cuda' if cfg_ft.CUDA else 'cpu'}")

    # --- Data Loading --- (Same as supervised)
    current_run_logger.info("Loading training dataset for downstream task...")
    train_dataset = SupervisedNMTDataset(
        data_dir=cfg_ft.PROCESSED_DATA_FINETUNE_DIR, split_type='train', augment=True,
        segment_duration_sec=cfg_ft.SEGMENT_DURATION_SEC, target_sfreq=cfg_ft.TARGET_SFREQ,
        test_mode_reduce_data=cfg_ft.TEST_MODE_REDUCE_DATA if hasattr(cfg_ft, 'TEST_MODE_REDUCE_DATA') else False,
        n_recordings_test_mode=10 if hasattr(cfg_ft, 'TEST_MODE_REDUCE_DATA') else None)
    train_loader = DataLoader(train_dataset, batch_size=cfg_ft.BATCH_SIZE_FINETUNE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    current_run_logger.info("Loading validation dataset for downstream task...")
    val_dataset = SupervisedNMTDataset(
        data_dir=cfg_ft.PROCESSED_DATA_FINETUNE_DIR, split_type='eval', augment=False,
        segment_duration_sec=cfg_ft.SEGMENT_DURATION_SEC, target_sfreq=cfg_ft.TARGET_SFREQ,
        test_mode_reduce_data=cfg_ft.TEST_MODE_REDUCE_DATA if hasattr(cfg_ft, 'TEST_MODE_REDUCE_DATA') else False,
        n_recordings_test_mode=10 if hasattr(cfg_ft, 'TEST_MODE_REDUCE_DATA') else None)
    val_loader = DataLoader(val_dataset, batch_size=cfg_ft.BATCH_SIZE_FINETUNE, shuffle=False, num_workers=4, pin_memory=True)
    current_run_logger.info(f"Finetune Train segs: {len(train_dataset)}, Val segs: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0: current_run_logger.error("Dataset empty. Exiting."); return

    # --- Model Initialization & Loading Pretrained Weights ---
    device = torch.device('cuda' if cfg_ft.CUDA else 'cpu')
    freeze_encoder_flag = True if mode == 'linear_probe' else False
    
    model = NMT_EEGPT_Classifier(
        n_channels_model=cfg_ssl_model_structure.N_CHANNELS_MODEL,
        segment_time_len_samples=cfg_ssl_model_structure.INPUT_TIME_LENGTH_MODEL,
        patch_time_len_samples=cfg_ssl_model_structure.PATCH_TIME_LENGTH_SAMPLES,
        embed_dim=cfg_ssl_model_structure.EMBED_DIM,
        encoder_layers=cfg_ssl_model_structure.ENCODER_LAYERS,
        num_heads=cfg_ssl_model_structure.NUM_HEADS,
        ff_dim=cfg_ssl_model_structure.FEEDFORWARD_DIM,
        dropout_transformer=cfg_ssl_model_structure.DROPOUT_PRETRAIN,
        num_summary_tokens=cfg_ssl_model_structure.NUM_SUMMARY_TOKENS,
        n_classes=cfg_ft.N_CLASSES,
        use_adaptive_spatial_filter=cfg_ft.USE_ADAPTIVE_SPATIAL_FILTER,
        n_input_channels_to_asf=cfg_ft.N_CHANNELS_INPUT_TO_MODEL,
        pretrained_encoder_path=cfg_ft.PRETRAINED_NMT_EEGPT_ENCODER_PATH, # Path to FULL Pretrain_NMT_EEGPT model
        freeze_encoder=freeze_encoder_flag
    ).to(device) # Move model to device before optimizer
    current_run_logger.info(f"NMT-EEGPT Classifier for {mode}. Total Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e3:.2f}K")

    # --- Loss & Optimizer ---
    current_lr = cfg_ft.INIT_LR_FINETUNE
    if mode == 'full_finetune': current_lr *= 0.1 # Smaller LR for full fine-tune
    
    if cfg_ft.CLASS_WEIGHTS_FINETUNE: # Use Class Weights
        class_weights_tensor_ft = torch.tensor(cfg_ft.CLASS_WEIGHTS_FINETUNE, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor_ft)
        current_run_logger.info(f"Using weighted CrossEntropyLoss: {cfg_ft.CLASS_WEIGHTS_FINETUNE}")
    else: # Default to Focal Loss
        criterion = FocalLoss(alpha=cfg_ft.FOCAL_LOSS_ALPHA_FINETUNE, gamma=cfg_ft.FOCAL_LOSS_GAMMA_FINETUNE, n_classes=cfg_ft.N_CLASSES)
        current_run_logger.info(f"Using FocalLoss (alpha={cfg_ft.FOCAL_LOSS_ALPHA_FINETUNE}, gamma={cfg_ft.FOCAL_LOSS_GAMMA_FINETUNE}).")
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr, weight_decay=cfg_ft.WEIGHT_DECAY_FINETUNE)
    scheduler = None
    if cfg_ft.USE_SCHEDULER_FINETUNE:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=cfg_ft.SCHEDULER_FACTOR_FINETUNE, 
            patience=cfg_ft.SCHEDULER_PATIENCE_FINETUNE, min_lr=cfg_ft.MIN_LR_FINETUNE)
            
    # --- Monitor & Checkpointing ---
    model_name_finetune = f"NMT_EEGPT_Classifier_{mode}"
    monitor_metric_ft = cfg_ft.MONITOR_METRIC_FINETUNE if hasattr(cfg_ft, 'MONITOR_METRIC_FINETUNE') else 'val_f1_score'
    metric_mode_ft = 'max' if 'loss' not in monitor_metric_ft.lower() else 'min'
    model_save_path_ft = os.path.join(cfg_ft.MODEL_SAVE_DIR_SSL_FINETUNE, f'{model_name_finetune}_best.pt')
    checkpoint_dir_path_ft = os.path.join(cfg_ft.MODEL_SAVE_DIR_SSL_FINETUNE, f'{model_name_finetune}_checkpoints/')
    
    monitor_ft = TrainingMonitor(
        model_path=model_save_path_ft, checkpoint_dir=checkpoint_dir_path_ft, 
        patience=cfg_ft.PATIENCE_EARLY_STOPPING_FINETUNE,
        monitor_metric_name=monitor_metric_ft, metric_mode=metric_mode_ft)
    
    start_epoch_ft = 0
    # <<< --- ATTEMPT TO LOAD LATEST FINETUNE CHECKPOINT --- >>>
    if os.path.isdir(checkpoint_dir_path_ft) and any(f.startswith('checkpoint_epoch_') for f in os.listdir(checkpoint_dir_path_ft)):
        current_run_logger.info(f"Attempting to load latest finetune checkpoint from {checkpoint_dir_path_ft}...")
        resume_data_ft = monitor_ft.load_latest_checkpoint(model, optimizer)
        if resume_data_ft:
            _model_loaded, _optimizer_loaded, loaded_epoch_num, best_metric_val_resumed, counter_resumed = resume_data_ft
            if loaded_epoch_num > 0:
                model = _model_loaded.to(device)
                optimizer = _optimizer_loaded
                start_epoch_ft = loaded_epoch_num
                monitor_ft.best_metric_val = best_metric_val_resumed
                monitor_ft.counter = counter_resumed
                monitor_ft.best_epoch = start_epoch_ft -1
                current_run_logger.info(f"Resumed finetuning ({mode}) from epoch {start_epoch_ft}. Monitor state restored.")
            else: current_run_logger.info(f"No suitable finetune checkpoint. Starting {mode} from scratch."); start_epoch_ft = 0
        else: current_run_logger.info(f"No finetune checkpoint by monitor. Starting {mode} from scratch."); start_epoch_ft = 0
    else: current_run_logger.info(f"No finetune checkpoint dir/files. Starting {mode} from scratch."); start_epoch_ft = 0
    # <<< --- END OF FINETUNE CHECKPOINT LOADING --- >>>

    current_run_logger.info(f"Starting {mode} loop from epoch {start_epoch_ft}...")
    # --- Training Loop (similar to supervised training) ---
    # ... (The loop is identical to the one in train_supervised_baselines.py, just using _ft variables) ...
    # Replace current_run_logger with the one defined in this script.
    # Use tqdm desc like f"Epoch {epoch+1}/{cfg_ft.MAX_EPOCHS_FINETUNE} [{mode} T]"
    # Use monitor_ft.step(...)
    # Log with current_run_logger
    training_start_time = time.time()
    for epoch in range(start_epoch_ft, cfg_ft.MAX_EPOCHS_FINETUNE):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_ft.MAX_EPOCHS_FINETUNE} [{mode} T]", leave=False, disable=None):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        running_val_loss = 0.0; all_val_preds = []; all_val_labels = []
        with torch.no_grad():
            for batch_x_val, batch_y_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg_ft.MAX_EPOCHS_FINETUNE} [{mode} V]", leave=False, disable=None):
                batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_x_val)
                loss_val = criterion(outputs_val, batch_y_val)
                running_val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                all_val_preds.extend(predicted_val.cpu().numpy()); all_val_labels.extend(batch_y_val.cpu().numpy())
        epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) if len(all_val_labels) > 0 else 0
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0) if len(all_val_labels) > 0 else 0
        val_recall = recall_score(all_val_labels, all_val_preds, average='binary', zero_division=0) if len(all_val_labels) > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        metrics_for_monitor = {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'val_accuracy': val_accuracy, 'val_f1_score': val_f1, 'val_recall': val_recall}
        log_msg_epoch = (f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}, Time: {epoch_duration:.2f}s")
        current_run_logger.info(log_msg_epoch)
        print(log_msg_epoch)
        if cfg_ft.USE_SCHEDULER_FINETUNE and scheduler is not None: scheduler.step(metrics_for_monitor.get(monitor_metric_ft, epoch_val_loss))
        if monitor_ft.step(epoch, model, optimizer, metrics_for_monitor, checkpoint_interval=10): # Use a finetune checkpoint interval
            current_run_logger.info(f"Early stopping for {mode} at epoch {epoch+1}."); break
    total_training_time = time.time() - training_start_time
    current_run_logger.info(f"--- {mode} finished. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s ---")
    if monitor_ft.best_epoch is not None and monitor_ft.best_metric_val != (-float('inf') if metric_mode_ft == 'max' else float('inf')):
         current_run_logger.info(f"Best {mode} model saved: {model_save_path_ft} (Epoch: {monitor_ft.best_epoch+1}, {monitor_ft.monitor_metric_name}: {monitor_ft.best_metric_val:.4f})")
    else: current_run_logger.info(f"No best {mode} model saved. Check logs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune or Linear Probe NMT-EEGPT.")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['linear_probe', 'full_finetune'],
                        help="Mode for using the pretrained NMT-EEGPT.")
    args = parser.parse_args()
    if hasattr(cfg_ft, 'LOG_DIR_SSL_FINETUNE'): os.makedirs(cfg_ft.LOG_DIR_SSL_FINETUNE, exist_ok=True)
    if hasattr(cfg_ft, 'MODEL_SAVE_DIR_SSL_FINETUNE'): os.makedirs(cfg_ft.MODEL_SAVE_DIR_SSL_FINETUNE, exist_ok=True)
    if not os.path.exists(cfg_ft.PRETRAINED_NMT_EEGPT_ENCODER_PATH):
        print(f"ERROR: Pretrained NMT-EEGPT model checkpoint not found at {cfg_ft.PRETRAINED_NMT_EEGPT_ENCODER_PATH}")
    elif not os.path.exists(cfg_ft.PROCESSED_DATA_FINETUNE_DIR) or len(os.listdir(os.path.join(cfg_ft.PROCESSED_DATA_FINETUNE_DIR, 'train', 'normal'))) == 0 :
         print(f"ERROR: Processed supervised data for finetuning not found at {cfg_ft.PROCESSED_DATA_FINETUNE_DIR}")
    else: finetune_or_probe(args.mode)