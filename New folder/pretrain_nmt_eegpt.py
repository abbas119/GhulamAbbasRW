# NMT_EEGPT_Project/pretrain_nmt_eegpt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import time
import random
from einops import rearrange
from tqdm import tqdm

import config_ssl_pretrain as cfg_ssl
from dataset_ssl import NMT_SSL_Patched_Dataset
from models.nmt_eegpt_pretrain_model import NMT_EEGPT_Pretrain
from monitors import TrainingMonitor
import torch.nn.functional as F

# Use torch.cuda.amp for PyTorch versions like yours (1.6+ including 2.x) for CUDA-specific AMP
from torch.cuda.amp import GradScaler, autocast as autocast_cuda # Explicitly import for CUDA

# Setup logger uniquely for this module
logger_ssl_pretrain = logging.getLogger('ssl_pretrain_logger')
if not logger_ssl_pretrain.handlers:
    logger_ssl_pretrain.setLevel(logging.INFO)
    ch_ssl_pretrain = logging.StreamHandler()
    ch_ssl_pretrain.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s'))
    logger_ssl_pretrain.addHandler(ch_ssl_pretrain)

def setup_logging_ssl(log_dir, model_name_str="NMT_EEGPT_Pretrain"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'pretraining_log_{model_name_str}.txt')
    
    file_logger_instance_name = f'{model_name_str}_file_logger_instance'
    file_logger = logging.getLogger(file_logger_instance_name)
    
    if file_logger.hasHandlers():
        for handler in list(file_logger.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
                handler.close(); file_logger.removeHandler(handler)
    file_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in file_logger.handlers):
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        file_logger.addHandler(fh)
    
    root_logger = logging.getLogger()
    if not root_logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        if not root_logger.handlers:
             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler()])
        elif not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console_handler_root = logging.StreamHandler(); console_handler_root.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); root_logger.addHandler(console_handler_root)
            root_logger.setLevel(logging.INFO)
    return file_logger

def generate_random_masks_for_batch(batch_size, n_channels, n_time_patches_per_channel, 
                                   time_mask_percentage_from_cfg, 
                                   channel_mask_percentage_from_cfg, 
                                   device):
    num_time_patches_to_mask = int(n_time_patches_per_channel * time_mask_percentage_from_cfg)
    masked_time_cols = torch.zeros(batch_size, n_time_patches_per_channel, dtype=torch.bool, device=device)
    for i in range(batch_size):
        indices = torch.randperm(n_time_patches_per_channel, device=device)[:num_time_patches_to_mask]
        masked_time_cols[i, indices] = True 
    
    num_channels_to_mask = int(n_channels * channel_mask_percentage_from_cfg)
    masked_channel_rows = torch.zeros(batch_size, n_channels, dtype=torch.bool, device=device)
    for i in range(batch_size):
        indices = torch.randperm(n_channels, device=device)[:num_channels_to_mask]
        masked_channel_rows[i, indices] = True 
    
    m_bar_mask_2d = masked_channel_rows.unsqueeze(2) | masked_time_cols.unsqueeze(1) 
    m_bar_mask_flat = rearrange(m_bar_mask_2d, 'b c nt -> b (c nt)')
    return m_bar_mask_flat

def pretrain_nmt_eegpt():
    current_run_logger = setup_logging_ssl(cfg_ssl.LOG_DIR_SSL_PRETRAIN, "NMT_EEGPT_Pretrain")
    current_run_logger.info(f"--- Initializing NMT-EEGPT Self-Supervised Pretraining ---")
    device = torch.device('cuda' if cfg_ssl.CUDA else 'cpu')
    current_run_logger.info(f"Using device: {device.type}")
    
    use_amp_effective = cfg_ssl.USE_AMP and (device.type == 'cuda')
    if cfg_ssl.USE_AMP and not (device.type == 'cuda'):
        current_run_logger.warning("AMP requested but CUDA not available. Disabling AMP.")
        use_amp_effective = False

    current_run_logger.info("Loading SSL dataset (all NMT segments)...")
    ssl_dataset = NMT_SSL_Patched_Dataset(
        data_dir_ssl=cfg_ssl.PREPROCESSED_SSL_DATA_DIR,
        segment_duration_sec=cfg_ssl.SEGMENT_DURATION_SEC, target_sfreq=cfg_ssl.TARGET_SFREQ,
        patch_duration_ms=cfg_ssl.PATCH_DURATION_MS, n_channels=cfg_ssl.N_CHANNELS_MODEL, 
        test_mode_reduce_data=getattr(cfg_ssl, 'TEST_MODE_REDUCE_DATA', False),
        n_segments_test_mode=getattr(cfg_ssl, 'N_SEGMENTS_TEST_MODE_SSL', 100))
    ssl_loader = DataLoader(ssl_dataset, batch_size=cfg_ssl.BATCH_SIZE_PRETRAIN, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    current_run_logger.info(f"SSL Dataset loaded. Total 4s segments for pretraining: {len(ssl_dataset)}")
    if len(ssl_dataset) == 0: current_run_logger.error("SSL dataset is empty. Exiting."); return

    model = NMT_EEGPT_Pretrain(
        n_channels_model=cfg_ssl.N_CHANNELS_MODEL,
        segment_time_len_samples=cfg_ssl.INPUT_TIME_LENGTH_MODEL,
        patch_time_len_samples=cfg_ssl.PATCH_TIME_LENGTH_SAMPLES,
        embed_dim=cfg_ssl.EMBED_DIM, encoder_layers=cfg_ssl.ENCODER_LAYERS,
        predictor_layers=cfg_ssl.PREDICTOR_LAYERS, reconstructor_layers=cfg_ssl.RECONSTRUCTOR_LAYERS,
        num_heads=cfg_ssl.NUM_HEADS, ff_dim=cfg_ssl.FEEDFORWARD_DIM,
        dropout_transformer=cfg_ssl.DROPOUT_PRETRAIN,
        num_summary_tokens=cfg_ssl.NUM_SUMMARY_TOKENS, momentum_tau=cfg_ssl.MOMENTUM_TAU
    ).to(device)
    current_run_logger.info(f"NMT-EEGPT Pretrain Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg_ssl.INIT_LR_PRETRAIN, weight_decay=cfg_ssl.WEIGHT_DECAY_PRETRAIN)
    
    scaler = None 
    if use_amp_effective: # Line 120 in your log for the GradScaler FutureWarning
        scaler = GradScaler() # Use torch.cuda.amp.GradScaler() - this is fine for PyTorch 1.6+
        current_run_logger.info("Using torch.cuda.amp.GradScaler for AMP.")
    
    scheduler = None
    if cfg_ssl.LR_SCHEDULER_PRETRAIN == 'OneCycleLR':
        grad_accum_steps = getattr(cfg_ssl, 'GRAD_ACCUMULATION_STEPS', 1)
        if len(ssl_loader) > 0 :
            steps_per_epoch_effective = max(1, len(ssl_loader) // grad_accum_steps) 
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg_ssl.MAX_LR_ONE_CYCLE, 
                epochs=cfg_ssl.MAX_EPOCHS_PRETRAIN, steps_per_epoch=steps_per_epoch_effective,
                pct_start=cfg_ssl.PCT_START_ONE_CYCLE)
            current_run_logger.info(f"Using OneCycleLR scheduler with {steps_per_epoch_effective} effective steps per epoch.")
        else: current_run_logger.warning("SSL Loader empty, cannot init OneCycleLR.")

    model_name_str_pretrain = "NMT_EEGPT_Pretrain"
    monitor_metric_to_use_ssl = 'total_loss'; metric_mode_to_use_ssl = 'min'
    model_save_path_ssl = os.path.join(cfg_ssl.MODEL_SAVE_DIR_SSL_PRETRAIN, f'{model_name_str_pretrain}_best_ssl_loss.pt')
    checkpoint_dir_path_ssl = os.path.join(cfg_ssl.MODEL_SAVE_DIR_SSL_PRETRAIN, f'{model_name_str_pretrain}_checkpoints/')
    
    monitor_ssl = TrainingMonitor(model_path=model_save_path_ssl, checkpoint_dir=checkpoint_dir_path_ssl, 
        patience=getattr(cfg_ssl, 'PATIENCE_PRETRAIN', 50),
        monitor_metric_name=monitor_metric_to_use_ssl, metric_mode=metric_mode_to_use_ssl)
    
    start_epoch_ssl = 0
    # ... (Checkpoint loading logic remains the same) ...
    if os.path.isdir(checkpoint_dir_path_ssl) and any(f.startswith('checkpoint_epoch_') for f in os.listdir(checkpoint_dir_path_ssl)):
        current_run_logger.info(f"Attempting to load latest SSL pretrain checkpoint from {checkpoint_dir_path_ssl}...")
        resume_data_ssl = monitor_ssl.load_latest_checkpoint(model, optimizer)
        if resume_data_ssl:
            _model_loaded, _optimizer_loaded, loaded_epoch_num, best_metric_val_resumed, counter_resumed = resume_data_ssl
            if loaded_epoch_num > 0:
                model = _model_loaded.to(device); optimizer = _optimizer_loaded; start_epoch_ssl = loaded_epoch_num
                monitor_ssl.best_metric_val = best_metric_val_resumed; monitor_ssl.counter = counter_resumed
                monitor_ssl.best_epoch = start_epoch_ssl -1
                current_run_logger.info(f"Resumed SSL pretraining from epoch {start_epoch_ssl}. Monitor state restored.")
            else: current_run_logger.info("No suitable SSL checkpoint. Starting pretraining from scratch."); start_epoch_ssl = 0
        else: current_run_logger.info("No SSL checkpoint by monitor. Starting pretraining from scratch."); start_epoch_ssl = 0
    else: current_run_logger.info("No SSL checkpoint directory/files. Starting pretraining from scratch."); start_epoch_ssl = 0


    current_run_logger.info(f"Starting NMT-EEGPT pretraining loop from epoch {start_epoch_ssl}...")
    pre_training_start_time = time.time()

    for epoch in range(start_epoch_ssl, cfg_ssl.MAX_EPOCHS_PRETRAIN):
        epoch_start_time = time.time()
        model.train()
        running_loss_align = 0.0; running_loss_recon = 0.0; running_total_loss = 0.0
        
        current_grad_accum_steps = getattr(cfg_ssl, 'GRAD_ACCUMULATION_STEPS', 1)

        for batch_idx, (batch_segment_patches) in enumerate(tqdm(ssl_loader, desc=f"Epoch {epoch+1}/{cfg_ssl.MAX_EPOCHS_PRETRAIN} [SSL]", leave=False, disable=None)):
            batch_segment_patches = batch_segment_patches.to(device) 
            B, C, N_t_per_channel, P_t_dims = batch_segment_patches.shape

            patch_mask_flat_for_m_bar = generate_random_masks_for_batch(
                B, C, N_t_per_channel, 
                cfg_ssl.TIME_PATCH_MASK_PERCENTAGE, 
                cfg_ssl.CHANNEL_MASK_PERCENTAGE,    
                device
            )

            if batch_idx % current_grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # Corrected autocast usage for torch.cuda.amp.autocast
            # Line 175 in your log for autocast TypeError
            with autocast_cuda(enabled=use_amp_effective): # autocast_cuda is from torch.cuda.amp
                pred_align_feats, pred_recon_patches, target_align_feats, target_recon_patches = \
                    model.forward_pretrain(batch_segment_patches, patch_mask_flat_for_m_bar) 
                
                loss_A = mse_loss(pred_align_feats, F.layer_norm(target_align_feats.detach(), [target_align_feats.size(-1)]))
                loss_R = mse_loss(pred_recon_patches, F.layer_norm(target_recon_patches.detach(), [target_recon_patches.size(-1)]))
                total_loss = loss_A + loss_R
            
            total_loss_scaled = total_loss / current_grad_accum_steps
            if use_amp_effective and scaler is not None:
                scaler.scale(total_loss_scaled).backward()
            else: 
                total_loss_scaled.backward()

            running_loss_align += loss_A.item()
            running_loss_recon += loss_R.item()
            running_total_loss += total_loss.item()

            if (batch_idx + 1) % current_grad_accum_steps == 0:
                if use_amp_effective and scaler is not None: scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                
                if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    model.module._update_momentum_encoder()
                else: model._update_momentum_encoder()
                
                if scheduler and cfg_ssl.LR_SCHEDULER_PRETRAIN == 'OneCycleLR':
                    scheduler.step()
        
        num_optimizer_steps_this_epoch = max(1, len(ssl_loader) // current_grad_accum_steps)
        epoch_avg_loss_align = running_loss_align / num_optimizer_steps_this_epoch
        epoch_avg_loss_recon = running_loss_recon / num_optimizer_steps_this_epoch
        epoch_avg_total_loss = running_total_loss / num_optimizer_steps_this_epoch
        epoch_duration = time.time() - epoch_start_time

        metrics_for_ssl_monitor = {
            'total_loss': epoch_avg_total_loss, 
            'align_loss_la': epoch_avg_loss_align,
            'recon_loss_lr': epoch_avg_loss_recon
        }
        
        log_msg_ssl_epoch = (f"Epoch {epoch+1}: Total Loss: {epoch_avg_total_loss:.4f} | Align Loss: {epoch_avg_loss_align:.4f} | "
                             f"Recon Loss: {epoch_avg_loss_recon:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_duration:.2f}s")
        current_run_logger.info(log_msg_ssl_epoch)
        print(log_msg_ssl_epoch)

        if scheduler and cfg_ssl.LR_SCHEDULER_PRETRAIN == 'ReduceLROnPlateau':
            scheduler.step(epoch_avg_total_loss) 
        
        if monitor_ssl.step(epoch, model, optimizer, metrics_for_ssl_monitor, checkpoint_interval=cfg_ssl.PRETRAIN_SAVE_EVERY_EPOCHS):
            current_run_logger.info(f"SSL Pretrain Early stopping decision from monitor at epoch {epoch+1}.")
            break
            
    total_pretraining_time = time.time() - pre_training_start_time
    current_run_logger.info(f"--- NMT-EEGPT Pretraining finished. Total time: {total_pretraining_time // 3600:.0f}h {(total_pretraining_time % 3600) // 60:.0f}m ---")
    if monitor_ssl.best_epoch is not None and monitor_ssl.best_metric_val != float('inf'):
         current_run_logger.info(f"Best SSL model (based on {monitor_ssl.monitor_metric_name}) saved at: {model_save_path_ssl} (Achieved at Epoch: {monitor_ssl.best_epoch+1}, Value: {monitor_ssl.best_metric_val:.4f})")
    else:
         current_run_logger.info(f"No best SSL model saved. Check logs and patience. Last checkpoint is likely the one to use.")

if __name__ == '__main__': 
    os.makedirs(getattr(cfg_ssl, 'LOG_DIR_SSL_PRETRAIN', 'logs/ssl_pretrain/'), exist_ok=True)
    os.makedirs(getattr(cfg_ssl, 'MODEL_SAVE_DIR_SSL_PRETRAIN', 'models/saved_ssl_pretrain/'), exist_ok=True)
    if hasattr(cfg_ssl, 'PREPROCESSED_SSL_DATA_DIR'): 
        os.makedirs(cfg_ssl.PREPROCESSED_SSL_DATA_DIR, exist_ok=True)
    
    import mne
    print(f"MNE version being used by pretrain_nmt_eegpt.py: {mne.__version__}")

    required_attrs_check = ['PREPROCESSED_SSL_DATA_DIR', 'TIME_PATCH_MASK_PERCENTAGE', 'CHANNEL_MASK_PERCENTAGE']
    missing_attrs_from_config = [attr for attr in required_attrs_check if not hasattr(cfg_ssl, attr)]
    
    data_dir_ok = False
    if hasattr(cfg_ssl, 'PREPROCESSED_SSL_DATA_DIR'): 
        if os.path.exists(cfg_ssl.PREPROCESSED_SSL_DATA_DIR) and \
           len(os.listdir(cfg_ssl.PREPROCESSED_SSL_DATA_DIR)) >= 10: 
            data_dir_ok = True
    else: 
        if 'PREPROCESSED_SSL_DATA_DIR' not in missing_attrs_from_config: 
             missing_attrs_from_config.append('PREPROCESSED_SSL_DATA_DIR')

    if missing_attrs_from_config or not data_dir_ok:
        if missing_attrs_from_config:
            print(f"ERROR: Missing critical attributes in config_ssl_pretrain.py: {list(set(missing_attrs_from_config))}")
        if not data_dir_ok:
            print(f"SSL data directory '{getattr(cfg_ssl, 'PREPROCESSED_SSL_DATA_DIR', 'NOT DEFINED IN CONFIG')}' "
                  f"is not valid (or seems empty/insufficient).")
        print("Please ensure config_ssl_pretrain.py is correctly set up (with ALL_CAPS for masking percentages, etc.) "
              "and dataset_utils.py has run successfully to populate the SSL data directory.")
    else:
        pretrain_nmt_eegpt()