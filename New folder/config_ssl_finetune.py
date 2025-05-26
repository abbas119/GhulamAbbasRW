# NMT_EEGPT_Project/config_ssl_finetune.py
import torch
import os

# --- Dataset Paths ---
# Uses the same PROCESSED_DATA_SUPERVISED_DIR as supervised baselines for fair comparison
PROCESSED_DATA_FINETUNE_DIR = 'data/processed_nmt_supervised/' # Labeled segments for fine-tuning/probing

# --- Path to Pretrained NMT-EEGPT Model ---
# This path will point to the best checkpoint from the pretraining phase
PRETRAINED_NMT_EEGPT_ENCODER_PATH = 'models/saved_ssl_pretrain/nmt_eegpt_best_encoder.pt' # Example
# Or, if saving the whole pretrain model:
# PRETRAINED_NMT_EEGPT_MODEL_PATH = 'models/saved_ssl_pretrain/nmt_eegpt_epoch_XXX.pt'


# --- Data Parameters (should match supervised config for downstream task) ---
N_CHANNELS_INPUT_TO_MODEL = 21 # Original channels from user's data before ASF
N_CHANNELS_AFTER_ASF = 19      # Channels expected by the NMT-EEGPT encoder core
TARGET_SFREQ = 256
SEGMENT_DURATION_SEC = 4.0
INPUT_TIME_LENGTH = int(TARGET_SFREQ * SEGMENT_DURATION_SEC)
N_CLASSES = 2

# --- NMT-EEGPT Downstream Model Parameters (from pretraining, for loading structure) ---
# These must match the architecture of the loaded pretrained encoder
EMBED_DIM = 256             # Must match `config_ssl_pretrain.EMBED_DIM`
NUM_SUMMARY_TOKENS = 1      # Must match `config_ssl_pretrain.NUM_SUMMARY_TOKENS`
# Other encoder-specific params if needed to reconstruct the classifier model structure

USE_ADAPTIVE_SPATIAL_FILTER = True # As per EEGPT Fig 3 for downstream tasks [cite: 465]

# --- Fine-tuning / Linear Probing Parameters ---
FINETUNE_MODE = 'linear_probe' # Options: 'linear_probe', 'full_finetune', 'partial_finetune_Xlayers'

INIT_LR_FINETUNE = 1e-3       # Higher for linear probe, much lower for full fine-tune (e.g., 1e-5)
BATCH_SIZE_FINETUNE = 32
MAX_EPOCHS_FINETUNE = 100
PATIENCE_EARLY_STOPPING_FINETUNE = 20

# Optimizer for fine-tuning
OPTIMIZER_FINETUNE = 'AdamW'
WEIGHT_DECAY_FINETUNE = 1e-4

# LR Scheduler for fine-tuning
USE_SCHEDULER_FINETUNE = True
SCHEDULER_PATIENCE_FINETUNE = 5
SCHEDULER_FACTOR_FINETUNE = 0.1
MIN_LR_FINETUNE = 1e-7

# Class imbalance handling (same as supervised)
CLASS_WEIGHTS_FINETUNE = [1.0, 4.0] # Example
FOCAL_LOSS_ALPHA_FINETUNE = 0.25
FOCAL_LOSS_GAMMA_FINETUNE = 2.0

CUDA = torch.cuda.is_available()
MONITOR_METRIC_FINETUNE = 'val_f1_score'

# --- Paths for Saving ---
LOG_DIR_SSL_FINETUNE = 'logs/ssl_finetune/'
MODEL_SAVE_DIR_SSL_FINETUNE = 'models/saved_ssl_finetune/'
os.makedirs(LOG_DIR_SSL_FINETUNE, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR_SSL_FINETUNE, exist_ok=True)