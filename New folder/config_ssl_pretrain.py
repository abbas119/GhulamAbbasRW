# NMT_EEGPT_Project/config_ssl_pretrain.py
import torch
import os

# --- Dataset Paths ---
PREPROCESSED_SSL_DATA_DIR = 'data/processed_nmt_ssl/'
os.makedirs(PREPROCESSED_SSL_DATA_DIR, exist_ok=True)

# --- Data Parameters ---
N_CHANNELS_MODEL = 19
TARGET_SFREQ = 256.0        # Hz, align with EEGPT, use float
SEGMENT_DURATION_SEC = 4.0  # s, EEGPT uses 4s crops
INPUT_TIME_LENGTH_MODEL = int(TARGET_SFREQ * SEGMENT_DURATION_SEC) # Samples per 4s segment

PATCH_DURATION_MS = 250.0 # Use float
PATCH_TIME_LENGTH_SAMPLES = int(PATCH_DURATION_MS / 1000.0 * TARGET_SFREQ) # e.g., 64 samples
N_TIME_PATCHES = INPUT_TIME_LENGTH_MODEL // PATCH_TIME_LENGTH_SAMPLES # e.g., 1024 // 64 = 16

# --- Masking Strategy (Names must be ALL CAPS) ---
TIME_PATCH_MASK_PERCENTAGE = 0.5
CHANNEL_MASK_PERCENTAGE = 0.8

# --- NMT-EEGPT Model Parameters ---
EMBED_DIM = 256
ENCODER_LAYERS = 4
PREDICTOR_LAYERS = 2
RECONSTRUCTOR_LAYERS = 2
NUM_HEADS = 4 # Must be a divisor of EMBED_DIM
FEEDFORWARD_DIM = EMBED_DIM * 2
DROPOUT_PRETRAIN = 0.1
NUM_SUMMARY_TOKENS = 1
MOMENTUM_TAU = 0.01

# --- Pretraining Parameters ---
INIT_LR_PRETRAIN = 5e-4
BATCH_SIZE_PRETRAIN = 8 # Adjusted for 12GB VRAM, try 4 or 8
GRAD_ACCUMULATION_STEPS = 8 # Effective batch size = BATCH_SIZE_PRETRAIN * GRAD_ACCUMULATION_STEPS (e.g., 8*8=64)
MAX_EPOCHS_PRETRAIN = 100
OPTIMIZER_NAME = 'AdamW'
WEIGHT_DECAY_PRETRAIN = 0.05
LR_SCHEDULER_PRETRAIN = 'OneCycleLR'
MAX_LR_ONE_CYCLE = INIT_LR_PRETRAIN # Corrected name (underscore)
PCT_START_ONE_CYCLE = 0.3
PATIENCE_PRETRAIN = 50 # For SSL monitor (early stopping based on loss plateau)

CUDA = torch.cuda.is_available()
USE_AMP = True
TEST_MODE_REDUCE_DATA = False
N_SEGMENTS_TEST_MODE_SSL = 100 # Used if TEST_MODE_REDUCE_DATA is True

LOG_DIR_SSL_PRETRAIN = 'logs/ssl_pretrain/'
MODEL_SAVE_DIR_SSL_PRETRAIN = 'models/saved_ssl_pretrain/'
PRETRAIN_SAVE_EVERY_EPOCHS = 5
os.makedirs(LOG_DIR_SSL_PRETRAIN, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR_SSL_PRETRAIN, exist_ok=True)