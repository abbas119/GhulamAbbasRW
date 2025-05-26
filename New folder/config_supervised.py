# NMT_EEGPT_Project/config_supervised.py
import torch
import os

# --- Dataset Paths ---
# Original NMT dataset structure as per your setup
# Data will be preprocessed by dataset_utils.py and saved to a new processed_data_dir
BASE_NMT_DATA_PATH = 'D:/ValidData/Organized/' # Your raw NMT EDFs
PROCESSED_DATA_SUPERVISED_DIR = 'data/processed_nmt_supervised/' # Processed segments for supervised learning
os.makedirs(PROCESSED_DATA_SUPERVISED_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_SUPERVISED_DIR, 'train', 'normal'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_SUPERVISED_DIR, 'train', 'abnormal'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_SUPERVISED_DIR, 'eval', 'normal'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_SUPERVISED_DIR, 'eval', 'abnormal'), exist_ok=True)

# --- Data Preprocessing Parameters (used by dataset_utils.py) ---
N_RECORDINGS_PER_CLASS_SUPERVISED = None # None for all, or set a number for quick tests
SENSOR_TYPES = ['EEG']
# NMT paper Figure 1 shows 19 standard 10-20 scalp sites.
# We will select these 19 channels for all models for consistency.
N_CHANNELS_SELECTED = 19
TARGET_CHANNELS_10_20 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                           'F7', 'F8', 'T7', 'P7', 'T8', 'P8', 'Fz', 'Cz', 'Pz']
# Note: T7 often T3, P7 often T5, T8 often T4, P8 often T6. `dataset_utils.py` should handle mapping.

MAX_RECORDING_MINS_RAW = None # Max duration of raw EDFs to consider
SEC_TO_CUT_RAW = 60           # Initial seconds to cut from raw EDFs
DURATION_RECORDING_MINS_RAW = 15 # Duration to use from each raw EDF
TEST_RECORDING_MINS_RAW = 10    # Duration for eval set raw EDFs

MAX_ABS_VAL_CLIP = 800.0      # Clipping value for EEG uV
TARGET_SFREQ = 256            # Target sampling frequency (Hz) - Aligning with EEGPT [cite: 476]
BANDPASS_LOW = 0.5            # Bandpass filter low cutoff (Hz)
BANDPASS_HIGH = 45.0          # Bandpass filter high cutoff (Hz)
SEGMENT_DURATION_SEC = 4.0    # Duration of each segment for classification (s)
INPUT_TIME_LENGTH = int(TARGET_SFREQ * SEGMENT_DURATION_SEC) # Samples per segment

# --- Data Splitting (for supervised model training files) ---
# The NMT dataset already has train/eval splits in its folder structure.
# We'll maintain this split when saving processed segments.
SHUFFLE_TRAIN_LOADER = True

# --- Model Parameters (for your HybridCNNTransformer, CTNet, EEG Conformer) ---
# These are examples, you'll need to define them per model type if they differ significantly
# For your HybridCNNTransformer (from your config.py, adjusted for N_CHANNELS_SELECTED)
HYBRID_N_START_CHANS = 64
HYBRID_N_LAYERS_TRANSFORMER = 2
HYBRID_N_HEADS = 4
HYBRID_HIDDEN_DIM = 128 # This is d_model for transformer
HYBRID_FF_DIM = 256
HYBRID_DROPOUT = 0.4
N_CLASSES = 2

# Example for CTNet (refer to CTNet paper Table 1 for BCI IV-2a/b values and adapt)
CTNET_F1 = 8
CTNET_D = 2
CTNET_F2 = 16 # This becomes d_model for its Transformer
CTNET_KC1 = int(TARGET_SFREQ / 4) # Kernel size for temporal conv ~Fs/4
CTNET_P1 = 8 # First pooling
CTNET_K2 = 16 # Spatial conv kernel
CTNET_P2 = 8 # Second pooling (adjusts token size $T_C = T_{after\_P1} / P_2$)
CTNET_TRANSFORMER_DEPTH = 6
CTNET_TRANSFORMER_HEADS = 2
CTNET_DROPOUT_CNN = 0.25 # e.g., for cross-subject from CTNet paper
CTNET_DROPOUT_TRANSFORMER = 0.25 # p1 in CTNet paper
CTNET_DROPOUT_CLASSIFIER = 0.5   # p2 in CTNet paper

# Example for EEG Conformer (refer to Song et al., 2023 and Braindecode)
CONFORMER_N_FILTERS_TIME = 40
CONFORMER_FILTER_TIME_LENGTH = int(0.1 * TARGET_SFREQ) # e.g., 25 for 250Hz in Braindecode
CONFORMER_N_FILTERS_SPAT = 40 # Should match n_filters_time for depthwise conv structure
CONFORMER_POOL_TIME_LENGTH = int(0.3 * TARGET_SFREQ) # e.g., 75 for 250Hz
CONFORMER_POOL_TIME_STRIDE = int(0.06 * TARGET_SFREQ) # e.g., 15 for 250Hz
CONFORMER_TRANSFORMER_DEPTH = 3 # Shallower for smaller datasets initially
CONFORMER_TRANSFORMER_HEADS = 4
CONFORMER_DROPOUT = 0.3

# --- Training Parameters ---
INIT_LR = 1e-4 # Adjusted from your 1e-3, common starting point
BATCH_SIZE = 32 # Your RTX 3060 12GB might handle this for supervised models
                # If OOM, reduce batch size or use gradient accumulation
MAX_EPOCHS = 200 # Reduced from 500 for faster iteration, can be increased
PATIENCE_EARLY_STOPPING = 30 # Reduced from 100
CUDA = torch.cuda.is_available()
TEST_MODE_REDUCE_DATA = False # Set to True for quick test runs on small subset of data

MONITOR_METRIC = 'val_f1_score' # F1 is good for imbalanced data. Or use your combined acc+recall.
CHECKPOINT_INTERVAL = 10 # Save checkpoints every N epochs

# Learning Rate Scheduler (from your config.py)
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.1 # EEGPT used OneCycle, ReduceLROnPlateau is also good
MIN_LR = 1e-6

# Class imbalance handling
# For weighted CrossEntropyLoss
# Weights: inverse frequency, or manual. Example: if abnormal is class 1
# NMT: 2002 normal, 415 abnormal. Ratio ~4.8 : 1
# Weight for normal (class 0) = 1.0
# Weight for abnormal (class 1) = 4.0  (or calculate dynamically as in your train.py)
CLASS_WEIGHTS = [1.0, 4.0] # Example, adjust based on dynamic calculation or experiments
# For Focal Loss
FOCAL_LOSS_ALPHA = 0.25 # Standard value, gives more weight to hard-to-classify examples (often minority)
FOCAL_LOSS_GAMMA = 2.0  # Standard value, focusing parameter

# --- Paths for Saving ---
LOG_DIR_SUPERVISED = 'logs/supervised_baselines/'
MODEL_SAVE_DIR_SUPERVISED = 'models/saved_supervised/'
os.makedirs(LOG_DIR_SUPERVISED, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR_SUPERVISED, exist_ok=True)

# --- XAI Parameters ---
XAI_SAMPLE_COUNT = 5 # Number of samples to generate XAI for