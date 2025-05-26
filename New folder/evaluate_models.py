# NMT_EEGPT_Project/evaluate_models.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import argparse
import json

# Import configurations, datasets, models, and monitors
import config_supervised as cfg_sup # For supervised model params and data config
import config_ssl_finetune as cfg_ft # For NMT_EEGPT_Classifier structure if needed
import config_ssl_pretrain as cfg_ssl_pretrain_model_params # For NMT_EEGPT_Pretrain structure if NMT_EEGPT_Classifier loads it

from dataset_supervised import SupervisedNMTDataset
from models.hybrid_cnn_transformer import HybridCNNTransformer
from models.ctnet_model import CTNet
from models.eeg_conformer_model import EEGConformer
from models.nmt_eegpt_downstream_model import NMT_EEGPT_Classifier # For loading NMT-EEGPT
from monitors import PerformanceMetricsMonitor

EVAL_LOG_DIR = 'logs/evaluation_results/'
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

# Setup logger uniquely for this module
logger_eval = logging.getLogger('evaluate_models_logger')
if not logger_eval.handlers:
    logger_eval.setLevel(logging.INFO)
    ch_eval = logging.StreamHandler()
    ch_eval.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s'))
    logger_eval.addHandler(ch_eval)

def setup_eval_logging(log_dir, model_name_str): # File logger
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'evaluation_log_{model_name_str}.txt')
    
    file_logger = logging.getLogger(f'{model_name_str}_eval_file_logger')
    if file_logger.hasHandlers(): file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w') # Overwrite for each eval run
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    file_logger.addHandler(fh)
    return file_logger

def evaluate_single_model(model_type, model_path_to_load):
    current_run_logger = setup_eval_logging(EVAL_LOG_DIR, model_type)
    current_run_logger.info(f"--- Starting Evaluation for: {model_type} from {model_path_to_load} ---")
    
    if not os.path.exists(model_path_to_load):
        current_run_logger.error(f"Model path not found: {model_path_to_load}. Exiting.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_run_logger.info(f"Using device: {device}")

    # --- Data Loading (NMT 'eval' split from config_supervised) ---
    current_run_logger.info("Loading EVALUATION dataset (NMT 'eval' split)...")
    eval_dataset = SupervisedNMTDataset(
        data_dir=cfg_sup.PROCESSED_DATA_SUPERVISED_DIR,
        split_type='eval', 
        augment=False,
        segment_duration_sec=cfg_sup.SEGMENT_DURATION_SEC,
        target_sfreq=cfg_sup.TARGET_SFREQ,
        test_mode_reduce_data=False # Always evaluate on full eval set
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg_sup.BATCH_SIZE, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    current_run_logger.info(f"Evaluation segments: {len(eval_dataset)}")
    if len(eval_dataset) == 0:
        current_run_logger.error("Evaluation dataset is empty. Ensure dataset_utils.py has run and `eval` split exists.")
        return

    # --- Model Initialization & Loading Weights ---
    model = None
    if model_type == 'HybridCNNTransformer':
        model = HybridCNNTransformer(
            n_channels=cfg_sup.N_CHANNELS_SELECTED, n_start_chans=cfg_sup.HYBRID_N_START_CHANS,
            n_layers_transformer=cfg_sup.HYBRID_N_LAYERS_TRANSFORMER, n_heads=cfg_sup.HYBRID_N_HEADS,
            hidden_dim=cfg_sup.HYBRID_HIDDEN_DIM, ff_dim=cfg_sup.HYBRID_FF_DIM,
            dropout=cfg_sup.HYBRID_DROPOUT, input_time_length=cfg_sup.INPUT_TIME_LENGTH,
            n_classes=cfg_sup.N_CLASSES
        )
    elif model_type == 'CTNet':
        model = CTNet(
            n_channels=cfg_sup.N_CHANNELS_SELECTED, n_classes=cfg_sup.N_CLASSES,
            input_time_length=cfg_sup.INPUT_TIME_LENGTH, target_sfreq=cfg_sup.TARGET_SFREQ,
            f1=cfg_sup.CTNET_F1, d_multiplier=cfg_sup.CTNET_D, f2=cfg_sup.CTNET_F2,
            kc1_divisor=4, pool1_size=cfg_sup.CTNET_P1, k2_kernel_length=cfg_sup.CTNET_K2, 
            pool2_size=cfg_sup.CTNET_P2, transformer_depth=cfg_sup.CTNET_TRANSFORMER_DEPTH, 
            transformer_heads=cfg_sup.CTNET_TRANSFORMER_HEADS, dropout_cnn=cfg_sup.CTNET_DROPOUT_CNN, 
            dropout_transformer_p1=cfg_sup.CTNET_DROPOUT_TRANSFORMER, 
            dropout_classifier_p2=cfg_sup.CTNET_DROPOUT_CLASSIFIER
        )
    elif model_type == 'EEGConformer':
        model = EEGConformer( # Params from config_supervised
            n_channels=cfg_sup.N_CHANNELS_SELECTED, 
            n_classes=cfg_sup.N_CLASSES,
            input_time_length=cfg_sup.INPUT_TIME_LENGTH, 
            target_sfreq=cfg_sup.TARGET_SFREQ,
            n_filters_time=getattr(cfg_sup, 'CONFORMER_N_FILTERS_TIME', 40), 
            filter_time_length_ms=getattr(cfg_sup, 'CONFORMER_FILTER_TIME_LENGTH_MS', 100), 
            n_filters_spat=getattr(cfg_sup, 'CONFORMER_N_FILTERS_SPAT', 40), 
            pool_time_length_ms=getattr(cfg_sup, 'CONFORMER_POOL_TIME_LENGTH_MS', 300), 
            pool_time_stride_ms=getattr(cfg_sup, 'CONFORMER_POOL_TIME_STRIDE_MS', 60),  
            cnn_drop_prob=getattr(cfg_sup, 'CONFORMER_CNN_DROP_PROB', getattr(cfg_sup, 'CONFORMER_DROPOUT', 0.3)),
            # --- THIS IS THE CORRECTED ARGUMENT ---
            transformer_d_model_explicit=getattr(cfg_sup, 'CONFORMER_TRANSFORMER_D_MODEL_EXPLICIT', None), 
            # ---                               ---
            transformer_depth=getattr(cfg_sup, 'CONFORMER_TRANSFORMER_DEPTH', 3), 
            transformer_n_heads=getattr(cfg_sup, 'CONFORMER_TRANSFORMER_N_HEADS', 4),
            transformer_ff_dim_factor=getattr(cfg_sup, 'CONFORMER_TRANSFORMER_FF_DIM_FACTOR', 2),
            transformer_drop_prob=getattr(cfg_sup, 'CONFORMER_TRANSFORMER_DROP_PROB', getattr(cfg_sup, 'CONFORMER_DROPOUT', 0.1)),
            classifier_hidden_dim=getattr(cfg_sup, 'CONFORMER_CLASSIFIER_HIDDEN_DIM', 128),
            classifier_drop_prob=getattr(cfg_sup, 'CONFORMER_CLASSIFIER_DROP_PROB', getattr(cfg_sup, 'CONFORMER_DROPOUT', 0.3))
        )
    elif model_type == 'NMT_EEGPT_Classifier':
        # For NMT_EEGPT_Classifier, structure params come from cfg_ssl_pretrain_model_params
        # and downstream task specific params (like n_classes) from cfg_ft
        model = NMT_EEGPT_Classifier(
            n_channels_model=cfg_ssl_pretrain_model_params.N_CHANNELS_MODEL,
            segment_time_len_samples=cfg_ssl_pretrain_model_params.INPUT_TIME_LENGTH_MODEL,
            patch_time_len_samples=cfg_ssl_pretrain_model_params.PATCH_TIME_LENGTH_SAMPLES,
            embed_dim=cfg_ssl_pretrain_model_params.EMBED_DIM,
            encoder_layers=cfg_ssl_pretrain_model_params.ENCODER_LAYERS,
            num_heads=cfg_ssl_pretrain_model_params.NUM_HEADS,
            ff_dim=cfg_ssl_pretrain_model_params.FEEDFORWARD_DIM,
            dropout_transformer=cfg_ssl_pretrain_model_params.DROPOUT_PRETRAIN,
            num_summary_tokens=cfg_ssl_pretrain_model_params.NUM_SUMMARY_TOKENS,
            n_classes=cfg_ft.N_CLASSES, 
            use_adaptive_spatial_filter=cfg_ft.USE_ADAPTIVE_SPATIAL_FILTER,
            n_input_channels_to_asf=cfg_ft.N_CHANNELS_INPUT_TO_MODEL,
            pretrained_model_path=None, # Not needed for loading full state_dict directly
            freeze_encoder=False # Does not matter for eval
        )
    else:
        current_run_logger.error(f"Unknown model_type: {model_type}")
        return

    try:
        # Load the state dictionary. For NMT_EEGPT_Classifier, ensure the saved checkpoint
        # contains the state_dict for this NMT_EEGPT_Classifier class, not just the pretrain model.
        # If PRETRAINED_NMT_EEGPT_ENCODER_PATH in NMT_EEGPT_Classifier was used to load parts during its training,
        # then model_path_to_load here should be the final saved NMT_EEGPT_Classifier.
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        current_run_logger.info(f"Successfully loaded model weights from {model_path_to_load}")
    except Exception as e:
        current_run_logger.error(f"Error loading model weights from {model_path_to_load}: {e}", exc_info=True)
        return
        
    model.to(device)
    model.eval()

    # --- Evaluate ---
    current_run_logger.info("Evaluating model on the test set...")
    eval_monitor = PerformanceMetricsMonitor(model, eval_loader, device, criterion=None)
    metrics = eval_monitor.evaluate()

    # --- Log and Save Metrics ---
    current_run_logger.info(f"--- Evaluation Metrics for {model_type} ---")
    for key, value in metrics.items():
        if key not in ['predictions_list', 'labels_list', 'probabilities_class1_list']:
            log_value = value
            if isinstance(value, np.ndarray): # Convert CM to string for simple logging
                log_value = "\n" + np.array2string(value)
            current_run_logger.info(f"  {key}: {log_value}")
            # print(f"  {key}: {log_value}") # Optional print to console
        elif key == 'confusion_matrix' and isinstance(value, np.ndarray): # Already handled if ndarray
             pass # current_run_logger.info(f"  Confusion Matrix:\n{np.array2string(value)}")

    metrics_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    metrics_save_path = os.path.join(EVAL_LOG_DIR, f'evaluation_metrics_{model_type}.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    current_run_logger.info(f"Saved detailed metrics to {metrics_save_path}")

    # Save raw outputs for plotting or further analysis
    # Ensure these keys exist in metrics before trying to save
    if 'predictions_list' in metrics:
        np.save(os.path.join(EVAL_LOG_DIR, f'predictions_{model_type}.npy'), np.array(metrics['predictions_list']))
    if 'labels_list' in metrics:
        np.save(os.path.join(EVAL_LOG_DIR, f'labels_{model_type}.npy'), np.array(metrics['labels_list']))
    if 'probabilities_class1_list' in metrics:
        np.save(os.path.join(EVAL_LOG_DIR, f'probabilities_{model_type}.npy'), np.array(metrics['probabilities_class1_list']))
    current_run_logger.info(f"Saved predictions, labels, and probabilities for {model_type} in {EVAL_LOG_DIR}")
    current_run_logger.info(f"--- Evaluation complete for {model_type} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained EEG classification models.")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['HybridCNNTransformer', 'CTNet', 'EEGConformer', 'NMT_EEGPT_Classifier'],
                        help="Type of the model to evaluate.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model weights (.pt file).")
    args = parser.parse_args()

    # Ensure main config directories exist
    if hasattr(cfg_sup, 'LOG_DIR_SUPERVISED'): os.makedirs(cfg_sup.LOG_DIR_SUPERVISED, exist_ok=True) # For eval log dir
    if hasattr(cfg_sup, 'MODEL_SAVE_DIR_SUPERVISED'): os.makedirs(cfg_sup.MODEL_SAVE_DIR_SUPERVISED, exist_ok=True) # Not strictly needed for eval but good practice

    evaluate_single_model(args.model_type, args.model_path)