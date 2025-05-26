# NMT_EEGPT_Project/visualize_results.py
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc as sklearn_auc
import pandas as pd
import seaborn as sns
import logging

# --- Matplotlib Style ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define output directory for plots
VISUALIZATION_DIR = 'logs/visualizations/'
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

EVAL_LOG_DIR = 'logs/evaluation_results/' # Where evaluate_models.py saves its outputs

# Setup basic logging for this script
vis_log_file = os.path.join(VISUALIZATION_DIR, 'visualization_log.txt')
logging.basicConfig(
    filename=vis_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler_vis = logging.StreamHandler()
console_handler_vis.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(console_handler_vis)


def plot_confusion_matrix_custom(labels, predictions, model_name_str, class_names=['Normal', 'Abnormal'], save_dir=VISUALIZATION_DIR):
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, ax=ax, 
                xticklabels=class_names, 
                yticklabels=class_names)
    ax.set_title(f'Confusion Matrix - {model_name_str}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved confusion matrix for {model_name_str} to {save_path}")

def plot_roc_curve_custom(labels, probabilities_class1, model_name_str, save_dir=VISUALIZATION_DIR):
    if len(np.unique(labels)) < 2:
        logging.warning(f"Skipping ROC curve for {model_name_str}: only one class present in labels.")
        return
        
    fpr, tpr, _ = roc_curve(labels, probabilities_class1)
    roc_auc_val = sklearn_auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name_str}')
    plt.legend(loc='lower right')
    plt.grid(True)
    save_path = os.path.join(save_dir, f'roc_curve_{model_name_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved ROC curve for {model_name_str} to {save_path}")

def parse_training_log_robust(log_file_path):
    epochs_data = []
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if "Epoch" not in line or "Loss" not in line: # Basic filter
                    continue
                
                epoch_data = {}
                # Try to extract epoch number
                import re
                epoch_match = re.search(r"Epoch\s*(\d+)", line)
                if epoch_match:
                    epoch_data['epoch'] = int(epoch_match.group(1))
                else:
                    continue # Need epoch number

                # Extract key-value pairs (e.g., "Train Loss: 0.1234", "Val Acc: 0.85")
                # This regex is more flexible: finds "Key Name: float_value"
                metrics_found = re.findall(r"([A-Za-z\s_().]+):\s*([\d\.]+)", line)
                for key, val_str in metrics_found:
                    try:
                        metric_name = key.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('(l_a)', 'la').replace('(l_r)', 'lr')
                        # Common variations
                        if "train_loss" in metric_name : metric_name = "train_loss"
                        if "val_loss" in metric_name: metric_name = "val_loss"
                        if "val_acc" in metric_name: metric_name = "val_acc"
                        if "val_f1" in metric_name: metric_name = "val_f1" # Covers val_f1_score
                        if "val_recall" in metric_name: metric_name = "val_recall"
                        if "align_loss_la" in metric_name: metric_name = "align_loss_la"
                        if "recon_loss_lr" in metric_name: metric_name = "recon_loss_lr"
                        if "total_loss" in metric_name and "pretrain" in log_file_path: metric_name = "total_ssl_loss"


                        epoch_data[metric_name] = float(val_str)
                    except ValueError:
                        logging.warning(f"Could not parse value for '{key}' in line: {line.strip()}")
                
                if len(epoch_data) > 1: # Must have epoch and at least one metric
                    epochs_data.append(epoch_data)

    except FileNotFoundError:
        logging.warning(f"Training log file {log_file_path} not found.")
    except Exception as e:
        logging.error(f"Error parsing log file {log_file_path}: {e}", exc_info=True)
        
    return pd.DataFrame(epochs_data)


def plot_training_curves_flexible(model_name_for_plot, training_log_file_path, save_dir=VISUALIZATION_DIR):
    log_df = parse_training_log_robust(training_log_file_path)

    if log_df.empty or 'epoch' not in log_df.columns:
        logging.warning(f"No data or no 'epoch' column parsed from {training_log_file_path}. Skipping training curves for {model_name_for_plot}.")
        return

    # Determine which metrics are available for plotting
    available_metrics = [col for col in log_df.columns if col != 'epoch']
    if not available_metrics:
        logging.warning(f"No metric columns found in parsed log for {model_name_for_plot}. Skipping.")
        return

    num_plots = len(available_metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes] # Make it iterable

    for idx, metric_col in enumerate(available_metrics):
        try:
            axes[idx].plot(log_df['epoch'], log_df[metric_col], label=metric_col.replace('_', ' ').title(), lw=2)
            axes[idx].set_ylabel(metric_col.replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(True)
        except Exception as e:
            logging.error(f"Error plotting metric '{metric_col}' for {model_name_for_plot}: {e}")
            
    if num_plots > 0:
        axes[-1].set_xlabel('Epoch')
        fig.suptitle(f'Training Curves - {model_name_for_plot}', fontsize=16)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        save_path = os.path.join(save_dir, f'training_curves_{model_name_for_plot}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved training curves for {model_name_for_plot} to {save_path}")
    else:
        logging.warning(f"Skipped saving training curves for {model_name_for_plot} as no metrics were plotted.")
    plt.close(fig)


def main_visualize_for_model(model_eval_name, training_log_path):
    """
    Generates standard evaluation plots for a given model.
    Args:
        model_eval_name (str): Name used in evaluation output files (e.g., 'CTNet', 'NMT_EEGPT_Classifier_linear_probe').
        training_log_path (str): Full path to the specific training log file for this model.
    """
    logging.info(f"\n--- Generating visualizations for: {model_eval_name} ---")
    
    metrics_path = os.path.join(EVAL_LOG_DIR, f'evaluation_metrics_{model_eval_name}.json')
    predictions_path = os.path.join(EVAL_LOG_DIR, f'predictions_{model_eval_name}.npy')
    labels_path = os.path.join(EVAL_LOG_DIR, f'labels_{model_eval_name}.npy')
    probabilities_path = os.path.join(EVAL_LOG_DIR, f'probabilities_{model_eval_name}.npy')

    if not all(os.path.exists(p) for p in [metrics_path, predictions_path, labels_path, probabilities_path]):
        logging.error(f"Missing evaluation output files in {EVAL_LOG_DIR} for {model_eval_name}. Run evaluate_models.py first.")
        return

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f) 
        predictions = np.load(predictions_path)
        labels = np.load(labels_path)
        probabilities_class1 = np.load(probabilities_path)

        plot_confusion_matrix_custom(labels, predictions, model_eval_name)
        plot_roc_curve_custom(labels, probabilities_class1, model_eval_name)
        
        if training_log_path and os.path.exists(training_log_path):
            plot_training_curves_flexible(model_eval_name, training_log_path)
        else:
            logging.warning(f"Training log path not provided or not found: {training_log_path}. Skipping training curves.")
            
        logging.info(f"--- Standard visualizations complete for: {model_eval_name} ---")
        logging.info(f"XAI plots should be generated via xai/generate_xai_eegpt.py for NMT-EEGPT models.")

    except Exception as e:
        logging.error(f"Error during visualization for {model_eval_name}: {e}", exc_info=True)


# Example in visualize_results.py
# Inside visualize_results.py
# ... (rest of the script) ...

if __name__ == "__main__":
    # Ensure logging for visualize_results.py itself is set up if not already
    # This basicConfig will apply if no handlers are already on the root logger.
    # If your training scripts' loggers add console handlers to the root, this might be redundant or controlled there.
    if not logging.getLogger().hasHandlers(): # Check if root logger has any handlers
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    # Alternatively, ensure the specific logger used in main_visualize_for_model is configured.
    # For simplicity here, we'll rely on the above or that the main_visualize_for_model's logger gets set up.

    print("--- Generating visualizations for HybridCNNTransformer ---")
    main_visualize_for_model(
        model_eval_name='HybridCNNTransformer', 
        training_log_path='logs/supervised_baselines/training_log_HybridCNNTransformer.txt'
    )

    print("\n--- Generating visualizations for CTNet ---")
    main_visualize_for_model(
        model_eval_name='CTNet', 
        training_log_path='logs/supervised_baselines/training_log_CTNet.txt'
    )

    print("\n--- Generating visualizations for EEGConformer ---")
    main_visualize_for_model(
        model_eval_name='EEGConformer', 
        training_log_path='logs/supervised_baselines/training_log_EEGConformer.txt'
    )

    # You can also add a call to visualize the NMT-EEGPT pretraining losses later,
    # once pretraining is done.
    # print("\n--- Generating SSL Pretraining loss curves ---")
    # main_visualize_for_model(
    #    model_eval_name='NMT_EEGPT_Pretrain_Losses', # This won't have CM/ROC
    #    training_log_path='logs/ssl_pretrain/pretraining_log_NMT_EEGPT_Pretrain.txt'
    # )
    print("\n--- All requested visualizations generation attempted. Check logs/visualizations/ ---")
    print("To use: call main_visualize_for_model(model_eval_name, training_log_path) for each model.")
    print("Ensure evaluate_models.py has been run for the model if you expect CM/ROC, and training logs exist.")
    print("Run this script from the NMT_EEGPT_Project root directory.")