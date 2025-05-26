# NMT_EEGPT_Project/xai/generate_xai_eegpt.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Adjust paths if your project structure is different for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root

import config_ssl_finetune as cfg_ft_xai
import config_ssl_pretrain as cfg_ssl_model_structure_xai
import config_supervised as cfg_sup_data_xai # For channel names, data structure

from dataset_supervised import SupervisedNMTDataset
from models.nmt_eegpt_downstream_model import NMT_EEGPT_Classifier
from xai.xai_utils import compute_channel_embedding_similarity_matrix, get_attention_maps_from_model #, get_shap_explanations

XAI_OUTPUT_DIR = 'logs/xai_explanations/'
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)

# Setup logging
xai_log_file = os.path.join(XAI_OUTPUT_DIR, 'xai_generation_log.txt')
logging.basicConfig(
    filename=xai_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler_xai = logging.StreamHandler()
console_handler_xai.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
     logging.getLogger().addHandler(console_handler_xai)


def generate_and_save_xai(downstream_model_path, num_samples_to_explain=3, target_class_for_xai=1):
    logging.info(f"--- Starting XAI Generation for NMT-EEGPT model: {downstream_model_path} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Trained NMT-EEGPT Downstream Classifier Model
    logging.info("Loading trained NMT-EEGPT classifier model...")
    # Ensure config params match the model being loaded
    model = NMT_EEGPT_Classifier(
        n_channels_model=cfg_ssl_model_structure_xai.N_CHANNELS_MODEL,
        segment_time_len_samples=cfg_ssl_model_structure_xai.INPUT_TIME_LENGTH_MODEL,
        patch_time_len_samples=cfg_ssl_model_structure_xai.PATCH_TIME_LENGTH_SAMPLES,
        embed_dim=cfg_ssl_model_structure_xai.EMBED_DIM,
        encoder_layers=cfg_ssl_model_structure_xai.ENCODER_LAYERS,
        num_heads=cfg_ssl_model_structure_xai.NUM_HEADS,
        ff_dim=cfg_ssl_model_structure_xai.FEEDFORWARD_DIM,
        dropout_transformer=cfg_ssl_model_structure_xai.DROPOUT_PRETRAIN,
        num_summary_tokens=cfg_ssl_model_structure_xai.NUM_SUMMARY_TOKENS,
        n_classes=cfg_ft_xai.N_CLASSES,
        use_adaptive_spatial_filter=cfg_ft_xai.USE_ADAPTIVE_SPATIAL_FILTER,
        n_input_channels_to_asf=cfg_ft_xai.N_CHANNELS_INPUT_TO_MODEL,
        pretrained_encoder_path=None, 
        freeze_encoder=False 
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(downstream_model_path, map_location=device))
        logging.info(f"Successfully loaded downstream model weights from {downstream_model_path}")
    except Exception as e:
        logging.error(f"Error loading model weights from {downstream_model_path}: {e}", exc_info=True)
        return
    model.eval()

    # --- XAI Method 1: Channel Embedding Similarity ---
    logging.info("Generating Channel Embedding Similarity...")
    try:
        # Accessing channel embeddings from the loaded model structure
        # This assumes NMT_EEGPT_Classifier contains feature_extractor_base, 
        # which is an instance of Pretrain_NMT_EEGPT, which has an embedding_layer.
        if hasattr(model, 'feature_extractor_base') and \
           hasattr(model.feature_extractor_base, 'embedding_layer') and \
           hasattr(model.feature_extractor_base.embedding_layer, 'channel_embed'):
            
            channel_embed_weights = model.feature_extractor_base.embedding_layer.channel_embed.weight
            sim_matrix = compute_channel_embedding_similarity_matrix(channel_embed_weights)
            
            if sim_matrix is not None:
                # Get channel names (e.g., the 19 standard channels the model was pretrained on)
                model_internal_channel_names = cfg_sup_data_xai.TARGET_CHANNELS_10_20[:cfg_ssl_model_structure_xai.N_CHANNELS_MODEL]

                plt.figure(figsize=(10, 8))
                sns.heatmap(sim_matrix, annot=False, cmap='viridis', 
                            xticklabels=model_internal_channel_names, yticklabels=model_internal_channel_names)
                plt.title(f'NMT-EEGPT Channel Embedding Cosine Similarity')
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                plt.tight_layout()
                save_path = os.path.join(XAI_OUTPUT_DIR, f'channel_sim_{os.path.basename(downstream_model_path)}.png')
                plt.savefig(save_path, dpi=300); plt.close()
                logging.info(f"Saved channel embedding similarity to {save_path}")
        else:
            logging.warning("Could not find channel_embed weights in the model structure for similarity plot.")
    except Exception as e:
        logging.error(f"Failed to generate channel embedding similarity: {e}", exc_info=True)

    # --- Load Sample Data for Instance-Level XAI ---
    if num_samples_to_explain > 0:
        logging.info(f"Loading {num_samples_to_explain} sample(s) for instance-level XAI...")
        # Use 'eval' split to get samples model hasn't fine-tuned on (if split was different for finetune)
        # Or use 'train' split if you want to see explanations on training examples
        xai_sample_dataset = SupervisedNMTDataset(
            data_dir=cfg_ft_xai.PROCESSED_DATA_FINETUNE_DIR, # Using supervised data dir
            split_type='eval', augment=False, # No augmentation for XAI samples
            segment_duration_sec=cfg_ft_xai.SEGMENT_DURATION_SEC,
            target_sfreq=cfg_ft_xai.TARGET_SFREQ,
            test_mode_reduce_data=True, 
            n_recordings_test_mode=max(2, num_samples_to_explain) # Get a few recordings to pick samples from
        )
        if len(xai_sample_dataset) == 0:
            logging.error("XAI sample dataset is empty. Cannot proceed with instance-level XAI.")
            return
        
        # Ensure we don't try to explain more samples than available
        num_samples_to_explain = min(num_samples_to_explain, len(xai_sample_dataset))
        
        sample_indices = np.random.choice(len(xai_sample_dataset), num_samples_to_explain, replace=False)
        
        # --- XAI Method 2: Attention Map Visualization (Conceptual) ---
        # This requires identifying the MHA layers in your NMT-EEGPT encoder.
        # Example: if encoder is 'model.feature_extractor_base.online_encoder.transformer_encoder'
        # and it has 'layers.0.self_attn', 'layers.1.self_attn', etc.
        # target_mha_layers = [f'feature_extractor_base.online_encoder.transformer_encoder.layers.{i}.self_attn' for i in range(cfg_ssl_model_structure_xai.ENCODER_LAYERS)]
        target_mha_layers = [] # Populate this with actual layer names from your model instance
        if target_mha_layers: # Only if specific MHA layers are identified
            for i in range(num_samples_to_explain):
                sample_idx = sample_indices[i]
                sample_data, sample_label = xai_sample_dataset[sample_idx]
                sample_data_batch = sample_data.unsqueeze(0).to(device) # (1, C_input, T)

                logging.info(f"Conceptual: Generating Attention Map for sample {sample_idx}, true label {sample_label.item()}...")
                # attention_data_dict = get_attention_maps_from_model(model, sample_data_batch, target_mha_layers)
                # for layer_name, att_weights in attention_data_dict.items():
                #     # att_weights: (B, H, Q, K) -> for B=1 -> (H, Q, K)
                #     # Process and plot (e.g., average over heads, visualize for summary tokens)
                #     # This needs a specific plotting function based on what Q and K represent (patches, summary tokens)
                #     # For example, if Q is summary token and K are patches:
                #     # avg_head_att_summary_to_patches = att_weights.mean(dim=0)[0, :].cpu().numpy() # Att from 1st summary to all patches
                #     # plot_attention_map_summary(avg_head_att_summary_to_patches, f"NMT-EEGPT_{layer_name}", ...)
                #     pass
        else:
            logging.warning("Target MHA layer names for attention map extraction are not defined. Skipping attention maps.")


        # --- XAI Method 3: SHAP / LIME (Conceptual) ---
        logging.info("Conceptual: Generating SHAP explanations (requires SHAP library & setup)...")
        # This is complex to implement generically here.
        # background_loader = DataLoader(xai_sample_dataset, batch_size=32, shuffle=True) # Use a subset for background
        # samples_to_explain_list = [xai_sample_dataset[i][0] for i in sample_indices]
        # samples_to_explain_tensor = torch.stack(samples_to_explain_list).to(device)
        # sample_loader_for_shap = DataLoader(torch.utils.data.TensorDataset(samples_to_explain_tensor), batch_size=1)

        # shap_results = get_shap_explanations(model, background_loader, sample_loader_for_shap, device)
        # if shap_results:
        #     for sample_idx_in_batch, shap_values_for_classes in shap_results.items():
        #         original_sample_idx = sample_indices[sample_idx_in_batch]
        #         # shap_values_for_target_class = shap_values_for_classes[target_class_for_xai] # (1, C, T)
        #         # original_data = xai_sample_dataset[original_sample_idx][0].cpu().numpy()
        #         # Process and plot shap_values_for_target_class against original_data
        #         logging.info(f"Generated SHAP for original sample index {original_sample_idx} (conceptual).")

    logging.info(f"--- XAI Generation Finished for {os.path.basename(downstream_model_path)} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate XAI explanations for trained NMT-EEGPT models.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved NMT-EEGPT downstream classifier model (.pt file).")
    parser.add_argument('--num_samples', type=int, default=3,
                        help="Number of samples from eval set to explain.")
    parser.add_argument('--target_class', type=int, default=1,
                        help="Target class index for explanations (e.g., 1 for 'abnormal').")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Trained NMT-EEGPT downstream model not found at {args.model_path}")
    else:
        generate_and_save_xai(
            downstream_model_path=args.model_path,
            num_samples_to_explain=args.num_samples,
            target_class_idx_for_attention=args.target_class
        )
    # Example:
    # python xai/generate_xai_eegpt.py --model_path models/saved_ssl_finetune/nmt_eegpt_downstream_linear_probe_best.pt --num_samples 3