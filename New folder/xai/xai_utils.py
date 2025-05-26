# NMT_EEGPT_Project/xai/xai_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

# --- Channel Embedding Similarity (from EEGPT paper) ---
def compute_channel_embedding_similarity_matrix(channel_embedding_weights):
    """
    Computes cosine similarity matrix for channel embeddings.
    Args:
        channel_embedding_weights (torch.Tensor or np.ndarray): Shape (n_channels, embed_dim)
    Returns:
        np.ndarray: Similarity matrix (n_channels, n_channels) or None if error.
    """
    if channel_embedding_weights is None:
        logging.error("Channel embedding weights are None.")
        return None
        
    if isinstance(channel_embedding_weights, torch.Tensor):
        channel_embedding_weights = channel_embedding_weights.detach().cpu().numpy()
    
    if channel_embedding_weights.ndim != 2 or channel_embedding_weights.shape[0] < 2:
        logging.error(f"channel_embedding_weights must be 2D with at least 2 channels. Got shape: {channel_embedding_weights.shape}")
        return None
        
    try:
        similarity_matrix = cosine_similarity(channel_embedding_weights)
        return similarity_matrix
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}", exc_info=True)
        return None


# --- Attention Map Utilities (Conceptual - requires model-specific hooks) ---

class AttentionHook:
    def __init__(self, module, layer_name=""):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.attention_weights = None
        self.layer_name = layer_name

    def hook_fn(self, module, input, output):
        # For nn.MultiheadAttention, output is (attn_output, attn_output_weights)
        # attn_output_weights shape: (batch_size, num_heads, query_len, key_len) for batch_first=False
        # or (query_len, batch_size, num_heads, key_len) ??? Needs check.
        # If batch_first=True for MHA layer, attn_weights is (batch_size, num_heads, query_len, key_len)
        # Or if it's just the attention scores before softmax, it might be different.
        # This needs to be specific to the Transformer block implementation.
        # Let's assume output[1] is the attention weights (B, H, Q_len, K_len) for MHA.
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
            self.attention_weights = output[1].detach().cpu() 
            # logging.info(f"Captured attention from {self.layer_name}, shape: {self.attention_weights.shape}")
        # else:
            # logging.warning(f"Could not capture attention from {self.layer_name}. Output type: {type(output)}")


    def close(self):
        self.hook.remove()

def get_attention_maps_from_model(model, sample_input, target_mha_layer_names):
    """
    Conceptual function to extract attention maps using hooks.
    Args:
        model (nn.Module): The model (e.g., NMT_EEGPT_Classifier, expecting its encoder part).
        sample_input (torch.Tensor): A single sample input to the model (e.g., (1, C, T)).
        target_mha_layer_names (list of str): Names of nn.MultiheadAttention layers to hook.
                                              e.g., 'feature_extractor_base.online_encoder.transformer_encoder.layers.0.self_attn'
    Returns:
        dict: Layer_name -> attention_weights_tensor
    """
    model.eval()
    attention_outputs = {}
    hooks = []

    for name, module in model.named_modules():
        if name in target_mha_layer_names and isinstance(module, nn.MultiheadAttention):
            hook = AttentionHook(module, layer_name=name)
            hooks.append(hook)
            logging.info(f"Registered hook for MHA layer: {name}")

    if not hooks:
        logging.warning("No MHA layers found or hooked for attention map extraction.")
        return attention_outputs

    with torch.no_grad():
        _ = model(sample_input) # Forward pass to trigger hooks

    for hook in hooks:
        if hook.attention_weights is not None:
            attention_outputs[hook.layer_name] = hook.attention_weights
        hook.close()
    
    return attention_outputs


# --- SHAP / LIME Related Utilities (Conceptual) ---
# These require installing `shap` or `lime` and are highly dependent on model input/output.

def get_shap_explanations(model, background_data_loader, samples_to_explain_loader, device):
    """
    Conceptual function to get SHAP explanations.
    Args:
        model (nn.Module): Trained model.
        background_data_loader (DataLoader): DataLoader for background/baseline data for SHAP.
        samples_to_explain_loader (DataLoader): DataLoader for samples to explain.
        device: torch device.
    Returns:
        dict: sample_index -> shap_values (or None if error)
    """
    try:
        import shap
    except ImportError:
        logging.error("SHAP library not installed. Cannot generate SHAP explanations.")
        return None

    model.eval()
    all_shap_explanations = {}

    # Create a background dataset (e.g., a subset of training data, ~100-200 samples)
    background_tensors_list = []
    for i, (data, _) in enumerate(background_data_loader):
        background_tensors_list.append(data)
        if i * background_data_loader.batch_size > 200: # Limit background size
            break
    if not background_tensors_list:
        logging.error("No background data for SHAP.")
        return None
    background_tensors = torch.cat(background_tensors_list).to(device)

    # Create SHAP explainer (DeepExplainer or GradientExplainer for PyTorch)
    # Model needs to output logits or probabilities for this.
    # For models with complex inputs (like patch sequences), this needs careful adaptation.
    # explainer = shap.DeepExplainer(model, background_tensors) 
    # For Transformers, GradientExplainer might be more straightforward if DeepExplainer has issues.
    # We need a wrapper if model doesn't directly take (B,C,T) and output (B,Classes) for SHAP.
    
    # This part is highly model-dependent and complex.
    logging.warning("SHAP explanation for complex EEG Transformers is advanced and not fully implemented in this utility.")
    logging.warning("You would typically use GradientExplainer with inputs and target output class.")
    # Example structure:
    # explainer = shap.GradientExplainer(model, background_tensors)
    # for idx, (sample_data, _) in enumerate(samples_to_explain_loader):
    #     sample_data = sample_data.to(device)
    #     try:
    #         shap_values_for_sample_classes = explainer.shap_values(sample_data, nsamples='auto') # nsamples for Gradient
    #         # shap_values_for_sample_classes is a list of arrays (one per output class)
    #         # Each array is typically (B, C, T) matching input shape
    #         all_shap_explanations[idx] = shap_values_for_sample_classes 
    #     except Exception as e:
    #         logging.error(f"Error getting SHAP for sample {idx}: {e}")
    #         all_shap_explanations[idx] = None
            
    return all_shap_explanations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("XAI Utilities Loaded. Contains conceptual functions for attention and SHAP.")
    
    # Example for channel embedding similarity
    dummy_embeddings = torch.randn(19, 256) # 19 channels, 256 embed_dim
    sim_matrix = compute_channel_embedding_similarity_matrix(dummy_embeddings)
    if sim_matrix is not None:
        print("Dummy Channel Embedding Similarity Matrix shape:", sim_matrix.shape)
        # In a real scenario, visualize_results.py would plot this.