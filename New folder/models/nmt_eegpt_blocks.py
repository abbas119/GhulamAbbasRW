# NMT_EEGPT_Project/models/nmt_eegpt_blocks.py
import torch
import torch.nn as nn
# from einops import rearrange # Not strictly needed in this file with current LSTE design

class LocalSpatioTemporalEmbedding(nn.Module):
    def __init__(self, n_channels_model, num_time_patches_per_channel, 
                 patch_time_len_samples, embed_dim, pos_embed_dropout=0.1):
        super().__init__()
        self.patch_linear_projector = nn.Linear(patch_time_len_samples, embed_dim)
        self.channel_embed = nn.Embedding(n_channels_model, embed_dim)
        self.total_patches_per_segment = n_channels_model * num_time_patches_per_channel
        self.temporal_patch_sequence_pos_embed = nn.Parameter(
            torch.randn(1, self.total_patches_per_segment, embed_dim)
        )
        self.dropout = nn.Dropout(pos_embed_dropout)

    def forward(self, x_patches_flat, channel_ids_flat):
        patch_content_embed = self.patch_linear_projector(x_patches_flat)
        ch_embeds = self.channel_embed(channel_ids_flat)
        tokens = patch_content_embed + ch_embeds
        if tokens.size(1) == self.total_patches_per_segment:
            tokens = tokens + self.temporal_patch_sequence_pos_embed
        else:
            # Handle cases where only a subset of patches (e.g., unmasked) might be passed
            # This part of LSTE might need to be more flexible or only apply pos_embed to full sequence
            # For now, assuming it gets the full sequence or pre-embedded tokens with pos info already
            pass
        return self.dropout(tokens)

class NMT_EEGPT_TransformerModule(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout_transformer,
            batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x_tokens): 
        return self.transformer_encoder(x_tokens)

class EEGPT_Style_Encoder(NMT_EEGPT_TransformerModule):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, num_summary_tokens=1):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        self.num_summary_tokens_config = num_summary_tokens # How many S tokens are expected to be prepended
        if self.num_summary_tokens_config > 0:
            self.summary_tokens = nn.Parameter(torch.randn(1, self.num_summary_tokens_config, embed_dim))
        # This encoder does not add summary tokens itself; it expects them in the input.

    def forward(self, x_tokens_with_summary_already_prepended):
        # Assumes x_tokens_with_summary_already_prepended is shape (B, S + NumPatches, Dim)
        # It just processes this sequence.
        return super().forward(x_tokens_with_summary_already_prepended)

class EEGPT_Style_Predictor(NMT_EEGPT_TransformerModule):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, num_query_tokens_for_masked=0):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        self.num_query_tokens = num_query_tokens_for_masked
        if self.num_query_tokens > 0:
             self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, embed_dim))

    def forward(self, x_encoded_patch_features):
        # Input x_encoded_patch_features is (B, NumPatches, EmbedDim)
        # (i.e., AFTER summary tokens have been sliced off from encoder output)
        return super().forward(x_encoded_patch_features)

class EEGPT_Style_Reconstructor(NMT_EEGPT_TransformerModule):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, original_patch_dim):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        self.to_raw_patch_projection = nn.Linear(embed_dim, original_patch_dim)

    def forward(self, x_combined_features_for_reconstruction):
        # Input is (B, NumPatches, EmbedDim)
        reconstructed_token_embeddings = super().forward(x_combined_features_for_reconstruction)
        return reconstructed_token_embeddings # Projection to raw patch done in main model