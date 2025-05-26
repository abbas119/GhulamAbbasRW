# NMT_EEGPT_Project/models/nmt_eegpt_pretrain_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nmt_eegpt_blocks import LocalSpatioTemporalEmbedding, \
                              EEGPT_Style_Encoder, EEGPT_Style_Predictor, EEGPT_Style_Reconstructor
from einops import rearrange 

class NMT_EEGPT_Pretrain(nn.Module):
    def __init__(self, 
                 n_channels_model, 
                 segment_time_len_samples, 
                 patch_time_len_samples,
                 embed_dim, 
                 encoder_layers, predictor_layers, reconstructor_layers,
                 num_heads, ff_dim, dropout_transformer, 
                 num_summary_tokens=1, 
                 momentum_tau=0.01
                ):
        super().__init__()
        
        self.n_channels_model = n_channels_model
        self.segment_time_len_samples = segment_time_len_samples
        self.patch_time_len_samples = patch_time_len_samples
        self.num_time_patches_per_channel = segment_time_len_samples // patch_time_len_samples
        self.total_patches_per_segment = self.n_channels_model * self.num_time_patches_per_channel
        self.embed_dim = embed_dim
        self.conf_num_summary_tokens = num_summary_tokens

        self.embedding_layer = LocalSpatioTemporalEmbedding(
            n_channels_model=self.n_channels_model,
            num_time_patches_per_channel=self.num_time_patches_per_channel,
            patch_time_len_samples=self.patch_time_len_samples,
            embed_dim=self.embed_dim,
            pos_embed_dropout=dropout_transformer
        )

        self.online_encoder = EEGPT_Style_Encoder( # Encoder has its own summary token Parameters
            embed_dim, encoder_layers, num_heads, ff_dim, dropout_transformer, 
            num_summary_tokens=self.conf_num_summary_tokens 
        )
        
        self.online_predictor = EEGPT_Style_Predictor( # Predictor operates on patch features
            embed_dim, predictor_layers, num_heads, ff_dim, dropout_transformer,
            num_query_tokens_for_masked=0 
        )

        self.online_reconstructor = EEGPT_Style_Reconstructor( # Reconstructor operates on patch features
            embed_dim, reconstructor_layers, num_heads, ff_dim, dropout_transformer,
            original_patch_dim=self.patch_time_len_samples 
        )
        self.reconstructor_to_raw_patch = nn.Linear(embed_dim, self.patch_time_len_samples)


        self.momentum_tau = momentum_tau
        self.momentum_encoder = EEGPT_Style_Encoder( 
            embed_dim, encoder_layers, num_heads, ff_dim, dropout_transformer, 
            num_summary_tokens=self.conf_num_summary_tokens
        )
        self._init_momentum_encoder()
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    @torch.no_grad()
    def _init_momentum_encoder(self):
        # Copy parameters from online_encoder to momentum_encoder
        # This needs to iterate through corresponding sub-modules if architectures are identical
        # For EEGPT_Style_Encoder which inherits NMT_EEGPT_TransformerModule:
        self.momentum_encoder.transformer_encoder.load_state_dict(self.online_encoder.transformer_encoder.state_dict())
        if self.conf_num_summary_tokens > 0 and hasattr(self.online_encoder, 'summary_tokens'):
            self.momentum_encoder.summary_tokens.data.copy_(self.online_encoder.summary_tokens.data)
        
        for param_momentum in self.momentum_encoder.parameters():
            param_momentum.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        # Update momentum_encoder.transformer_encoder
        for param_online, param_momentum in zip(self.online_encoder.transformer_encoder.parameters(), 
                                                self.momentum_encoder.transformer_encoder.parameters()):
            param_momentum.data = param_momentum.data * (1.0 - self.momentum_tau) + param_online.data * self.momentum_tau
        # Update momentum_encoder.summary_tokens if they exist
        if self.conf_num_summary_tokens > 0 and hasattr(self.online_encoder, 'summary_tokens') and hasattr(self.momentum_encoder, 'summary_tokens'):
            self.momentum_encoder.summary_tokens.data = \
                self.momentum_encoder.summary_tokens.data * (1.0 - self.momentum_tau) + \
                self.online_encoder.summary_tokens.data * self.momentum_tau

    def forward_pretrain(self, x_segment_patches_raw, patch_mask_flat_for_m_bar):
        # x_segment_patches_raw: (B, C, N_time_patches, patch_time_len_samples)
        # patch_mask_flat_for_m_bar: (B, TotalPatches), True means masked (part of M_bar)
        B, C, N_t, P_t = x_segment_patches_raw.shape
        device = x_segment_patches_raw.device

        x_patches_flat_for_embed = rearrange(x_segment_patches_raw, 'b c nt pt -> b (c nt) pt')
        channel_ids_per_patch = torch.arange(C, device=device).repeat_interleave(N_t)
        channel_ids_flat = channel_ids_per_patch.unsqueeze(0).expand(B, -1)
        
        all_patch_tokens_embedded = self.embedding_layer(x_patches_flat_for_embed, channel_ids_flat) 
        # Shape: (B, TotalPatches=304, EmbedDim)

        # Prepare input for Online Encoder:
        # Replace M_bar patch embeddings with the learnable MASK token embedding
        input_patches_for_online_encoder = torch.where(
            patch_mask_flat_for_m_bar.unsqueeze(-1), 
            self.mask_token_embed.expand(B, self.total_patches_per_segment, -1), # MASK where True
            all_patch_tokens_embedded # Original embedding where False (visible M part)
        )
        
        # Prepend summary tokens (if any) to the sequence for the online encoder
        current_input_to_online_enc = input_patches_for_online_encoder
        if self.conf_num_summary_tokens > 0 and hasattr(self.online_encoder, 'summary_tokens'):
            summary_tokens_batch = self.online_encoder.summary_tokens.expand(B, -1, -1)
            current_input_to_online_enc = torch.cat((summary_tokens_batch, input_patches_for_online_encoder), dim=1)
        # print(f"[DEBUG] Shape of current_input_to_online_enc: {current_input_to_online_enc.shape}") # Expected (B, 305, D)

        # Online Encoder processes this sequence
        encoded_sequence_w_summary = self.online_encoder(current_input_to_online_enc)
        # print(f"[DEBUG] Shape of encoded_sequence_w_summary: {encoded_sequence_w_summary.shape}") # Expected (B, 305, D)

        # Extract features corresponding to patch positions (remove summary token features)
        encoded_patch_features = encoded_sequence_w_summary
        if self.conf_num_summary_tokens > 0:
            encoded_patch_features = encoded_sequence_w_summary[:, self.conf_num_summary_tokens:, :]
        # print(f"[DEBUG] Shape of encoded_patch_features (input to predictor): {encoded_patch_features.shape}") # Expected (B, 304, D)

        # Online Predictor: input is features of ALL patch positions from encoder
        # (some were actual MASK embeddings, others were visible patch embeddings)
        # Predictor aims to output refined features, especially for the MASKED (M_bar) positions
        predicted_features_all_patches = self.online_predictor(encoded_patch_features)
        # print(f"[DEBUG] Shape of predicted_features_all_patches: {predicted_features_all_patches.shape}") # Expected (B, 304, D)
        # print(f"[DEBUG] Shape of patch_mask_flat_for_m_bar: {patch_mask_flat_for_m_bar.shape}") # Expected (B, 304)

        # Select predicted features corresponding to the MASKED patches (M_bar) for Alignment Loss (L_A)
        pred_align_feats = predicted_features_all_patches[patch_mask_flat_for_m_bar]
        # Shape: (Num_Truly_Masked_In_Batch_Across_All_Samples, EmbedDim)

        # Momentum Encoder Branch (for L_A targets)
        with torch.no_grad():
            self._update_momentum_encoder()
            # Momentum encoder sees all ORIGINAL UNMASKED embedded patch tokens + its summary tokens
            momentum_encoder_input_seq = all_patch_tokens_embedded
            if self.conf_num_summary_tokens > 0 and hasattr(self.momentum_encoder, 'summary_tokens'):
                summary_tokens_batch_momentum = self.momentum_encoder.summary_tokens.expand(B, -1, -1)
                momentum_encoder_input_seq = torch.cat((summary_tokens_batch_momentum, all_patch_tokens_embedded), dim=1)
            
            menc_encoded_sequence_w_summary = self.momentum_encoder(momentum_encoder_input_seq)
            
            menc_patch_features = menc_encoded_sequence_w_summary
            if self.conf_num_summary_tokens > 0:
                menc_patch_features = menc_encoded_sequence_w_summary[:, self.conf_num_summary_tokens:, :]
            
            # Target for L_A: features from momentum encoder for the MASKED patches (M_bar)
            target_align_feats = menc_patch_features[patch_mask_flat_for_m_bar].detach()

        # Online Reconstructor (for L_R targets)
        # Input: Features from online_encoder for UNMASKED parts (M),
        #        Features from online_predictor for MASKED parts (M_bar).
        # `encoded_patch_features` contains encoder's view (visible M + processed MASK for M_bar)
        # `predicted_features_all_patches` contains predictor's refined view for ALL positions
        # EEGPT paper states Reconstructor input is {enc_j | j in M} U {pred_j | j in M_bar}
        # So, where mask is True (M_bar), use predictor's output. Where False (M), use encoder's output.
        reconstructor_input_features = torch.where(
            patch_mask_flat_for_m_bar.unsqueeze(-1), # expand to (B, TotalPatches, 1)
            predicted_features_all_patches,      # Use predictor's features for M_bar (masked original)
            encoded_patch_features               # Use encoder's features for M (visible original)
        )
        
        reconstructed_token_embeddings = self.online_reconstructor(reconstructor_input_features)
        
        # Project MASKED reconstructed embeddings back to raw patch space for L_R
        reconstructed_embeddings_of_masked = reconstructed_token_embeddings[patch_mask_flat_for_m_bar]
        pred_recon_patches = self.reconstructor_to_raw_patch(reconstructed_embeddings_of_masked) # Use the separate linear layer

        # Target for L_R: original raw values of MASKED patches (M_bar)
        target_recon_patches = x_patches_flat_for_embed[patch_mask_flat_for_m_bar]

        return pred_align_feats, pred_recon_patches, \
               target_align_feats, target_recon_patches

    def extract_features(self, x_segment_patches_raw):
        B, C, N_t, P_t = x_segment_patches_raw.shape
        device = x_segment_patches_raw.device

        x_patches_flat_for_embed = rearrange(x_segment_patches_raw, 'b c nt pt -> b (c nt) pt')
        channel_ids_per_patch = torch.arange(C, device=device).repeat_interleave(N_t)
        channel_ids_flat = channel_ids_per_patch.unsqueeze(0).expand(B, -1)
        
        all_patch_tokens_embedded = self.embedding_layer(x_patches_flat_for_embed, channel_ids_flat)
        
        current_input_to_online_enc_extract = all_patch_tokens_embedded
        if self.conf_num_summary_tokens > 0 and hasattr(self.online_encoder, 'summary_tokens'):
            summary_tokens_batch = self.online_encoder.summary_tokens.expand(B, -1, -1)
            current_input_to_online_enc_extract = torch.cat((summary_tokens_batch, all_patch_tokens_embedded), dim=1)
        
        encoded_output_w_summary = self.online_encoder(current_input_to_online_enc_extract) 
        
        if self.conf_num_summary_tokens > 0:
            features_from_summary = encoded_output_w_summary[:, :self.conf_num_summary_tokens, :] 
            return features_from_summary.mean(dim=1) # (B, EmbedDim)
        else: 
            # If no summary tokens, average all patch features (alternative representation)
            return encoded_output_w_summary.mean(dim=1)