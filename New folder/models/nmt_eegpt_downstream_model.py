# NMT_EEGPT_Project/models/nmt_eegpt_downstream_model.py
import torch
import torch.nn as nn
import logging # Added for logging within this file if needed
# Corrected import name:
from .nmt_eegpt_pretrain_model import NMT_EEGPT_Pretrain # To load encoder part or full model

class NMT_EEGPT_Classifier(nn.Module):
    def __init__(self, 
                 # Params to reconstruct the NMT_EEGPT_Pretrain model to get its encoder
                 n_channels_model, segment_time_len_samples, patch_time_len_samples,
                 embed_dim, encoder_layers, # These are specifically for the online_encoder part
                 # predictor_layers, reconstructor_layers, # Not needed to define classifier structure, but part of Pretrain_NMT_EEGPT
                 num_heads, ff_dim, dropout_transformer, 
                 num_summary_tokens,
                 # Classifier head params
                 n_classes,
                 # Adaptive Spatial Filter params
                 use_adaptive_spatial_filter=True, 
                 n_input_channels_to_asf=21, # User's original number of channels
                 # Path to load pretrained weights for the *entire* Pretrain_NMT_EEGPT model
                 pretrained_model_path=None, # Changed from pretrained_encoder_path for clarity
                 freeze_encoder=True 
                ):
        super().__init__()

        # Instantiate the full Pretrain_NMT_EEGPT model to easily access its components
        # and load its full state dict, from which we'll use the online_encoder.
        # When loading, all params for Pretrain_NMT_EEGPT must be provided, even if some parts
        # like predictor/reconstructor are not directly used in this classifier's forward pass.
        self.feature_extractor_base = NMT_EEGPT_Pretrain( # Use the correct class name
            n_channels_model=n_channels_model, 
            segment_time_len_samples=segment_time_len_samples,
            patch_time_len_samples=patch_time_len_samples,
            embed_dim=embed_dim,
            encoder_layers=encoder_layers, 
            # Provide dummy or actual values for these if Pretrain_NMT_EEGPT requires them
            # The actual values should match how the pretrained model was saved.
            # Using values that would match config_ssl_pretrain for these:
            predictor_layers=getattr(cfg_ssl_model_structure, 'PREDICTOR_LAYERS', 2), # Example
            reconstructor_layers=getattr(cfg_ssl_model_structure, 'RECONSTRUCTOR_LAYERS', 2), # Example
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            dropout_transformer=dropout_transformer,
            num_summary_tokens=num_summary_tokens,
            momentum_tau=getattr(cfg_ssl_model_structure, 'MOMENTUM_TAU', 0.01) # Example
        )

        if pretrained_model_path:
            try:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                # The checkpoint from pretrain_nmt_eegpt.py saves 'model_state_dict'
                # which is the state_dict of the Pretrain_NMT_EEGPT instance.
                
                # Filter unexpected keys or handle mismatches if any
                model_state_dict = self.feature_extractor_base.state_dict()
                # This filters keys from checkpoint that are not in the current model or have different shapes
                filtered_checkpoint_state = {
                    k: v for k, v in checkpoint['model_state_dict'].items()
                    if k in model_state_dict and v.shape == model_state_dict[k].shape
                }
                missing_keys, unexpected_keys = self.feature_extractor_base.load_state_dict(filtered_checkpoint_state, strict=False)
                
                if missing_keys:
                    logging.warning(f"Loaded pretrained NMT-EEGPT with MISSING keys: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Loaded pretrained NMT-EEGPT with UNEXPECTED keys in checkpoint: {unexpected_keys}")
                if not missing_keys and not unexpected_keys:
                    logging.info(f"Successfully loaded ALL pretrained NMT-EEGPT weights from {pretrained_model_path}")
                else:
                    logging.info(f"Partially loaded pretrained NMT-EEGPT weights from {pretrained_model_path}")

            except Exception as e:
                logging.error(f"Error loading pretrained NMT-EEGPT weights from {pretrained_model_path}: {e}. Model's feature_extractor_base will be randomly initialized.", exc_info=True)
        else:
            logging.warning("No pretrained_model_path provided for NMT_EEGPT_Classifier. Feature extractor will be randomly initialized (not recommended for SSL approach).")

        # Freeze parameters of the feature_extractor_base if specified
        if freeze_encoder:
            for param in self.feature_extractor_base.parameters():
                param.requires_grad = False
            # Optionally, unfreeze specific parts like summary tokens or the embedding layer
            # For strict linear probing, only the ASF and new classifier head are trained.
            # Example: Unfreeze summary tokens of the *online_encoder* part for tuning
            if hasattr(self.feature_extractor_base, 'online_encoder') and \
               hasattr(self.feature_extractor_base.online_encoder, 'summary_tokens'):
                self.feature_extractor_base.online_encoder.summary_tokens.requires_grad = True
                logging.info("Unfrozen summary tokens of the online_encoder for downstream task.")
            # Also unfreeze the embedding layer potentially
            # self.feature_extractor_base.embedding_layer.requires_grad_(True)


        self.use_asf = use_adaptive_spatial_filter
        self.n_input_channels_to_asf = n_input_channels_to_asf
        self.n_model_channels = n_channels_model # Channels the pretrained encoder expects

        if self.use_asf and self.n_input_channels_to_asf != self.n_model_channels:
            self.adaptive_spatial_filter = nn.Conv1d(
                self.n_input_channels_to_asf, 
                self.n_model_channels, 
                kernel_size=1, 
                bias=False 
            )
            logging.info(f"Using Adaptive Spatial Filter: {self.n_input_channels_to_asf} -> {self.n_model_channels} channels.")
            # ASF parameters will be trainable by default
        elif self.use_asf and self.n_input_channels_to_asf == self.n_model_channels:
            self.adaptive_spatial_filter = nn.Identity() # Or a learnable 1x1 conv for scaling
            logging.info(f"ASF: Input channels ({self.n_input_channels_to_asf}) match model channels ({self.n_model_channels}). Using Identity or learnable scaling if ASF enabled.")
        else: # Not using ASF
            if self.n_input_channels_to_asf != self.n_model_channels:
                logging.error(f"Channel mismatch: Input data has {self.n_input_channels_to_asf} channels, but model's feature extractor expects {self.n_model_channels} and ASF is disabled.")
                raise ValueError("Channel count mismatch without ASF to resolve it.")
            self.adaptive_spatial_filter = nn.Identity()
            

        # Classifier head
        self.classifier_head = nn.Linear(embed_dim, n_classes) # Assumes .extract_features returns (B, embed_dim)

    def forward(self, x):
        # Input x: (B, n_input_channels_to_asf, segment_time_len_samples)
        
        if self.use_asf:
            x = self.adaptive_spatial_filter(x) # Output: (B, n_model_channels, segment_time_len_samples)
        
        # x must now be (B, C_model, T_segment)
        # Convert to patches: (B, C_model, N_time_patches, patch_time_len_samples)
        B, C_model, T_seg = x.shape
        P_t_model = self.feature_extractor_base.patch_time_len_samples
        N_t_model = T_seg // P_t_model
        
        if T_seg % P_t_model != 0:
            raise ValueError(f"Segment length {T_seg} not divisible by model's patch time length {P_t_model}")
            
        x_segment_patches_for_extractor = x.reshape(B, C_model, N_t_model, P_t_model)
        
        # Extract features using the feature_extractor_base's .extract_features method
        # This method is defined in NMT_EEGPT_Project/models/nmt_eegpt_pretrain_model.py
        if self.training and not self.feature_extractor_base.online_encoder.summary_tokens.requires_grad:
            # If linear probing, feature extractor should be in eval mode if it has dropout/BN
            self.feature_extractor_base.eval() 
            with torch.no_grad(): # Ensure no grads for frozen part
                 features = self.feature_extractor_base.extract_features(x_segment_patches_for_extractor)
            self.feature_extractor_base.train(mode=self.training) # Restore original mode
        else: # Full fine-tuning or if summary_tokens are tunable
            features = self.feature_extractor_base.extract_features(x_segment_patches_for_extractor) 
        
        logits = self.classifier_head(features) # (B, n_classes)
        return logits

if __name__ == '__main__':
    # Example Usage (requires config_ssl_pretrain and config_ssl_finetune to be importable)
    # For direct testing, define necessary params from those configs here.
    # This also assumes a dummy pretrained model checkpoint exists.
    
    # Mock cfg_ssl_model_structure attributes (replace with actual import if possible)
    class MockCfgSSLPretrain:
        N_CHANNELS_MODEL = 19
        INPUT_TIME_LENGTH_MODEL = 1024
        PATCH_TIME_LENGTH_SAMPLES = 64
        EMBED_DIM = 256
        ENCODER_LAYERS = 4
        PREDICTOR_LAYERS = 2 # Needed for Pretrain_NMT_EEGPT structure
        RECONSTRUCTOR_LAYERS = 2 # Needed for Pretrain_NMT_EEGPT structure
        NUM_HEADS = 4
        FEEDFORWARD_DIM = 256 * 2
        DROPOUT_PRETRAIN = 0.1
        NUM_SUMMARY_TOKENS = 1
        MOMENTUM_TAU = 0.01
    
    cfg_ssl_model_structure = MockCfgSSLPretrain()

    # Mock cfg_ft_xai attributes
    class MockCfgFT:
        N_CLASSES = 2
        USE_ADAPTIVE_SPATIAL_FILTER = True
        N_CHANNELS_INPUT_TO_MODEL = 21 # User's actual data channels
        # PRETRAINED_NMT_EEGPT_ENCODER_PATH = 'models/saved_ssl_pretrain/dummy_nmt_eegpt_full_model.pt' # Use this path
        PRETRAINED_NMT_EEGPT_ENCODER_PATH = None # Test without loading for init

    cfg_ft_xai = MockCfgFT()
    
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy Pretrain_NMT_EEGPT model and save its state_dict for testing loader
    if cfg_ft_xai.PRETRAINED_NMT_EEGPT_ENCODER_PATH: # Only if a path is defined for test
        dummy_pretrain_model_for_saving = NMT_EEGPT_Pretrain(
            n_channels_model=cfg_ssl_model_structure.N_CHANNELS_MODEL,
            segment_time_len_samples=cfg_ssl_model_structure.INPUT_TIME_LENGTH_MODEL,
            patch_time_len_samples=cfg_ssl_model_structure.PATCH_TIME_LENGTH_SAMPLES,
            embed_dim=cfg_ssl_model_structure.EMBED_DIM,
            encoder_layers=cfg_ssl_model_structure.ENCODER_LAYERS,
            predictor_layers=cfg_ssl_model_structure.PREDICTOR_LAYERS,
            reconstructor_layers=cfg_ssl_model_structure.RECONSTRUCTOR_LAYERS,
            num_heads=cfg_ssl_model_structure.NUM_HEADS,
            ff_dim=cfg_ssl_model_structure.FEEDFORWARD_DIM,
            dropout_transformer=cfg_ssl_model_structure.DROPOUT_PRETRAIN,
            num_summary_tokens=cfg_ssl_model_structure.NUM_SUMMARY_TOKENS,
            momentum_tau=cfg_ssl_model_structure.MOMENTUM_TAU
        )
        os.makedirs(os.path.dirname(cfg_ft_xai.PRETRAINED_NMT_EEGPT_ENCODER_PATH), exist_ok=True)
        torch.save({'model_state_dict': dummy_pretrain_model_for_saving.state_dict()}, cfg_ft_xai.PRETRAINED_NMT_EEGPT_ENCODER_PATH)
        logging.info(f"Saved dummy full pretrained model to {cfg_ft_xai.PRETRAINED_NMT_EEGPT_ENCODER_PATH}")


    # Instantiate the downstream classifier
    downstream_model = NMT_EEGPT_Classifier(
        n_channels_model=cfg_ssl_model_structure.N_CHANNELS_MODEL,
        segment_time_len_samples=cfg_ssl_model_structure.INPUT_TIME_LENGTH_MODEL,
        patch_time_len_samples=cfg_ssl_model_structure.PATCH_TIME_LENGTH_SAMPLES,
        embed_dim=cfg_ssl_model_structure.EMBED_DIM,
        encoder_layers=cfg_ssl_model_structure.ENCODER_LAYERS,
        num_heads=cfg_ssl_model_structure.NUM_HEADS,
        ff_dim=cfg_ssl_model_structure.FEEDFORWARD_DIM,
        dropout_transformer=cfg_ssl_model_structure.DROPOUT_PRETRAIN,
        num_summary_tokens=cfg_ssl_model_structure.NUM_SUMMARY_TOKENS,
        n_classes=cfg_ft_xai.N_CLASSES,
        use_adaptive_spatial_filter=cfg_ft_xai.USE_ADAPTIVE_SPATIAL_FILTER,
        n_input_channels_to_asf=cfg_ft_xai.N_CHANNELS_INPUT_TO_MODEL,
        pretrained_model_path=cfg_ft_xai.PRETRAINED_NMT_EEGPT_ENCODER_PATH,
        freeze_encoder=True 
    ).to(device)

    param_count_total = sum(p.numel() for p in downstream_model.parameters())
    param_count_trainable = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
    logging.info(f"NMT_EEGPT_Classifier initialized. Total Params: {param_count_total/1e6:.2f}M, Trainable Params: {param_count_trainable/1e3:.2f}K")

    # Test with dummy input (Batch, UserChannels=21, TimeSamples)
    dummy_input_downstream = torch.randn(4, cfg_ft_xai.N_CHANNELS_INPUT_TO_MODEL, cfg_ssl_model_structure.INPUT_TIME_LENGTH_MODEL).to(device)
    try:
        output = downstream_model(dummy_input_downstream)
        logging.info(f"Downstream model output shape: {output.shape}") # Expected: (4, 2)
    except Exception as e:
        logging.error(f"Error during downstream model test forward pass: {e}", exc_info=True)