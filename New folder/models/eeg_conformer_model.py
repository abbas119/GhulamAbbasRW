# NMT_EEGPT_Project/models/eeg_conformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange # Not used in this version

# Helper for PatchEmbedding's CNN part (not used in this EEGConformer version, but good for other models)
# class ConvBlock(nn.Module): ...

class EEGConformer(nn.Module):
    def __init__(self, n_channels, n_classes, input_time_length,
                 target_sfreq,
                 # CNN (Patch Embedding) params
                 n_filters_time=40, filter_time_length_ms=100,
                 n_filters_spat=40,
                 pool_time_length_ms=300,
                 pool_time_stride_ms=60,
                 cnn_drop_prob=0.3,
                 # Transformer Encoder params
                 transformer_d_model_explicit=None, # Explicitly set d_model for Transformer if desired
                 transformer_depth=3, 
                 transformer_n_heads=4,
                 transformer_ff_dim_factor=2,
                 transformer_drop_prob=0.1,
                 # Classification head params
                 classifier_hidden_dim=128,
                 classifier_drop_prob=0.3
                 ):
        super(EEGConformer, self).__init__()

        filter_time_samples = int(filter_time_length_ms / 1000 * target_sfreq)
        pool_time_samples = int(pool_time_length_ms / 1000 * target_sfreq)
        pool_stride_samples = int(pool_time_stride_ms / 1000 * target_sfreq)

        # 1. Patch Embedding CNN
        self.temporal_conv = nn.Conv2d(1, n_filters_time, 
                                       kernel_size=(1, filter_time_samples),
                                       padding=(0, filter_time_samples // 2), bias=False)
        self.bn_temporal = nn.BatchNorm2d(n_filters_time)
        
        self.spatial_conv_depthwise = nn.Conv2d(n_filters_time, n_filters_time,
                                      kernel_size=(n_channels, 1), 
                                      groups=n_filters_time, bias=False)
        self.spatial_conv_pointwise = nn.Conv2d(n_filters_time, n_filters_spat,
                                                kernel_size=(1,1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(n_filters_spat)
        self.elu_spatial = nn.ELU()
        
        self.pooling = nn.AvgPool2d(kernel_size=(1, pool_time_samples), stride=(1, pool_stride_samples))
        self.dropout_cnn = nn.Dropout(cnn_drop_prob)

        # Calculate shape after CNN
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, n_channels, input_time_length) # Add batch and input conv channel
            cnn_out_shape = self._forward_cnn_for_shape_calc(dummy_x).shape
            # cnn_out_shape is (1, n_filters_spat, 1, num_patches_after_cnn_pooling)
        
        num_patches_for_transformer = cnn_out_shape[-1]
        self.cnn_output_feature_dim = n_filters_spat # Features from spatial conv

        # --- Corrected Logic for cnn_to_transformer_projection ---
        if transformer_d_model_explicit is None:
            self.transformer_d_model = self.cnn_output_feature_dim # Infer from CNN
            self.cnn_to_transformer_projection = nn.Identity()
        elif transformer_d_model_explicit != self.cnn_output_feature_dim:
            self.transformer_d_model = transformer_d_model_explicit
            self.cnn_to_transformer_projection = nn.Linear(self.cnn_output_feature_dim, self.transformer_d_model)
        else: # transformer_d_model_explicit == self.cnn_output_feature_dim
            self.transformer_d_model = transformer_d_model_explicit
            self.cnn_to_transformer_projection = nn.Identity()
        # --- End of Corrected Logic ---
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_d_model,
            nhead=transformer_n_heads,
            dim_feedforward=self.transformer_d_model * transformer_ff_dim_factor,
            dropout=transformer_drop_prob,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        # 3. Classification Head
        self.use_mean_pooling_for_classifier = True 
        if self.use_mean_pooling_for_classifier:
             self.classification_head = nn.Sequential(
                nn.Linear(self.transformer_d_model, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_drop_prob),
                nn.Linear(classifier_hidden_dim, n_classes)
            )
        else: # Flatten all patch outputs
            self.classification_head = nn.Sequential(
                nn.Linear(self.transformer_d_model * num_patches_for_transformer, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_drop_prob),
                nn.Linear(classifier_hidden_dim, n_classes)
            )

    def _forward_cnn_for_shape_calc(self, x): # x: (B, 1, C, T) - used only for shape calculation
        x = self.temporal_conv(x)
        x = self.bn_temporal(x)
        x = self.spatial_conv_depthwise(x)
        x = self.spatial_conv_pointwise(x)
        x = self.bn_spatial(x)
        x = self.elu_spatial(x)
        x = self.pooling(x)
        x = self.dropout_cnn(x)
        return x

    def forward(self, x):
        # Input x: (batch_size, n_channels, input_time_length)
        x = x.unsqueeze(1) # (B, 1, C, T) for Conv2d
        
        # CNN Patch Embedding part
        x = self.temporal_conv(x)       # (B, n_filters_time, C, T_processed_time)
        x = self.bn_temporal(x)         # No activation after BN before depthwise as per EEGNet
        
        x = self.spatial_conv_depthwise(x) # (B, n_filters_time, 1, T_processed_time)
        x = self.spatial_conv_pointwise(x) # (B, n_filters_spat, 1, T_processed_time)
        x = self.bn_spatial(x)
        x = self.elu_spatial(x)         # Activation after BN for spatial block
        
        x = self.pooling(x)             # (B, n_filters_spat, 1, num_patches)
        x = self.dropout_cnn(x)
        
        x = x.squeeze(2)                # (B, n_filters_spat, num_patches)
        x = x.permute(0, 2, 1)          # (B, num_patches, n_filters_spat) -> (Batch, SeqLen, FeatureDim_CNN)
        
        x = self.cnn_to_transformer_projection(x) # (B, num_patches, transformer_d_model)

        # Transformer Encoder
        x = self.transformer_encoder(x) # (B, num_patches, transformer_d_model)
        
        # Classification Head
        if self.use_mean_pooling_for_classifier:
            x = x.mean(dim=1) # (B, transformer_d_model)
        else:
            x = x.flatten(start_dim=1) # (B, num_patches * transformer_d_model)
            
        x = self.classification_head(x) # (B, n_classes)
        return x

if __name__ == '__main__':
    _N_CHANNELS_SELECTED = 19
    _N_CLASSES = 2
    _INPUT_TIME_LENGTH = 1024 # 4s * 256Hz
    _TARGET_SFREQ = 256

    _N_FILTERS_TIME = 40
    _FILTER_TIME_LENGTH_MS = 100 
    _N_FILTERS_SPAT = _N_FILTERS_TIME 
    _POOL_TIME_LENGTH_MS = 300 
    _POOL_TIME_STRIDE_MS = 60  
    _CNN_DROP_PROB = 0.3
    # _TRANSFORMER_D_MODEL_EXPLICIT = 64 # Example if you want to force a different d_model
    _TRANSFORMER_D_MODEL_EXPLICIT = None # Let it infer from CNN output (_N_FILTERS_SPAT)
    _TRANSFORMER_DEPTH = 3
    _TRANSFORMER_N_HEADS = 4
    _TRANSFORMER_FF_DIM_FACTOR = 2
    _TRANSFORMER_DROP_PROB = 0.1
    _CLASSIFIER_HIDDEN_DIM = 64 
    _CLASSIFIER_DROP_PROB = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EEGConformer(
        n_channels=_N_CHANNELS_SELECTED, n_classes=_N_CLASSES,
        input_time_length=_INPUT_TIME_LENGTH, target_sfreq=_TARGET_SFREQ,
        n_filters_time=_N_FILTERS_TIME, filter_time_length_ms=_FILTER_TIME_LENGTH_MS,
        n_filters_spat=_N_FILTERS_SPAT, pool_time_length_ms=_POOL_TIME_LENGTH_MS,
        pool_time_stride_ms=_POOL_TIME_STRIDE_MS, cnn_drop_prob=_CNN_DROP_PROB,
        transformer_d_model_explicit=_TRANSFORMER_D_MODEL_EXPLICIT, 
        transformer_depth=_TRANSFORMER_DEPTH, transformer_n_heads=_TRANSFORMER_N_HEADS,
        transformer_ff_dim_factor=_TRANSFORMER_FF_DIM_FACTOR,
        transformer_drop_prob=_TRANSFORMER_DROP_PROB,
        classifier_hidden_dim=_CLASSIFIER_HIDDEN_DIM,
        classifier_drop_prob=_CLASSIFIER_DROP_PROB
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGConformer initialized. Parameter count: {param_count/1e6:.2f}M")

    dummy_input = torch.randn(4, _N_CHANNELS_SELECTED, _INPUT_TIME_LENGTH).to(device)
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}") # Expected: (4, 2)
    except Exception as e:
        print(f"Error during model test: {e}")

    # For shape calculation in __init__
    # Need to pass (B,1,C,T) to _forward_cnn_for_shape_calc
    dummy_init_shape_calc = torch.zeros(1, 1, _N_CHANNELS_SELECTED, _INPUT_TIME_LENGTH)
    with torch.no_grad():
        cnn_out_test = model._forward_cnn_for_shape_calc(dummy_init_shape_calc)
    print(f"Internal CNN out shape for init: {cnn_out_test.shape}")
    # Expected e.g. (1, n_filters_spat, 1, num_patches)