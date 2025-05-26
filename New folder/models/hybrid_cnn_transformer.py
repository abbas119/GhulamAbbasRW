# NMT_EEGPT_Project/models/hybrid_cnn_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F # For functional operations if needed

class HybridCNNTransformer(nn.Module):
    def __init__(self, 
                 n_channels, # Number of input EEG channels (e.g., 19 or 21 from your config)
                 n_start_chans, # Number of output channels from the first CNN layer
                 n_layers_transformer, # Number of layers in the Transformer encoder
                 n_heads, # Number of attention heads in the Transformer
                 hidden_dim, # Dimension of the features fed into the Transformer (d_model)
                 ff_dim, # Feed-forward dimension in the Transformer
                 dropout, # Dropout rate
                 input_time_length, # Samples per segment (e.g., 4s * 256Hz = 1024)
                 n_classes=2):
        super(HybridCNNTransformer, self).__init__()
        
        self.n_channels = n_channels
        self.input_time_length = input_time_length # Keep for reference, though pooling makes it flexible
        self.hidden_dim = hidden_dim

        # CNN (inspired by your original model)
        # Input shape to conv1: (batch_size, 1, n_channels, input_time_length)
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=n_start_chans, 
            kernel_size=(self.n_channels, 3), # Kernel convolves across all channels spatially, and 3 time points
            padding=(0, 1) # Pad time dimension to maintain length
        )
        self.bn1 = nn.BatchNorm2d(n_start_chans)
        
        # Input shape to conv2: (batch_size, n_start_chans, 1, input_time_length)
        self.conv2 = nn.Conv2d(
            in_channels=n_start_chans, 
            out_channels=n_start_chans * 2, 
            kernel_size=(1, 3), # Kernel convolves only along time
            padding=(0, 1)
        )
        self.bn2 = nn.BatchNorm2d(n_start_chans * 2)
        
        # Adaptive Average Pooling to get fixed-size output for Transformer
        # Output shape after pool: (batch_size, n_start_chans * 2, 1, hidden_dim_cnn_out)
        # We want the last dimension to be `hidden_dim` for the transformer.
        # The CNN part should output (Batch, SeqLen_for_Transformer, FeatureDim_for_Transformer)
        # Your original pool: nn.AdaptiveAvgPool2d((1, hidden_dim))
        # This means after conv2 (B, n_start_chans*2, 1, T_after_convs),
        # it becomes (B, n_start_chans*2, 1, hidden_dim).
        # Then squeeze(2) gives (B, n_start_chans*2, hidden_dim).
        # Then permute(0,2,1) gives (B, hidden_dim, n_start_chans*2) for transformer.
        # This is unusual. Usually, hidden_dim is the feature dim.
        # Let's adjust to output (B, num_patches_for_transformer, hidden_dim_transformer)
        
        # Let's assume the CNN part should be a feature extractor creating a sequence of feature vectors.
        # The output of conv2 is (B, n_start_chans*2, 1, T_processed).
        # We want to transform this into (B, SeqLen, hidden_dim) for the Transformer.
        # One way: Treat each time point after CNN as a token, and n_start_chans*2 as feature_dim
        # self.cnn_output_features = n_start_chans * 2
        # self.projection_to_hidden_dim = nn.Linear(self.cnn_output_features, hidden_dim) # If feature dims don't match

        # Your original approach:
        self.pool = nn.AdaptiveAvgPool2d((1, self.hidden_dim)) # Output: (B, n_start_chans*2, 1, self.hidden_dim)
        self.cnn_to_transformer_projection_features = n_start_chans * 2 # This will be the feature dim after permute

        # If hidden_dim for Transformer is different from n_start_chans*2, a projection is needed.
        # In your current code, hidden_dim passed to Transformer is the *sequence length* after pooling.
        # And n_start_chans*2 becomes the *feature dimension* for the Transformer.
        # Let's stick to your original logic, but clarify variable names.
        # d_model for Transformer will be n_start_chans * 2
        # seq_len for Transformer will be hidden_dim (the argument)
        
        self.transformer_d_model = n_start_chans * 2
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_d_model, # Feature dimension from CNN
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True, # Expects (batch, seq_len, feature_dim)
            norm_first=True # Common in recent transformers for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_transformer)
        
        # Classifier
        # Input to classifier is based on transformer_d_model
        self.classifier = nn.Linear(self.transformer_d_model, n_classes)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x: (batch_size, n_channels, input_time_length)
        
        x = x.unsqueeze(1) # Add channel for Conv2d: (batch_size, 1, n_channels, input_time_length)
        
        # CNN Part
        x = self.conv1(x) # Output: (batch_size, n_start_chans, 1, input_time_length)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x) # Output: (batch_size, n_start_chans * 2, 1, input_time_length)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.pool(x) # Output: (batch_size, n_start_chans * 2, 1, self.hidden_dim (as seq_len))
        x = self.dropout_layer(x) # Dropout after pooling, before transformer
        
        x = x.squeeze(2) # Output: (batch_size, n_start_chans * 2, self.hidden_dim (as seq_len))
        
        # Permute for Transformer: (batch_size, seq_len, feature_dim)
        # Your original code had x.permute(0, 2, 1) -> (B, self.hidden_dim, n_start_chans*2)
        # This means self.hidden_dim is the sequence length, and n_start_chans*2 is d_model
        x = x.permute(0, 2, 1) # (Batch, self.hidden_dim (as SeqLen), n_start_chans*2 (as d_model))
                               # This interpretation matches the TransformerEncoderLayer d_model init
        
        # Transformer Part
        x = self.transformer_encoder(x) # Output: (Batch, self.hidden_dim (as SeqLen), n_start_chans*2 (as d_model))
        
        # Classifier Part
        # Use mean of sequence features before classifier (common practice)
        x = x.mean(dim=1) # Output: (Batch, n_start_chans*2 (as d_model))
        
        x = self.classifier(x) # Output: (Batch, n_classes)
        return x

if __name__ == '__main__':
    # Example Usage (parameters from a hypothetical config_supervised)
    _N_CHANNELS_SELECTED = 19
    _INPUT_TIME_LENGTH = 1024 # 4s * 256Hz
    _N_CLASSES = 2
    
    _HYBRID_N_START_CHANS = 32 # Reduced for quick test
    _HYBRID_N_LAYERS_TRANSFORMER = 1
    _HYBRID_N_HEADS = 2
    _HYBRID_HIDDEN_DIM_AS_SEQ_LEN = 64 # This is the sequence length input to transformer
    _HYBRID_FF_DIM = 128
    _HYBRID_DROPOUT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridCNNTransformer(
        n_channels=_N_CHANNELS_SELECTED,
        n_start_chans=_HYBRID_N_START_CHANS,
        n_layers_transformer=_HYBRID_N_LAYERS_TRANSFORMER,
        n_heads=_HYBRID_N_HEADS,
        hidden_dim=_HYBRID_HIDDEN_DIM_AS_SEQ_LEN, # This is used as Transformer's input sequence length
        ff_dim=_HYBRID_FF_DIM,
        dropout=_HYBRID_DROPOUT,
        input_time_length=_INPUT_TIME_LENGTH,
        n_classes=_N_CLASSES
    ).to(device)

    print(f"HybridCNNTransformer initialized. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # Test with dummy input
    # Batch size 4, 19 channels, 1024 time points
    dummy_input = torch.randn(4, _N_CHANNELS_SELECTED, _INPUT_TIME_LENGTH).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (4, 2)

    # Print model summary (requires torchinfo)
    # try:
    #     from torchinfo import summary
    #     summary(model, input_size=(4, _N_CHANNELS_SELECTED, _INPUT_TIME_LENGTH))
    # except ImportError:
    #     print("torchinfo not installed. Skipping model summary.")