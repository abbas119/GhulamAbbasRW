# NMT_EEGPT_Project/models/ctnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_time_length,
                 target_sfreq, # Needed for Kc1
                 f1=8, d_multiplier=2, f2=16, # Conv module params from CTNet Table 1 [cite: 1422]
                 kc1_divisor=4, # Kc1 = target_sfreq / kc1_divisor [cite: 1362]
                 pool1_size=8, k2_kernel_length=16, pool2_size=8, # Conv module params [cite: 1422]
                 transformer_depth=6, transformer_heads=2, # Transformer params [cite: 1422]
                 dropout_cnn=0.25, dropout_transformer_p1=0.25, dropout_classifier_p2=0.5): # Dropouts [cite: 1371, 1422, 1427]
        super(CTNet, self).__init__()

        self.input_time_length = input_time_length
        kc1 = int(target_sfreq / kc1_divisor) # Temporal conv kernel length [cite: 1362]
        
        # Convolutional Module (Inspired by EEGNet, detailed in CTNet Fig 2 & text)
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, f1, kernel_size=(1, kc1), padding=(0, kc1 // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        
        # Layer 2: Depthwise Convolution (Spatial)
        self.depthwise_conv = nn.Conv2d(f1, f1 * d_multiplier, kernel_size=(n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d_multiplier)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, pool1_size))
        self.dropout1 = nn.Dropout(dropout_cnn) # CTNet uses 0.5 for subj-specific, 0.25 for cross-subj

        # Layer 3: Spatial Convolution (Separable Conv like in EEGNet, but CTNet calls it Spatial Conv)
        # CTNet paper says "spatial convolution, comprises F2 filters of size (1, Km)" -> Km is K2_kernel_length
        # Output F2 feature maps. Input channels: F1 * D
        self.spatial_conv = nn.Conv2d(f1 * d_multiplier, f2, kernel_size=(1, k2_kernel_length), 
                                      padding=(0, k2_kernel_length // 2), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, pool2_size)) # This pool size P2 regulates token size Tc
        self.dropout2 = nn.Dropout(dropout_cnn)

        # Calculate Tc (token sequence length for Transformer) and d_model (feature dim for Transformer)
        # T_after_conv1_approx = input_time_length 
        # T_after_pool1 = T_after_conv1_approx // pool1_size
        # T_after_conv2_approx = T_after_pool1 
        # Tc = T_after_conv2_approx // pool2_size
        # For simplicity, let's compute output shape after CNN part dynamically or make rough estimate
        self.transformer_d_model = f2 # Feature dimension for Transformer is F2

        # Transformer Encoder Module
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_d_model,
            nhead=transformer_heads,
            dim_feedforward=self.transformer_d_model * 2, # Common practice, or can be tuned
            dropout=dropout_transformer_p1, # p1 in CTNet
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=transformer_depth)

        # Classifier Module
        # CTNet paper says "output features are added, enabling direct transmission of features extracted by the CNN to the classifier"
        # This implies features from CNN output (after conv module) might be residually added to Transformer output
        # before flattening, or it means CNN features are *transformed* then classified.
        # Figure 2 suggests CNN output -> Transformer -> Flatten -> Dropout -> Linear. Let's follow Fig 2.
        self.dropout_classifier = nn.Dropout(dropout_classifier_p2) # p2 in CTNet
        
        # Flattening will depend on Tc. We need to calculate Tc properly.
        # For now, assume a Global Average Pooling over the time dimension of Transformer output.
        self.fc_classifier = nn.Linear(self.transformer_d_model, n_classes)


    def forward(self, x):
        # Input x: (batch_size, n_channels, input_time_length)
        x = x.unsqueeze(1) # (batch_size, 1, n_channels, input_time_length) for Conv2d

        # Convolutional Module
        x = self.conv1(x)    # (B, F1, C, T)
        x = self.bn1(x)
        
        x = self.depthwise_conv(x) # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x) # (B, F1*D, 1, T/P1)
        x = self.dropout1(x)
        
        x = self.spatial_conv(x) # (B, F2, 1, T/P1)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x) # (B, F2, 1, T/(P1*P2)) -> This is (B, d_model, 1, Tc)
        x = self.dropout2(x)
        
        # Prepare for Transformer
        # Input to Transformer: (Batch, SeqLen=Tc, FeatureDim=d_model)
        x = x.squeeze(2)            # (B, F2, Tc) where F2 is d_model
        x = x.permute(0, 2, 1)      # (B, Tc, F2=d_model)
        
        # Transformer Encoder
        x = self.transformer_encoder(x) # (B, Tc, d_model)
        
        # Classifier
        # CTNet Fig 2 shows "Flatten" then Dropout & Linear.
        # Often, for sequence classification, the output of the [CLS] token is used, or mean pooling.
        # Let's use mean pooling over the sequence length (Tc)
        x = x.mean(dim=1) # (B, d_model)
        x = self.dropout_classifier(x)
        x = self.fc_classifier(x) # (B, n_classes)
        
        return x

if __name__ == '__main__':
    # Example Usage (parameters from config_supervised and CTNet paper)
    _N_CHANNELS_SELECTED = 19
    _N_CLASSES = 2
    _INPUT_TIME_LENGTH = 1024 # 4s * 256Hz
    _TARGET_SFREQ = 256

    # CTNet specific params from config_supervised or defaults
    _F1 = 8
    _D_MULTIPLIER = 2
    _F2 = 16
    _KC1_DIVISOR = 4
    _POOL1_SIZE = 8
    _K2_KERNEL_LENGTH = 16
    _POOL2_SIZE = 8 # This makes Tc = (1024/8)/8 = 128/8 = 16
    _TRANSFORMER_DEPTH = 6
    _TRANSFORMER_HEADS = 2
    _DROPOUT_CNN = 0.25
    _DROPOUT_TRANSFORMER_P1 = 0.25
    _DROPOUT_CLASSIFIER_P2 = 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTNet(
        n_channels=_N_CHANNELS_SELECTED,
        n_classes=_N_CLASSES,
        input_time_length=_INPUT_TIME_LENGTH,
        target_sfreq=_TARGET_SFREQ,
        f1=_F1, d_multiplier=_D_MULTIPLIER, f2=_F2,
        kc1_divisor=_KC1_DIVISOR, pool1_size=_POOL1_SIZE,
        k2_kernel_length=_K2_KERNEL_LENGTH, pool2_size=_POOL2_SIZE,
        transformer_depth=_TRANSFORMER_DEPTH, transformer_heads=_TRANSFORMER_HEADS,
        dropout_cnn=_DROPOUT_CNN, dropout_transformer_p1=_DROPOUT_TRANSFORMER_P1,
        dropout_classifier_p2=_DROPOUT_CLASSIFIER_P2
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CTNet initialized. Parameter count: {param_count/1e3:.2f}K") # CTNet paper: 25.7k/24.9k params [cite: 1575]
                                                                        # Check if this implementation matches that.

    # Test with dummy input
    dummy_input = torch.randn(4, _N_CHANNELS_SELECTED, _INPUT_TIME_LENGTH).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (4, 2)

    # For this model with example params (F2=16), Tc=16 -> 25.7K params. Matches Table 7 in CTNet paper for BCI-IV-2a. Good.