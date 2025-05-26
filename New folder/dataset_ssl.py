# NMT_EEGPT_Project/dataset_ssl.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
import logging
# import config_ssl_pretrain as cfg_ssl # For direct running, handle path or copy values

class NMT_SSL_Patched_Dataset(Dataset):
    def __init__(self, data_dir_ssl, # Path to PREPROCESSED_DATA_DIR_SSL
                 segment_duration_sec=4.0, target_sfreq=256,
                 patch_duration_ms=250,
                 n_channels=19, # Number of channels in the preprocessed .npy files
                 test_mode_reduce_data=False, n_segments_test_mode=100 # For quick tests
                 ):
        self.data_dir_ssl = data_dir_ssl
        self.segment_len_samples = int(target_sfreq * segment_duration_sec)
        self.patch_time_len_samples = int(patch_duration_ms / 1000 * target_sfreq)
        self.n_channels = n_channels
        
        if self.segment_len_samples % self.patch_time_len_samples != 0:
            raise ValueError("Segment length must be divisible by patch time length.")
        self.num_time_patches_per_channel = self.segment_len_samples // self.patch_time_len_samples
        self.total_patches_per_segment = self.n_channels * self.num_time_patches_per_channel

        logging.info(f"Initializing NMT_SSL_Patched_Dataset.")
        logging.info(f"  Segment length: {self.segment_len_samples} samples ({segment_duration_sec}s)")
        logging.info(f"  Patch time length: {self.patch_time_len_samples} samples ({patch_duration_ms}ms)")
        logging.info(f"  Num time patches per channel: {self.num_time_patches_per_channel}")
        logging.info(f"  Num channels: {self.n_channels}")
        logging.info(f"  Total patches per segment: {self.total_patches_per_segment}")

        self.segment_file_paths = sorted(glob(os.path.join(self.data_dir_ssl, '*.npy')))
        
        if test_mode_reduce_data:
            logging.info(f"Test mode: Reducing SSL dataset from {len(self.segment_file_paths)} total segments to {n_segments_test_mode}.")
            if len(self.segment_file_paths) > n_segments_test_mode:
                self.segment_file_paths = self.segment_file_paths[:n_segments_test_mode]
            else:
                logging.warning(f"Not enough segments ({len(self.segment_file_paths)}) for test_mode_reduce_data count ({n_segments_test_mode}). Using all available.")


        if not self.segment_file_paths:
            logging.error(f"No .npy segment files found in {self.data_dir_ssl}. Ensure dataset_utils.py has run and populated this directory with all normal/abnormal segments.")
            raise ValueError(f"No data found for SSL pretraining.")
        logging.info(f"Found {len(self.segment_file_paths)} total segments for SSL pretraining.")


    def __len__(self):
        return len(self.segment_file_paths)

    def __getitem__(self, idx):
        file_path = self.segment_file_paths[idx]
        
        # Load segment - shape (n_channels, segment_len_samples)
        segment_data = np.load(file_path).astype(np.float32)

        if segment_data.shape != (self.n_channels, self.segment_len_samples):
            # This should ideally not happen if dataset_utils.py is correct
            logging.error(f"Segment {file_path} has unexpected shape {segment_data.shape}. Expected ({self.n_channels}, {self.segment_len_samples}).")
            # You might need to skip this sample or handle resizing/padding
            # For now, returning a dummy tensor to avoid crashing, but this needs fixing.
            # It's better to ensure data integrity beforehand.
            # If this occurs, it indicates an issue in `dataset_utils.py` or the saved .npy files.
            # A robust solution might involve trying to reshape/pad if minor, or skipping if major.
            # For now, let's assume the shapes are correct.
            # return torch.zeros((self.n_channels, self.num_time_patches_per_channel, self.patch_time_len_samples)), \
            #        torch.zeros((self.n_channels, self.num_time_patches_per_channel), dtype=torch.long)
            raise ValueError(f"Data integrity issue with {file_path}")


        # 1. Patching the segment (EEGPT Section 2.3)
        # Input segment: (C, T_segment)
        # Output patches: (C, N_time_patches, patch_time_len_samples)
        patches = segment_data.reshape(self.n_channels,
                                       self.num_time_patches_per_channel,
                                       self.patch_time_len_samples)
        
        # For EEGPT's local spatio-temporal embedding, we need to associate each patch p_i,j
        # with its channel index 'i' and time patch index 'j'.
        # The model's embedding layer will handle combining patch content with channel and position embeddings.
        # Here, we just provide the raw patch data. The pretraining loop or model will handle masking.

        # We also need the original indices if masking shuffles patches for the Transformer input.
        # For now, let's return the structured patches. The model's forward pass
        # will take these and apply embeddings (content, channel, position).
        
        # The EEGPT model flattens these C * N_time_patches into a sequence for the Transformer.
        # For example, output could be:
        #   - flat_patches: (C * N_time_patches, patch_time_len_samples)
        #   - channel_ids: (C * N_time_patches,) indicating original channel for each flat patch
        #   - time_patch_ids: (C * N_time_patches,) indicating original time_patch_index for each flat patch

        return torch.from_numpy(patches) # (C, N_time_patches, patch_time_len_samples)


if __name__ == '__main__':
    # Example Usage (ensure config_ssl_pretrain.py values are accessible)
    logging.basicConfig(level=logging.INFO)
    
    _PREPROCESSED_SSL_DATA_DIR = 'data/processed_nmt_ssl/'
    _SEGMENT_DURATION_SEC_SSL = 4.0
    _TARGET_SFREQ_SSL = 256
    _PATCH_DURATION_MS_SSL = 250
    _N_CHANNELS_SSL = 19 # Must match what dataset_utils saves
    _TEST_MODE_REDUCE_SSL = True
    _N_SEGMENTS_TEST_MODE_SSL = 50

    if not os.path.exists(_PREPROCESSED_SSL_DATA_DIR) or \
       len(os.listdir(_PREPROCESSED_SSL_DATA_DIR)) == 0 :
        print(f"Processed SSL data not found or empty in {_PREPROCESSED_SSL_DATA_DIR}. Please run dataset_utils.py first.")
    else:
        print("Attempting to load SSL Patched dataset...")
        try:
            ssl_dataset = NMT_SSL_Patched_Dataset(
                data_dir_ssl=_PREPROCESSED_SSL_DATA_DIR,
                segment_duration_sec=_SEGMENT_DURATION_SEC_SSL,
                target_sfreq=_TARGET_SFREQ_SSL,
                patch_duration_ms=_PATCH_DURATION_MS_SSL,
                n_channels=_N_CHANNELS_SSL,
                test_mode_reduce_data=_TEST_MODE_REDUCE_SSL,
                n_segments_test_mode=_N_SEGMENTS_TEST_MODE_SSL
            )
            if len(ssl_dataset) > 0:
                print(f"SSL Patched Dataset loaded. Number of 4s segments: {len(ssl_dataset)}")
                sample_patches = ssl_dataset[0]
                # sample_patches should be (C, N_time_patches, patch_time_len_samples)
                print(f"Sample patches shape: {sample_patches.shape}")
                # Expected: (19, 16, 64) for 19ch, 4s segment, 256Hz, 250ms patch
            else:
                print("SSL Patched Dataset loaded, but is empty.")
        except ValueError as e:
            print(f"Error initializing NMT_SSL_Patched_Dataset: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")