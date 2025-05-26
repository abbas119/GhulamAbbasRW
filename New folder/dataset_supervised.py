# NMT_EEGPT_Project/dataset_supervised.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from glob import glob
import logging
import random

# Assuming config_supervised.py is in the parent directory or path is set
# For direct running, you might need to adjust path or copy relevant cfg vars here
# For simplicity, I'll assume cfg_supervised can be imported or values are known
# import config_supervised as cfg_sup # This might fail if run directly, handle path

class SupervisedNMTDataset(Dataset):
    def __init__(self, data_dir, split_type, label_map={'normal': 0, 'abnormal': 1},
                 segment_duration_sec=4.0, target_sfreq=256, augment=False,
                 test_mode_reduce_data=False, n_recordings_test_mode=10):
        """
        Args:
            data_dir (str): Path to the PROCESSED_DATA_SUPERVISED_DIR.
            split_type (str): 'train' or 'eval'.
            label_map (dict): Mapping from folder name to label.
            segment_duration_sec, target_sfreq: For calculating expected samples.
            augment (bool): Whether to apply augmentation (for training).
            test_mode_reduce_data (bool): If True, use fewer recordings.
            n_recordings_test_mode (int): Number of recordings per class for test mode.
        """
        self.data_dir = data_dir
        self.split_type = split_type
        self.label_map = label_map
        self.segment_len_samples = int(target_sfreq * segment_duration_sec)
        self.augment = augment
        self.file_paths = []
        self.labels = []

        logging.info(f"Initializing SupervisedNMTDataset for split: {split_type}, augment: {augment}")

        for label_name, label_idx in label_map.items():
            class_path = os.path.join(self.data_dir, self.split_type, label_name)
            if not os.path.isdir(class_path):
                logging.warning(f"Class path not found: {class_path}")
                continue
            
            npy_files = sorted(glob(os.path.join(class_path, '*.npy')))
            
            if test_mode_reduce_data:
                # In test_mode, we need to be careful. These are segments, not recordings.
                # This logic simplifies by taking first N segments if reducing.
                # A better way would be to select N original recordings in dataset_utils for test_mode.
                logging.info(f"Test mode: Reducing data for class '{label_name}' from {len(npy_files)} segments.")
                
                # This is a simplified way to limit data for quick tests.
                # It picks a small number of *segment files* per class for testing.
                # For true "n_recordings" test mode, filtering should happen at EDF processing stage.
                unique_recording_ids = sorted(list(set(["_".join(os.path.basename(f).split('_')[:3]) for f in npy_files]))) # e.g., normal_train_rec0000024
                if len(unique_recording_ids) > n_recordings_test_mode :
                     selected_ids = random.sample(unique_recording_ids, n_recordings_test_mode)
                else:
                     selected_ids = unique_recording_ids
                
                current_class_files = [f for f in npy_files if "_".join(os.path.basename(f).split('_')[:3]) in selected_ids]
                logging.info(f"Selected {len(current_class_files)} segments from {len(selected_ids)} recordings for class '{label_name}' in test_mode.")

            else:
                current_class_files = npy_files

            self.file_paths.extend(current_class_files)
            self.labels.extend([label_idx] * len(current_class_files))
            logging.info(f"Found {len(current_class_files)} segments for class '{label_name}' in {split_type} split.")

        if not self.file_paths:
            logging.error(f"No .npy segment files found for split '{split_type}' in {self.data_dir}. Ensure dataset_utils.py has run.")
            raise ValueError(f"No data found for {split_type} split.")

        logging.info(f"Total segments for {split_type}: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load segment - shape (n_channels, segment_len_samples)
        segment = np.load(file_path).astype(np.float32)

        if segment.shape[1] != self.segment_len_samples:
            logging.warning(f"Segment {file_path} has incorrect length {segment.shape[1]}, expected {self.segment_len_samples}. Skipping/Padding might be needed or check dataset_utils.")
            # Handle this case: either pad, truncate, or raise error
            # For now, let's assume correct length from dataset_utils.py

        if self.augment:
            # Example: Time shift augmentation (from your dataset.py)
            # Ensure segment is C, T
            max_shift_percent = 0.1 # Shift up to 10% of segment length
            max_shift_samples = int(max_shift_percent * segment.shape[1])
            shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
            if shift != 0:
                segment_aug = np.roll(segment, shift, axis=1)
                if shift > 0: # Shifted right, zero pad left
                    segment_aug[:, :shift] = 0.0
                elif shift < 0: # Shifted left, zero pad right
                    segment_aug[:, shift:] = 0.0
                segment = segment_aug
            
            # Example: Gaussian noise
            # noise_factor = 0.01
            # noise = np.random.normal(0, np.std(segment) * noise_factor, segment.shape)
            # segment = segment + noise
            # segment = segment.astype(np.float32)


        return torch.from_numpy(segment), torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    # Example Usage (ensure config_supervised.py values are accessible)
    # This assumes dataset_utils.py has been run and populated 'data/processed_nmt_supervised/'
    logging.basicConfig(level=logging.INFO) # For direct script run
    
    # Mock config values for direct testing if config_supervised.py is not directly importable
    _PROCESSED_DATA_SUPERVISED_DIR = 'data/processed_nmt_supervised/'
    _SEGMENT_DURATION_SEC = 4.0
    _TARGET_SFREQ = 256
    _TEST_MODE_REDUCE_DATA = True # Set to False for full dataset
    _N_RECORDINGS_TEST_MODE = 5 # Number of original recordings to draw segments from, per class

    if not os.path.exists(_PROCESSED_DATA_SUPERVISED_DIR) or \
       not os.path.exists(os.path.join(_PROCESSED_DATA_SUPERVISED_DIR, 'train', 'normal')) or \
       len(os.listdir(os.path.join(_PROCESSED_DATA_SUPERVISED_DIR, 'train', 'normal'))) == 0:
        print(f"Processed data not found or empty in {_PROCESSED_DATA_SUPERVISED_DIR}. Please run dataset_utils.py first.")
    else:
        print("Attempting to load supervised dataset...")
        try:
            train_dataset = SupervisedNMTDataset(
                data_dir=_PROCESSED_DATA_SUPERVISED_DIR,
                split_type='train',
                augment=True,
                segment_duration_sec=_SEGMENT_DURATION_SEC,
                target_sfreq=_TARGET_SFREQ,
                test_mode_reduce_data=_TEST_MODE_REDUCE_DATA,
                n_recordings_test_mode=_N_RECORDINGS_TEST_MODE
            )
            if len(train_dataset) > 0:
                print(f"Supervised Training dataset loaded. Number of segments: {len(train_dataset)}")
                data_sample, label_sample = train_dataset[0]
                print(f"Sample data shape: {data_sample.shape}, Label: {label_sample}")
            else:
                print("Supervised Training dataset loaded, but is empty.")

            eval_dataset = SupervisedNMTDataset(
                data_dir=_PROCESSED_DATA_SUPERVISED_DIR,
                split_type='eval',
                augment=False,
                segment_duration_sec=_SEGMENT_DURATION_SEC,
                target_sfreq=_TARGET_SFREQ,
                test_mode_reduce_data=_TEST_MODE_REDUCE_DATA,
                n_recordings_test_mode=_N_RECORDINGS_TEST_MODE
            )
            if len(eval_dataset) > 0:
                print(f"Supervised Evaluation dataset loaded. Number of segments: {len(eval_dataset)}")
                data_sample_eval, label_sample_eval = eval_dataset[0]
                print(f"Sample eval data shape: {data_sample_eval.shape}, Label: {label_sample_eval}")
            else:
                print("Supervised Evaluation dataset loaded, but is empty.")

        except ValueError as e:
            print(f"Error initializing SupervisedNMTDataset: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")