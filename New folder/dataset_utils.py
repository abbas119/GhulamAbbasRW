# NMT_EEGPT_Project/dataset_utils.py
import mne
import numpy as np
import resampy
import os
from glob import glob
import logging
from tqdm import tqdm

# --- Configuration ---
BASE_NMT_DATA_PATH = 'D:/ValidData/Organized/'
PROCESSED_DATA_DIR_SUPERVISED = 'data/processed_nmt_supervised/'
PROCESSED_DATA_DIR_SSL = 'data/processed_nmt_ssl/'

TARGET_SFREQ = 256.0  # Use float for consistency
N_CHANNELS_SELECTED = 19
# This is your canonical 19-channel list. ALL selected data will be mapped to these names AND this order.
TARGET_CHANNELS_10_20_STANDARD_ORDER = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                                        'F7', 'F8', 'T7', 'P7', 'T8', 'P8', 'Fz', 'Cz', 'Pz']

# Channel name mapping: Maps names found in YOUR EDFs (UPPERCASE) to the standard names in TARGET_CHANNELS_10_20_STANDARD_ORDER
# YOU MUST VERIFY AND COMPLETE THIS MAPPING BASED ON YOUR ACTUAL EDF CHANNEL NAMES.
# Example: if your EDF has 'EEG FZ-REF', map it to 'Fz'. If it has 'T3', map it to 'T7'.
# Keys = UPPERCASE version of channel name in your EDF files
# Values = Correctly cased standard name from TARGET_CHANNELS_10_20_STANDARD_ORDER
CHANNEL_MAPPING_FROM_EDF_TO_STANDARD = {
    'FP1': 'FP1', 'EEG FP1': 'FP1', 'EEG FP1-REF': 'FP1', 'EEG FP1-LE': 'FP1',
    'FP2': 'FP2', 'EEG FP2': 'FP2', 'EEG FP2-REF': 'FP2', 'EEG FP2-LE': 'FP2',
    'F3': 'F3',   'EEG F3': 'F3',   'EEG F3-REF': 'F3',   'EEG F3-LE': 'F3',
    'F4': 'F4',   'EEG F4': 'F4',   'EEG F4-REF': 'F4',   'EEG F4-LE': 'F4',
    'C3': 'C3',   'EEG C3': 'C3',   'EEG C3-REF': 'C3',   'EEG C3-LE': 'C3',
    'C4': 'C4',   'EEG C4': 'C4',   'EEG C4-REF': 'C4',   'EEG C4-LE': 'C4',
    'P3': 'P3',   'EEG P3': 'P3',   'EEG P3-REF': 'P3',   'EEG P3-LE': 'P3',
    'P4': 'P4',   'EEG P4': 'P4',   'EEG P4-REF': 'P4',   'EEG P4-LE': 'P4',
    'O1': 'O1',   'EEG O1': 'O1',   'EEG O1-REF': 'O1',   'EEG O1-LE': 'O1',
    'O2': 'O2',   'EEG O2': 'O2',   'EEG O2-REF': 'O2',   'EEG O2-LE': 'O2',
    'F7': 'F7',   'EEG F7': 'F7',   'EEG F7-REF': 'F7',   'EEG F7-LE': 'F7',
    'F8': 'F8',   'EEG F8': 'F8',   'EEG F8-REF': 'F8',   'EEG F8-LE': 'F8',
    'T3': 'T7',   'EEG T3': 'T7',   'EEG T3-REF': 'T7',   'EEG T3-LE': 'T7', # Map T3 to T7
    'T4': 'T8',   'EEG T4': 'T8',   'EEG T4-REF': 'T8',   'EEG T4-LE': 'T8', # Map T4 to T8
    'T5': 'P7',   'EEG T5': 'P7',   'EEG T5-REF': 'P7',   'EEG T5-LE': 'P7', # Map T5 to P7
    'T6': 'P8',   'EEG T6': 'P8',   'EEG T6-REF': 'P8',   'EEG T6-LE': 'P8', # Map T6 to P8
    'FZ': 'Fz',   'EEG FZ': 'Fz',   'EEG FZ-REF': 'Fz',   'EEG FZ-LE': 'Fz', # Ensure FZ from EDF maps to Fz
    'CZ': 'Cz',   'EEG CZ': 'Cz',   'EEG CZ-REF': 'Cz',   'EEG CZ-LE': 'Cz',
    'PZ': 'Pz',   'EEG PZ': 'Pz',   'EEG PZ-REF': 'Pz',   'EEG PZ-LE': 'Pz',
    # Add any other variations seen in your EDFs that should map to the 19 standard channels.
    # If A1/A2 are ever read as data channels and need to be explicitly ignored, they won't map to one of the 19.
}

SEC_TO_CUT_RAW = 60.0
MAX_DURATION_TO_PROCESS_MINS = 15.0
MAX_ABS_VAL_CLIP = 800.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
SEGMENT_DURATION_SEC = 4.0
SAMPLES_PER_SEGMENT = int(TARGET_SFREQ * SEGMENT_DURATION_SEC)

LOG_FILE_UTILS = 'logs/dataset_utils.log'
os.makedirs('logs', exist_ok=True)

# Setup logger uniquely for this module
logger = logging.getLogger('dataset_utils_logger')
if not logger.handlers: # Avoid adding handlers multiple times
    logger.setLevel(logging.INFO)
    # File Handler
    fh = logging.FileHandler(LOG_FILE_UTILS, mode='w') # 'w' to overwrite log
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    # Console Handler
    ch_util = logging.StreamHandler()
    ch_util.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch_util)


def preprocess_and_segment_edf(edf_path, output_dir_base_supervised, output_dir_base_ssl, file_id_prefix):
    try:
        # Include 'stim_channel=None' if MNE warns about it, or handle specific STIM channels if they exist and are problematic
        raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose='WARNING')
        original_sfreq = float(raw.info['sfreq'])
        raw_ch_names_original_case = list(raw.ch_names)
        logger.debug(f"File: {edf_path}, Original channels ({len(raw_ch_names_original_case)}): {raw_ch_names_original_case}, SFreq: {original_sfreq}")

        # --- Channel Mapping and Selection ---
        rename_map_for_mne = {} # Store {current_name_in_raw: standard_name_to_map_to}
        
        # Step 1: Identify which of the raw channels can be mapped to our standard target channels
        mapped_raw_channels_to_standard_name = {} # {standard_name: original_name_in_raw_that_mapped_to_it}
        
        for raw_ch_name_in_file in raw_ch_names_original_case:
            raw_ch_name_upper = raw_ch_name_in_file.upper()
            
            standard_name_candidate = None
            if raw_ch_name_upper in CHANNEL_MAPPING_FROM_EDF_TO_STANDARD:
                standard_name_candidate = CHANNEL_MAPPING_FROM_EDF_TO_STANDARD[raw_ch_name_upper]
            elif raw_ch_name_upper in [std_ch.upper() for std_ch in TARGET_CHANNELS_10_20_STANDARD_ORDER]:
                 for std_ch_target_case in TARGET_CHANNELS_10_20_STANDARD_ORDER:
                     if std_ch_target_case.upper() == raw_ch_name_upper:
                         standard_name_candidate = std_ch_target_case
                         break
            
            if standard_name_candidate and standard_name_candidate in TARGET_CHANNELS_10_20_STANDARD_ORDER:
                if standard_name_candidate not in mapped_raw_channels_to_standard_name: # Keep first encountered mapping for a standard name
                    mapped_raw_channels_to_standard_name[standard_name_candidate] = raw_ch_name_in_file
                if raw_ch_name_in_file != standard_name_candidate: # If a rename is needed
                    rename_map_for_mne[raw_ch_name_in_file] = standard_name_candidate

        logger.debug(f"File: {edf_path}, Proposed MNE rename map: {rename_map_for_mne}")
        
        # Perform renaming if necessary
        if rename_map_for_mne:
            try:
                raw.rename_channels(rename_map_for_mne)
                logger.info(f"Renamed channels for {edf_path}. Current raw.ch_names after rename: {raw.ch_names}")
            except Exception as e:
                logger.warning(f"Could not rename some channels for {edf_path}: {e}. Current raw.ch_names: {raw.ch_names}")

        # Now, raw.ch_names are the (potentially renamed) standard names.
        # We need to pick exactly the TARGET_CHANNELS_10_20_STANDARD_ORDER if they exist.
        
        channels_present_for_picking = [ch for ch in TARGET_CHANNELS_10_20_STANDARD_ORDER if ch in raw.ch_names]
        
        if len(channels_present_for_picking) < N_CHANNELS_SELECTED:
            missing_channels = [ch for ch in TARGET_CHANNELS_10_20_STANDARD_ORDER if ch not in channels_present_for_picking]
            logger.warning(f"File: {edf_path}. After mapping and renaming, not all {N_CHANNELS_SELECTED} target channels are present. Missing: {missing_channels}. Available for picking: {channels_present_for_picking}. Skipping file.")
            return 0

        # Pick the standard channels in the desired order
        raw.pick_channels(TARGET_CHANNELS_10_20_STANDARD_ORDER, ordered=True) # Ensure this pick method is correct
        logger.debug(f"File: {edf_path}, Channels after pick_channels(ordered=True): {raw.ch_names}")

        # Double check final channel count (should be redundant if pick_channels worked)
        if len(raw.ch_names) != N_CHANNELS_SELECTED:
            logger.error(f"File: {edf_path}. Channel count is {len(raw.ch_names)} after pick, expected {N_CHANNELS_SELECTED}. Channels: {raw.ch_names}. Skipping.")
            return 0

        raw.set_eeg_reference('average', projection=False, verbose='WARNING')
        raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design='firwin', verbose='WARNING')

        tmin_crop = SEC_TO_CUT_RAW
        if raw.times[-1] < tmin_crop + SEGMENT_DURATION_SEC:
            logger.warning(f"File {edf_path} too short ({raw.times[-1]:.2f}s) for one segment after tmin_crop={tmin_crop:.2f}s. Skipping.")
            return 0
            
        tmax_crop = min(tmin_crop + (MAX_DURATION_TO_PROCESS_MINS * 60.0), raw.times[-1])
        
        if tmax_crop - tmin_crop < SEGMENT_DURATION_SEC:
            logger.warning(f"File {edf_path} not enough duration ({tmax_crop-tmin_crop:.2f}s) for one segment. Skipping.")
            return 0
        raw.crop(tmin=tmin_crop, tmax=tmax_crop, include_tmax=False)
        
        data = raw.get_data(units='uV')

        if original_sfreq != TARGET_SFREQ:
            data = resampy.resample(data, sr_orig=original_sfreq, sr_new=TARGET_SFREQ, axis=1, filter='kaiser_fast')

        data = np.clip(data, -MAX_ABS_VAL_CLIP, MAX_ABS_VAL_CLIP)
        
        mean_per_channel = np.mean(data, axis=1, keepdims=True)
        std_per_channel = np.std(data, axis=1, keepdims=True)
        data = (data - mean_per_channel) / (std_per_channel + 1e-8)

        num_segments_in_file = data.shape[1] // SAMPLES_PER_SEGMENT
        segments_saved_count = 0
        
        label_str_from_path = 'abnormal' if 'abnormal' in edf_path.lower().replace('\\',os.sep) else 'normal'
        
        for i in range(num_segments_in_file):
            segment = data[:, i * SAMPLES_PER_SEGMENT : (i + 1) * SAMPLES_PER_SEGMENT]
            if segment.shape[1] == SAMPLES_PER_SEGMENT:
                segment_id = f"{file_id_prefix}_seg{i:03d}"
                
                ssl_output_path = os.path.join(output_dir_base_ssl, f"{segment_id}_{label_str_from_path}.npy")
                os.makedirs(os.path.dirname(ssl_output_path), exist_ok=True)
                np.save(ssl_output_path, segment.astype(np.float32))
                
                original_split_type = os.path.basename(os.path.dirname(edf_path))
                if original_split_type not in ['train', 'eval']: # Handle deeper structures if any
                    path_parts = edf_path.split(os.sep)
                    if len(path_parts) >= 3: # e.g. D:/.../Organized/normal/train/file.edf
                        original_split_type = path_parts[-2] # Should be 'train' or 'eval'
                if original_split_type not in ['train', 'eval']: # Failsafe
                    logger.error(f"Could not determine train/eval split for {edf_path} (path part: {original_split_type}). Defaulting to 'train' for supervised save path.")
                    original_split_type = 'train'

                supervised_class_output_dir = os.path.join(output_dir_base_supervised, original_split_type, label_str_from_path)
                os.makedirs(supervised_class_output_dir, exist_ok=True)
                sup_output_path = os.path.join(supervised_class_output_dir, f"{segment_id}.npy")
                np.save(sup_output_path, segment.astype(np.float32))
                
                segments_saved_count += 1
        
        if segments_saved_count > 0:
            logger.info(f"Processed {edf_path}: Saved {segments_saved_count} segments. Final channel order (first 5): {raw.ch_names[:5]}...")
        else:
            logger.warning(f"Processed {edf_path}: No segments saved (check duration/channel issues).")
        return segments_saved_count

    except Exception as e:
        logger.error(f"Failed to process {edf_path}: {e}", exc_info=True)
        return 0

def run_preprocessing():
    logger.info("--- Starting NMT Dataset Preprocessing Utility (Revised Channel Handling v3) ---")
    os.makedirs(PROCESSED_DATA_DIR_SUPERVISED, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR_SSL, exist_ok=True)
    
    for split in ['train', 'eval']:
        for label_type in ['normal', 'abnormal']:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR_SUPERVISED, split, label_type), exist_ok=True)

    overall_file_counter = 0 
    total_segments_processed = 0

    for label_type_folder in ['normal', 'abnormal']:
        for split_folder in ['train', 'eval']:
            current_raw_path = os.path.join(BASE_NMT_DATA_PATH, label_type_folder, split_folder)
            if not os.path.isdir(current_raw_path):
                logger.warning(f"Directory not found, skipping: {current_raw_path}")
                continue

            edf_files = sorted(glob(os.path.join(current_raw_path, '*.edf')))
            if not edf_files:
                 logger.warning(f"No EDF files found in {current_raw_path}")
                 continue
            logger.info(f"Found {len(edf_files)} EDF files in {current_raw_path}")

            for edf_file_path in tqdm(edf_files, desc=f"Processing {label_type_folder}/{split_folder}"):
                unique_file_id_prefix = f"rec{overall_file_counter:05d}"
                
                num_saved_for_file = preprocess_and_segment_edf(
                    edf_file_path, 
                    PROCESSED_DATA_DIR_SUPERVISED, 
                    PROCESSED_DATA_DIR_SSL,
                    unique_file_id_prefix
                )
                overall_file_counter +=1 
                if num_saved_for_file > 0:
                    total_segments_processed += num_saved_for_file
                                    
    logger.info(f"--- NMT Dataset Preprocessing Complete ---")
    logger.info(f"Total EDF files attempted: {overall_file_counter}") 
    logger.info(f"Total 4-second segments saved: {total_segments_processed}")

if __name__ == '__main__':
    # Add a check for MNE version at the start of the script
    import mne
    print(f"MNE version being used by dataset_utils.py: {mne.__version__}")
    if mne.__version__ < '1.0': # Check if version is less than 1.0
        print("WARNING: MNE version is older than 1.0. The 'ordered=True' argument in pick_channels might not be supported.")
        print("Consider upgrading MNE or ensure your version handles channel picking and ordering correctly.")
        # Fallback for older MNE if necessary, but 1.6.0 should be fine.
    
    run_preprocessing()
    print(f"Preprocessing finished. Check logs at {LOG_FILE_UTILS}")