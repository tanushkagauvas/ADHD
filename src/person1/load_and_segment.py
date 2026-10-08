"""
Step 1: Load and Segment adhdata.csv (Person 1 Script)

Reads the raw CSV file, groups by individual child ID,
and slices the 19-channel EEG signals into non-overlapping 5-second segments
(at 128 Hz, 5 seconds = 640 sample points per segment).

Saves:
- data/processed/clean_epochs.npy  -> Shape: (N, 19, 640), float32
- data/processed/labels.npy        -> Shape: (N,), int64 (0: Control, 1: ADHD)
- data/processed/subject_ids.npy   -> Shape: (N,), string
"""

import os
import pandas as pd
import numpy as np

# 19 Standard 10-20 EEG Electrodes from the paper
CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 
    'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 
    'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]

WINDOW_SIZE = 640  # 5.0 seconds * 128 Hz = 640 points

def process_csv_dataset(csv_path="data/raw/adhdata.csv", output_dir="data/processed"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find dataset at {csv_path}. Please check path.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading dataset from: {csv_path} ...")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")
    print(f"Total unique subjects: {df['ID'].nunique()}")
    
    epochs_list = []
    labels_list = []
    subjects_list = []

    # Group by each subject so segments do not cross between different children
    for sub_id, group in df.groupby('ID'):
        # Extract signal data for the 19 channels
        signal = group[CHANNELS].values.astype(np.float32)  # Shape: (total_samples, 19)
        
        # Label: 1 for ADHD, 0 for Control
        raw_class = str(group['Class'].iloc[0]).strip()
        label = 1 if raw_class.upper() == 'ADHD' else 0
        
        n_samples = len(signal)
        n_windows = n_samples // WINDOW_SIZE
        
        for w in range(n_windows):
            start_idx = w * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            
            # Segment shape: (WINDOW_SIZE, 19) -> transpose to (19 channels, 640 time points)
            segment = signal[start_idx:end_idx, :].T
            
            epochs_list.append(segment)
            labels_list.append(label)
            subjects_list.append(sub_id)

    # Convert to NumPy arrays
    clean_epochs = np.array(epochs_list, dtype=np.float32)      # Shape: (N, 19, 640)
    labels = np.array(labels_list, dtype=np.int64)              # Shape: (N,)
    subject_ids = np.array(subjects_list, dtype=str)            # Shape: (N,)

    # Save to disk
    epochs_out = os.path.join(output_dir, "clean_epochs.npy")
    labels_out = os.path.join(output_dir, "labels.npy")
    sub_out = os.path.join(output_dir, "subject_ids.npy")

    np.save(epochs_out, clean_epochs)
    np.save(labels_out, labels)
    np.save(sub_out, subject_ids)

    print("\nProcessing complete!")
    print(f"Saved {len(clean_epochs)} 5-second segments:")
    print(f" - clean_epochs.npy: {clean_epochs.shape} (float32)")
    print(f" - labels.npy:       {labels.shape} (ADHD count: {np.sum(labels == 1)}, Control count: {np.sum(labels == 0)})")
    print(f" - subject_ids.npy:  {subject_ids.shape}")
    print("\nPerson 1 can now run connectivity calculation (PLI + Coherence) on clean_epochs.npy!")

if __name__ == "__main__":
    process_csv_dataset()
