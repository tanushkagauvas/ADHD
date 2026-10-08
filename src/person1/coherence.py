"""
Step 1.3: Spectral Coherence (COH) Calculation (Person 1 Script)

Implements Equation (8) from the paper:
    COH_kl(f) = | S_kl(f) |^2 / ( S_kk(f) * S_ll(f) )

Where:
- k, l: EEG channels (1 to 19)
- S_kl(f): Cross-spectral density between channel k and channel l at frequency f
- S_kk(f), S_ll(f): Self-spectral densities of channel k and channel l
- Averaged across all frequencies to obtain a 19x19 COH matrix
- Diagonal is set to zero (no self-correlation)
- Values range from [0, 1]

Input:
- data/processed/clean_epochs.npy  -> Shape: (N, 19, 640), float32
- data/processed/labels.npy        -> Shape: (N,), int64

Output:
- data/processed/coh_matrices.npy  -> Shape: (N, 19, 19), float32
"""

import os
import time
import numpy as np
from scipy.signal import windows

def compute_coherence_batch(epochs_batch, fs=128, nperseg=128, noverlap=64):
    """
    Vectorized computation of spectral coherence for a batch of epochs.
    
    Parameters:
        epochs_batch (np.ndarray): Shape (B, 19, 640)
        fs (int): Sampling frequency (128 Hz)
        nperseg (int): Segment length for Welch PSD (128 samples = 1.0 second)
        noverlap (int): Overlap between segments (64 samples = 50% overlap)
        
    Returns:
        coh_matrices (np.ndarray): Shape (B, 19, 19), symmetric, zero diagonal.
    """
    B, n_channels, n_times = epochs_batch.shape
    step = nperseg - noverlap
    n_seg = (n_times - nperseg) // step + 1
    
    # 1. Hann window
    win = windows.hann(nperseg).astype(np.float32)  # Shape: (128,)
    
    # 2. Extract overlapping segments: shape (n_seg, B, 19, nperseg)
    seg_list = []
    for s in range(n_seg):
        start = s * step
        end = start + nperseg
        seg = epochs_batch[:, :, start:end] * win[None, None, :]
        seg_list.append(seg)
    segments = np.array(seg_list, dtype=np.float32)  # (n_seg, B, 19, nperseg)
    
    # 3. FFT along time axis: (n_seg, B, 19, n_freq)
    X = np.fft.rfft(segments, axis=-1)
    
    # Transpose to (B, n_freq, n_seg, 19)
    # Axes: 0:n_seg, 1:B, 2:19, 3:n_freq -> transpose to (1, 3, 0, 2)
    X_t = np.transpose(X, (1, 3, 0, 2))  # (B, n_freq, n_seg, 19)
    
    # 4. Cross-spectral density: S_kl(f) = (1/n_seg) * sum_s X_k(f, s) * conj(X_l(f, s))
    # Einsum over segment axis (axis 2) -> (B, n_freq, 19, 19)
    S = np.einsum('bfsk,bfsl->bfkl', X_t, np.conj(X_t)) / float(n_seg)
    
    # 5. Numerator: | S_kl(f) |^2
    num = np.abs(S) ** 2  # (B, n_freq, 19, 19)
    
    # 6. Denominator: S_kk(f) * S_ll(f)
    # Extract real diagonal: shape (B, n_freq, 19)
    diag_S = np.real(np.diagonal(S, axis1=2, axis2=3))
    denom = diag_S[:, :, :, None] * diag_S[:, :, None, :]  # (B, n_freq, 19, 19)
    
    # 7. Coherence per frequency (Equation 8)
    coh_f = num / (denom + 1e-12)  # (B, n_freq, 19, 19)
    
    # Clip numerical precision to [0, 1]
    coh_f = np.clip(coh_f, 0.0, 1.0)
    
    # 8. Average across all frequencies
    coh = np.mean(coh_f, axis=1)  # (B, 19, 19)
    
    # Set diagonals to 0.0
    for b in range(B):
        np.fill_diagonal(coh[b], 0.0)
        
    return coh.astype(np.float32)

def compute_coherence_dataset(
    epochs_path="data/processed/clean_epochs.npy",
    labels_path="data/processed/labels.npy",
    output_path="data/processed/coh_matrices.npy"
):
    if not os.path.exists(epochs_path):
        raise FileNotFoundError(f"Missing {epochs_path}. Run Step 1.1 first!")

    print(f"Loading preprocessed epochs from: {epochs_path} ...")
    epochs = np.load(epochs_path)   # Shape: (N, 19, 640)
    labels = np.load(labels_path)   # Shape: (N,)
    n_epochs, n_channels, n_times = epochs.shape
    
    print(f"Loaded {n_epochs} segments ({n_channels} channels, {n_times} time points).")
    print(f"Calculating Spectral Coherence (COH) for all {n_epochs} segments...")
    
    start_time = time.time()
    coh_matrices = np.zeros((n_epochs, n_channels, n_channels), dtype=np.float32)
    
    batch_size = 250
    for i in range(0, n_epochs, batch_size):
        end = min(i + batch_size, n_epochs)
        batch = epochs[i:end]
        
        coh_batch = compute_coherence_batch(batch, fs=128, nperseg=128, noverlap=64)
        coh_matrices[i:end] = coh_batch
        
        elapsed = time.time() - start_time
        print(f"Processed [{end}/{n_epochs}] segments ({end/n_epochs*100:.1f}%) in {elapsed:.2f}s")
        
    total_time = time.time() - start_time
    print(f"\nSpectral Coherence computation completed in {total_time:.2f} seconds!")
    print(f"Output shape: {coh_matrices.shape}, Dtype: {coh_matrices.dtype}")
    print(f"COH Value Range: Min = {coh_matrices.min():.4f}, Max = {coh_matrices.max():.4f}, Mean = {coh_matrices.mean():.4f}")
    
    # Statistical validation preview: ADHD vs Control group mean coherence
    adhd_mean_coh = coh_matrices[labels == 1].mean()
    ctrl_mean_coh = coh_matrices[labels == 0].mean()
    print(f"Mean COH for ADHD group:    {adhd_mean_coh:.4f}")
    print(f"Mean COH for Control group: {ctrl_mean_coh:.4f}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, coh_matrices)
    print(f"Saved successfully to: {output_path}")

if __name__ == "__main__":
    compute_coherence_dataset()
