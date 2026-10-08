"""
Step 1.2: Phase Lag Index (PLI) Calculation (Person 1 Script)

Implements Equation (7) from the paper:
    PLI_kl = | (1 / M) * sum_{m=1}^M sign(Delta phi_kl(t_m)) |

Where:
- k, l: EEG channels (1 to 19)
- M = 640: Number of time sample points in each 5-second segment (at 128 Hz)
- Delta phi_kl(t_m) = phi_k(t_m) - phi_l(t_m): Instantaneous phase difference
- Phase is obtained via the Hilbert transform (scipy.signal.hilbert)
- PLI values range from [0, 1], with diagonal set to 0.

Input:
- data/processed/clean_epochs.npy  -> Shape: (N, 19, 640), float32
- data/processed/labels.npy        -> Shape: (N,), int64

Output:
- data/processed/pli_matrices.npy   -> Shape: (N, 19, 19), float32
"""

import os
import time
import numpy as np
from scipy.signal import hilbert

def compute_pli_single_epoch(epoch):
    """
    Computes 19x19 PLI matrix for a single 5s epoch of shape (19, 640).
    
    Parameters:
        epoch (np.ndarray): Shape (19, 640)
        
    Returns:
        pli (np.ndarray): Shape (19, 19), symmetric, diagonal is 0.
    """
    # 1. Analytic signal via Hilbert transform along time axis (axis=-1)
    analytic = hilbert(epoch, axis=-1)
    
    # 2. Extract instantaneous phase angle in radians: [-pi, pi]
    phase = np.angle(analytic)  # Shape: (19, 640)
    
    # 3. Compute pairwise phase difference Delta phi_kl(t) = phi_k(t) - phi_l(t)
    # Shape: (19, 19, 640)
    phase_diff = phase[:, None, :] - phase[None, :, :]
    
    # Wrap phase difference into [-pi, pi] to handle branch cuts
    phase_diff = (phase_diff + np.pi) % (2.0 * np.pi) - np.pi
    
    # 4. Equation 7: Average sign of phase difference over time samples M
    # sign() is +1 for diff > 0, -1 for diff < 0, 0 for diff == 0
    pli = np.abs(np.mean(np.sign(phase_diff), axis=-1))  # Shape: (19, 19)
    
    # 5. Set diagonal to zero (no self-connection)
    np.fill_diagonal(pli, 0.0)
    
    return pli.astype(np.float32)

def compute_pli_dataset(
    epochs_path="data/processed/clean_epochs.npy",
    labels_path="data/processed/labels.npy",
    output_path="data/processed/pli_matrices.npy"
):
    if not os.path.exists(epochs_path):
        raise FileNotFoundError(f"Missing {epochs_path}. Run Step 1.1 first!")

    print(f"Loading preprocessed epochs from: {epochs_path} ...")
    epochs = np.load(epochs_path)   # Shape: (N, 19, 640)
    labels = np.load(labels_path)   # Shape: (N,)
    n_epochs, n_channels, n_times = epochs.shape
    
    print(f"Loaded {n_epochs} segments ({n_channels} channels, {n_times} time points).")
    print(f"Calculating Phase Lag Index (PLI) for all {n_epochs} segments...")
    
    start_time = time.time()
    pli_matrices = np.zeros((n_epochs, n_channels, n_channels), dtype=np.float32)
    
    # Compute in batches for optimal speed and memory efficiency
    batch_size = 200
    for i in range(0, n_epochs, batch_size):
        end = min(i + batch_size, n_epochs)
        batch = epochs[i:end]  # Shape: (B, 19, 640)
        
        # Batch Hilbert transform
        analytic = hilbert(batch, axis=-1)
        phase = np.angle(analytic)  # Shape: (B, 19, 640)
        
        # Pairwise phase difference: (B, 19, 19, 640)
        p_diff = phase[:, :, None, :] - phase[:, None, :, :]
        p_diff = (p_diff + np.pi) % (2.0 * np.pi) - np.pi
        
        # Equation 7
        pli_batch = np.abs(np.mean(np.sign(p_diff), axis=-1))
        
        # Zero out diagonals for the batch
        for b in range(pli_batch.shape[0]):
            np.fill_diagonal(pli_batch[b], 0.0)
            
        pli_matrices[i:end] = pli_batch
        
        if (i // batch_size) % 3 == 0 or end == n_epochs:
            elapsed = time.time() - start_time
            print(f"Processed [{end}/{n_epochs}] segments ({end/n_epochs*100:.1f}%) in {elapsed:.2f}s")
            
    total_time = time.time() - start_time
    print(f"\nPLI computation completed in {total_time:.2f} seconds!")
    print(f"Output shape: {pli_matrices.shape}, Dtype: {pli_matrices.dtype}")
    print(f"PLI Value Range: Min = {pli_matrices.min():.4f}, Max = {pli_matrices.max():.4f}, Mean = {pli_matrices.mean():.4f}")
    
    # Statistical validation preview: Compare ADHD vs Control mean connectivity
    adhd_mean_pli = pli_matrices[labels == 1].mean()
    ctrl_mean_pli = pli_matrices[labels == 0].mean()
    print(f"Mean PLI for ADHD group:    {adhd_mean_pli:.4f}")
    print(f"Mean PLI for Control group: {ctrl_mean_pli:.4f}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, pli_matrices)
    print(f"Saved successfully to: {output_path}")

if __name__ == "__main__":
    compute_pli_dataset()
