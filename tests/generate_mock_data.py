"""
Mock Data Generator
Generates dummy EEG epochs and adjacency matrices matching the paper's exact shapes:
- Raw Epochs: (N, 19, 640)  -> 19 channels, 5s windows at 128Hz (640 samples)
- Adjacency Matrices: (N, 19, 19) -> Symmetric binary small-world graphs
- Node Features: (N, 19, 4) -> 19 channels, [2 Time + 2 Freq] features
- Labels: (N,) -> 0 (Healthy Control), 1 (ADHD)
"""

import os
import numpy as np

def generate_mock_dataset(n_samples=100, save_dir="data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating mock dataset with {n_samples} samples...")
    
    # 1. Raw 5-second EEG epochs (N, 19 channels, 640 time points)
    raw_epochs = np.random.randn(n_samples, 19, 640).astype(np.float32)
    
    # 2. Binary Adjacency Matrices (N, 19, 19)
    adj_matrices = np.zeros((n_samples, 19, 19), dtype=np.uint8)
    for i in range(n_samples):
        # Generate random symmetric matrix with zero diagonal
        p_matrix = np.random.rand(19, 19)
        sym = (p_matrix + p_matrix.T) / 2.0
        binary_adj = (sym > 0.65).astype(np.uint8)
        np.fill_diagonal(binary_adj, 0)
        adj_matrices[i] = binary_adj
        
    # 3. Continuous P-COH matrices (for Mann-Whitney U test & brain plots)
    p_coh = np.random.rand(n_samples, 19, 19).astype(np.float32)
    for i in range(n_samples):
        p_coh[i] = (p_coh[i] + p_coh[i].T) / 2.0
        np.fill_diagonal(p_coh[i], 0.0)
        
    # 4. Binary Labels: 0 (Control) and 1 (ADHD)
    labels = np.random.choice([0, 1], size=(n_samples,)).astype(np.int64)
    
    # 5. Node Features: (N, 19 nodes, 4 features = 2 time + 2 freq)
    node_features = np.random.randn(n_samples, 19, 4).astype(np.float32)
    
    # Save files
    np.save(os.path.join(save_dir, "clean_epochs.npy"), raw_epochs)
    np.save(os.path.join(save_dir, "adjacency_matrices.npy"), adj_matrices)
    np.save(os.path.join(save_dir, "p_coh_continuous.npy"), p_coh)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    np.save(os.path.join(save_dir, "node_features.npy"), node_features)
    
    print(f"Successfully saved mock data into '{save_dir}/':")
    print(f" - clean_epochs.npy:        {raw_epochs.shape}")
    print(f" - adjacency_matrices.npy:  {adj_matrices.shape}")
    print(f" - p_coh_continuous.npy:    {p_coh.shape}")
    print(f" - labels.npy:              {labels.shape}")
    print(f" - node_features.npy:       {node_features.shape}")

if __name__ == "__main__":
    generate_mock_dataset(n_samples=50)
