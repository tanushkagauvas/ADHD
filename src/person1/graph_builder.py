"""
Step 1.4: Non-Linear Fusion P-COH and Small-World Binarization (Person 1 Script)

Implements Equations (9) and (10) from the research paper:
1. Feature Fusion:
   z_kl = PLI_kl + COH_kl   (in range [0, 2])

2. Non-linear Mapping Function F(z):
   F(z) = (e^z - 1) / (2e - 2),          0 <= z <= 1
   F(z) = 1 - (e^(2-z) - 1) / (2e - 2),  1 < z <= 2

3. Small-Worldness Threshold Search:
   Iterate threshold tau in [0.30, 0.70] with step 0.01.
   Choose tau* that maximizes small-worldness sigma = (C / C_rand) / (L / L_rand).

4. Binarization:
   A_kl = 1 if P-COH_kl >= tau* else 0 (diagonal = 0)

Input:
- data/processed/pli_matrices.npy  -> Shape: (N, 19, 19), float32
- data/processed/coh_matrices.npy  -> Shape: (N, 19, 19), float32
- data/processed/labels.npy        -> Shape: (N,), int64

Output:
- data/processed/p_coh_continuous.npy    -> Shape: (N, 19, 19), float32 (Continuous P-COH for stats & plots)
- data/processed/adjacency_matrices.npy  -> Shape: (N, 19, 19), uint8   (Binary Adjacency A for GCN)
"""

import os
import time
import numpy as np
import networkx as nx

def mapping_function_F(z):
    """
    Implements Equation (10):
    Piecewise exponential mapping function mapping z in [0, 2] to [0, 1].
    Enhances contrast: reduces smaller features and increases larger ones.
    """
    e = np.e
    denom = 2.0 * e - 2.0
    
    out = np.zeros_like(z, dtype=np.float32)
    mask1 = (z >= 0.0) & (z <= 1.0)
    mask2 = (z > 1.0) & (z <= 2.0)
    
    out[mask1] = (np.exp(z[mask1]) - 1.0) / denom
    out[mask2] = 1.0 - (np.exp(2.0 - z[mask2]) - 1.0) / denom
    
    return np.clip(out, 0.0, 1.0)

def find_optimal_threshold(p_coh_matrices, tau_min=0.30, tau_max=0.70, step=0.01, sample_size=300):
    """
    Iterates threshold tau from 0.30 to 0.70 with step 0.01.
    Finds tau* that maximizes the network small-world coefficient sigma.
    """
    print(f"\n--- Searching Optimal Binarization Threshold in [{tau_min}, {tau_max}] (step {step}) ---")
    thresholds = np.arange(tau_min, tau_max + step / 2.0, step)
    
    n_total = len(p_coh_matrices)
    # Use stratified sample or first N epochs for quick, representative graph evaluation
    eval_indices = np.linspace(0, n_total - 1, min(sample_size, n_total), dtype=int)
    eval_matrices = p_coh_matrices[eval_indices]
    
    best_tau = tau_min
    max_sw = -1.0
    sw_results = {}
    
    for tau in thresholds:
        tau_val = round(float(tau), 2)
        sigmas = []
        
        for p_mat in eval_matrices:
            adj = (p_mat >= tau_val).astype(np.uint8)
            G = nx.from_numpy_array(adj)
            
            # Must have at least 3 edges to form a meaningful small-world network
            if G.number_of_edges() < 3:
                continue
                
            try:
                C = nx.average_clustering(G)
                if C <= 0:
                    continue
                    
                # Characteristic path length L in the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) < 3:
                    continue
                subG = G.subgraph(largest_cc)
                L = nx.average_shortest_path_length(subG)
                
                n = len(largest_cc)
                k = 2.0 * subG.number_of_edges() / float(n)
                if k <= 1.0 or L <= 0:
                    continue
                    
                # Small-world benchmark random graph (Watts-Strogatz / Humphries & Gurney)
                C_rand = k / float(n)
                L_rand = np.log(n) / np.log(k)
                
                if C_rand > 0 and L_rand > 0:
                    sigma = (C / C_rand) / (L / L_rand)
                    sigmas.append(sigma)
            except Exception:
                pass
                
        # Mean small-world coefficient for this threshold
        mean_sigma = float(np.mean(sigmas)) if len(sigmas) >= 10 else 0.0
        sw_results[tau_val] = mean_sigma
        
        if mean_sigma > max_sw:
            max_sw = mean_sigma
            best_tau = tau_val
            
    print(f"Optimal Threshold Selected: tau* = {best_tau:.2f} (Peak Small-Worldness sigma = {max_sw:.4f})")
    return best_tau, max_sw

def build_graph_adjacency_dataset(
    pli_path="data/processed/pli_matrices.npy",
    coh_path="data/processed/coh_matrices.npy",
    labels_path="data/processed/labels.npy",
    out_p_coh="data/processed/p_coh_continuous.npy",
    out_adj="data/processed/adjacency_matrices.npy"
):
    print("Loading PLI and COH matrices...")
    pli = np.load(pli_path)   # (N, 19, 19)
    coh = np.load(coh_path)   # (N, 19, 19)
    labels = np.load(labels_path)
    n_epochs = len(pli)
    
    t0 = time.time()
    
    # 1. Feature Fusion: z = PLI + COH (Equation 9)
    print("1. Computing feature fusion z = PLI + COH ...")
    z = pli + coh  # Shape: (N, 19, 19)
    
    # 2. Non-linear mapping F(z) (Equation 10)
    print("2. Applying non-linear mapping F(z) -> P-COH ...")
    p_coh = mapping_function_F(z)
    
    # Zero out diagonal
    for i in range(n_epochs):
        np.fill_diagonal(p_coh[i], 0.0)
        
    print(f"P-COH Matrix: Min = {p_coh.min():.4f}, Max = {p_coh.max():.4f}, Mean = {p_coh.mean():.4f}")
    
    # Save continuous P-COH for statistics and brain topographic visualization
    np.save(out_p_coh, p_coh)
    print(f"Saved continuous P-COH matrices to: {out_p_coh}")
    
    # 3. Small-Worldness Threshold Search
    optimal_tau, peak_sigma = find_optimal_threshold(p_coh, tau_min=0.30, tau_max=0.70, step=0.01)
    
    # 4. Binarization into Adjacency Matrix A
    print(f"3. Binarizing P-COH with optimal threshold tau* = {optimal_tau:.2f} ...")
    adj_matrices = (p_coh >= optimal_tau).astype(np.uint8)
    
    # Ensure zero diagonal (no self-loops in A; GCN adds self-loops dynamically via A + I)
    for i in range(n_epochs):
        np.fill_diagonal(adj_matrices[i], 0)
        
    np.save(out_adj, adj_matrices)
    print(f"Saved binary Adjacency matrices to: {out_adj}")
    
    # Summary Metrics
    total_edges = np.sum(adj_matrices) / (2.0 * n_epochs)
    adhd_edges = np.sum(adj_matrices[labels == 1]) / (2.0 * np.sum(labels == 1))
    ctrl_edges = np.sum(adj_matrices[labels == 0]) / (2.0 * np.sum(labels == 0))
    
    dt = time.time() - t0
    print(f"\nGraph Adjacency construction complete in {dt:.2f} seconds!")
    print(f" - Adjacency Tensor Shape: {adj_matrices.shape} (uint8)")
    print(f" - Average Edges per Epoch: {total_edges:.2f} edges (out of 171 max pairs)")
    print(f" - ADHD Group Average Edges:    {adhd_edges:.2f}")
    print(f" - Control Group Average Edges: {ctrl_edges:.2f}")
    print("\n>>> PERSON 1 DELIVERABLES FOR PERSON 2 ARE NOW 100% COMPLETE! <<<")

if __name__ == "__main__":
    build_graph_adjacency_dataset()
