#!/usr/bin/env python3
"""
analyze_singular_value_evolution.py

A more powerful analysis suggested by the user to visualize the DYNAMIC
EVOLUTION of the singular value spectra over the entire course of training.

This script generates a multi-panel plot to show how the "capability gap"
(difference in singular value magnitudes) between the 0% and 20% models
widens over time.
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# --- 辅助函数 (与之前脚本一致) ---

def get_checkpoint_path(ratio, seed, iteration):
    """Finds the checkpoint path for a given configuration."""
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No directory found for ratio={ratio}, seed={seed}")
    selected_dir = dirs[-1]
    return f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"

def load_weight_matrix(ratio, seed, iteration):
    """Loads the lm_head.weight matrix from a checkpoint."""
    try:
        path = get_checkpoint_path(ratio, seed, iteration)
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        W = state_dict['lm_head.weight'].float().numpy()
        del checkpoint, state_dict
        return W
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

def compute_singular_values(W):
    """Performs SVD and returns the singular values."""
    if W is None:
        return None
    _, s, _ = svd(W, full_matrices=False)
    return s

# --- 主分析与绘图流程 ---

def main():
    """Main function to run the analysis and generate the evolution plot."""
    print("="*80)
    print("Analyzing the EVOLUTION of Singular Value Spectra")
    print("="*80)

    iterations = [3000, 10000, 20000, 30000, 40000, 50000]
    seeds = [42, 123, 456]
    
    # --- Data Collection ---
    results = {}
    for it in iterations:
        print(f"\n--- Processing Iteration: {it} ---")
        all_s_0, all_s_20 = [], []
        for seed in seeds:
            print(f"  - Seed {seed}")
            W0 = load_weight_matrix(0, seed, it)
            W20 = load_weight_matrix(20, seed, it)
            
            s0 = compute_singular_values(W0)
            s20 = compute_singular_values(W20)

            if s0 is not None and s20 is not None:
                all_s_0.append(s0)
                all_s_20.append(s20)
        
        if all_s_0 and all_s_20:
            # Aggregate and store results for this iteration
            s0_stack = np.vstack(all_s_0)
            s20_stack = np.vstack(all_s_20)
            results[it] = {
                's0_mean': np.mean(s0_stack, axis=0),
                's0_std': np.std(s0_stack, axis=0),
                's20_mean': np.mean(s20_stack, axis=0),
                's20_std': np.std(s20_stack, axis=0),
            }

    if not results:
        print("\nError: Could not load any data. Aborting plot generation.")
        return

    print("\nData collection complete. Generating evolution plot...")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a 2x3 grid of subplots, sharing both x and y axes for comparison
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True, sharey=True)
    fig.suptitle('Evolution of Singular Value Spectra Over Training', fontsize=24, y=0.97)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, it in enumerate(iterations):
        if it not in results:
            continue
        
        ax = axes[i]
        data = results[it]
        
        # Plot the mean spectra
        ax.plot(data['s0_mean'], label='0% Compositionality (Degraded)', color='salmon', linewidth=2)
        ax.plot(data['s20_mean'], label='20% Compositionality (Ideal)', color='skyblue', linewidth=2)
        
        # Plot the standard deviation bands
        ax.fill_between(range(len(data['s0_mean'])), data['s0_mean'] - data['s0_std'], data['s0_mean'] + data['s0_std'], color='salmon', alpha=0.2)
        ax.fill_between(range(len(data['s20_mean'])), data['s20_mean'] - data['s20_std'], data['s20_mean'] + data['s20_std'], color='skyblue', alpha=0.2)
        
        ax.set_yscale('log')
        ax.set_title(f'Iteration {it//1000}k', fontsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, len(data['s0_mean']) - 1)

    # Set common labels
    fig.text(0.5, 0.04, 'Singular Value Index (from largest to smallest)', ha='center', va='center', fontsize=18)
    fig.text(0.08, 0.5, 'Singular Value Magnitude (log scale)', ha='center', va='center', rotation='vertical', fontsize=18)
    
    # Create a single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.92), fontsize=14)

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.95]) # Adjust layout to make space for titles and labels
    output_filename = 'singular_value_evolution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nEvolution plot successfully saved as '{output_filename}'")
    print("="*80)

if __name__ == "__main__":
    main()