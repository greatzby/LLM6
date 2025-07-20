#!/usr/bin/env python3
"""
analyze_ratio_evolution.py

This script generates the "killer evidence" plot by quantifying the
ratio of singular values (0% model / 20% model) over training.

It directly addresses the potential skepticism that the changes are "not obvious"
by transforming the subtle absolute differences into a clear, quantitative
ratio plot centered around a baseline of 1.0.
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
    """Main function to run the analysis and generate the ratio evolution plot."""
    print("="*80)
    print("Analyzing the EVOLUTION of Singular Value RATIOS")
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
            s0_stack = np.vstack(all_s_0)
            s20_stack = np.vstack(all_s_20)
            results[it] = {
                's0_mean': np.mean(s0_stack, axis=0),
                's20_mean': np.mean(s20_stack, axis=0),
            }

    if not results:
        print("\nError: Could not load any data. Aborting plot generation.")
        return

    print("\nData collection complete. Generating ratio evolution plot...")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True, sharey=True)
    fig.suptitle('Evolution of Singular Value Ratio (0% Model / 20% Model) Over Training', fontsize=24, y=0.97)
    axes = axes.flatten()

    for i, it in enumerate(iterations):
        if it not in results:
            continue
        
        ax = axes[i]
        data = results[it]
        
        # --- CORE CALCULATION: THE RATIO ---
        # Use np.divide for safe division, handle cases where s20_mean might be zero
        s_ratio = np.divide(data['s0_mean'], data['s20_mean'], 
                            out=np.ones_like(data['s0_mean']), 
                            where=data['s20_mean']!=0)
        
        # Plot the ratio
        ax.plot(s_ratio, color='purple', linewidth=2.5, label='Ratio (0% / 20%)')
        
        # Plot the crucial y=1.0 baseline
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Baseline (No Difference)')
        
        ax.set_title(f'Iteration {it//1000}k', fontsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, len(s_ratio) - 1)
        
        # Set y-axis limits to better focus on the deviation from 1.0
        ax.set_ylim(0.8, 1.2)

    # Set common labels
    fig.text(0.5, 0.04, 'Singular Value Index (from largest to smallest)', ha='center', va='center', fontsize=18)
    fig.text(0.08, 0.5, 'Singular Value Ratio', ha='center', va='center', rotation='vertical', fontsize=18)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.92), fontsize=14)

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.95])
    output_filename = 'singular_value_ratio_evolution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nRatio evolution plot successfully saved as '{output_filename}'")
    print("This is your key evidence to counter skepticism.")
    print("="*80)

if __name__ == "__main__":
    main()