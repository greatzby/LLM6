#!/usr/bin/env python3
"""
plot_singular_values_linear.py

This script MODIFIES the original plot based on Prof. Liang's feedback:
1.  Uses a LINEAR scale for the y-axis.
2.  Ensures all 6 subplots share the EXACT SAME y-axis range for fair comparison.
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 我们从你之前的代码中复用这些辅助函数 ---

def get_checkpoint_path(ratio, seed, iteration):
    """Gets the path to a specific model checkpoint."""
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found for ratio={ratio}, seed={seed}")
    
    selected_dir = sorted(dirs)[-1]
    ckpt_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    return ckpt_path

def load_weight_matrix(ratio, seed, iteration=50000):
    """Loads the lm_head weight matrix from a checkpoint."""
    path = get_checkpoint_path(ratio, seed, iteration)
    checkpoint = torch.load(path, map_location='cpu')
    
    state_dict = checkpoint.get('model', checkpoint)
    W = state_dict['lm_head.weight'].float().numpy()
    
    del checkpoint, state_dict
    return W

# --- 这是新的核心绘图函数 ---

def plot_all_singular_values_linear():
    """
    Loads the 6 models, computes their SVD, and plots their singular value
    distributions on a single figure with a LINEAR and UNIFIED y-axis.
    """
    seeds = [42, 123, 456]
    ratios = [0, 20]
    
    all_singular_values = {}
    global_max_sv = 0

    # --- 第一步: 加载所有数据并找到全局最大奇异值 ---
    print("--- Step 1: Loading all models and finding global max singular value ---")
    for ratio in ratios:
        for seed in seeds:
            print(f"Loading model: ratio={ratio}, seed={seed}...")
            try:
                W = load_weight_matrix(ratio, seed)
                S = svd(W, compute_uv=False)
                all_singular_values[(ratio, seed)] = S
                
                # 更新全局最大值
                if S[0] > global_max_sv:
                    global_max_sv = S[0]

            except FileNotFoundError as e:
                print(e)
                all_singular_values[(ratio, seed)] = None

    print(f"\nGlobal maximum singular value found: {global_max_sv:.2f}")

    # --- 第二步: 使用统一的Y轴范围进行绘图 ---
    print("\n--- Step 2: Plotting all distributions with a unified linear y-axis ---")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharex=True)
    fig.suptitle('Singular Value Distributions (Linear Scale, Unified Y-Axis)', fontsize=20)

    for i, ratio in enumerate(ratios):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            S = all_singular_values.get((ratio, seed))
            
            if S is not None:
                ax.plot(S, 'b-', marker='.', markersize=4)
                ax.set_title(f'{ratio}% Mix, Seed {seed}', fontsize=14)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                # **核心修改**: 设置统一的Y轴范围
                ax.set_ylim(0, global_max_sv * 1.05) # 留出5%的顶部空间
                
                # 格式化Y轴标签，使其更易读
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

            else:
                ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center', color='red')
                ax.set_title(f'{ratio}% Mix, Seed {seed}', fontsize=14)

    # 设置共享的坐标轴标签
    fig.text(0.5, 0.04, 'Singular Value Rank Index', ha='center', va='center', fontsize=16)
    fig.text(0.08, 0.5, 'Singular Value (Linear Scale)', ha='center', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.09, 0.05, 1, 0.95])
    
    output_filename = 'singular_value_distributions_linear.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved as '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    plot_all_singular_values_linear()