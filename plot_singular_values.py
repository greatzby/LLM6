#!/usr/bin/env python3
"""
plot_singular_values.py

This script specifically addresses the TODO from Prof. Liang:
Plot the singular value distributions for the 6 final models 
(3 seeds x 2 mix ratios) to visually identify a truncation threshold.
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# --- 我们从你之前的代码中复用这些辅助函数 ---

def get_checkpoint_path(ratio, seed, iteration):
    """Gets the path to a specific model checkpoint."""
    # 使用最新的run
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found for ratio={ratio}, seed={seed}")
    
    # 假设最新的目录是正确的
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

def plot_all_singular_values():
    """
    Loads the 6 models, computes their SVD, and plots their singular value
    distributions on a single figure with 6 subplots.
    """
    seeds = [42, 123, 456]  # 你的三个随机种子
    ratios = [0, 20]        # 你的两个混合比例
    
    # 创建一个2行3列的子图网格
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle('Singular Value Distributions of Final Models (iter=50k)', fontsize=20)

    for i, ratio in enumerate(ratios):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            print(f"Processing model: ratio={ratio}, seed={seed}...")
            
            try:
                # 加载最终的模型权重
                W = load_weight_matrix(ratio, seed)
                
                # 计算奇异值 (我们只需要奇异值，不需要U和V)
                S = svd(W, compute_uv=False)
                
                # 绘图
                ax.plot(S, 'b-', marker='.', markersize=4, label=f'Singular Values')
                
                # --- 这是最关键的一步：使用对数坐标轴 ---
                ax.set_yscale('log')
                
                ax.set_title(f'{ratio}% Mix, Seed {seed}', fontsize=14)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                # 在第一个子图上添加图例
                if i == 0 and j == 0:
                    ax.legend()

            except FileNotFoundError as e:
                print(e)
                ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center', color='red')
                ax.set_title(f'{ratio}% Mix, Seed {seed}', fontsize=14)


    # 设置共享的坐标轴标签
    fig.text(0.5, 0.04, 'Singular Value Rank Index', ha='center', va='center', fontsize=16)
    fig.text(0.08, 0.5, 'Singular Value (Log Scale)', ha='center', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.09, 0.05, 1, 0.95]) # 调整布局以适应大标题
    
    # 保存图像
    output_filename = 'singular_value_distributions.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved as '{output_filename}'")
    plt.close()


if __name__ == "__main__":
    plot_all_singular_values()