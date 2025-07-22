#!/usr/bin/env python3
"""
sfs_spectrum_inspector_print.py

该脚本专门用于“解剖”300维实验的子空间差异。
它会首先在终端打印出完整的300个奇异值列表，
然后绘制其谱图，以验证“有效随机维度”和“冗余维度”的假设。
"""
import os
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def load_weight_matrix(ckpt_path):
    """加载单个权重矩阵"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    W = state_dict.get('lm_head.weight', None)
    if W is None:
        raise KeyError(f"Cannot find 'lm_head.weight' in {ckpt_path}")
    return W.float().numpy()

def inspect_spectrum(W1, W2):
    """计算并返回正交分量的完整奇异值谱"""
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    V1 = Vt1.T
    V2 = Vt2.T
    V2_proj_on_V1 = V1 @ (V1.T @ V2)
    V_diff_orthogonal = V2 - V2_proj_on_V1
    
    # 计算完整的奇异值，不进行任何秩的判断
    s_values = svd(V_diff_orthogonal, compute_uv=False)
    return s_values

def main():
    # --- 配置 ---
    W1_path = 'out/composition_embed300_seed42_20250722_150525/ckpt_50000.pt'
    W2_path = 'out/composition_embed300_seed123_20250722_153809/ckpt_50000.pt'
    RANK_TOLERANCE = 0.1
    OBSERVED_RANK = 92 # 从你之前的实验我们知道这个值

    # --- 执行计算 ---
    W1 = load_weight_matrix(W1_path)
    W2 = load_weight_matrix(W2_path)
    full_s_values = inspect_spectrum(W1, W2)

    # --- 在终端打印完整的奇异值列表 ---
    print("\n" + "="*80)
    print("                 FULL SINGULAR VALUE SPECTRUM (300 DIMS)")
    print("="*80 + "\n")

    # 打印前92个值
    print(f"--- Part 1: The 'Effective Random Subspace' (SFS) - First {OBSERVED_RANK} singular values ---")
    for i in range(OBSERVED_RANK):
        print(f"  Index {i+1:03d}: {full_s_values[i]:.8f}")

    print(f"\n--- Part 2: The 'Redundant/Noise Subspace' - Remaining {len(full_s_values) - OBSERVED_RANK} singular values ---")
    # 打印后208个值
    for i in range(OBSERVED_RANK, len(full_s_values)):
        # 注意这里的数值会非常非常小
        print(f"  Index {i+1:03d}: {full_s_values[i]:.8f}")
    
    print("\n" + "="*80)
    print("Terminal output complete. Now generating plot...")
    print("="*80 + "\n")


    # --- 绘图 (与之前相同) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(full_s_values, marker='.', linestyle='-', label='Singular Values of Orthogonal Component')
    
    ax.fill_between(range(OBSERVED_RANK), 0, full_s_values[:OBSERVED_RANK], color='orange', alpha=0.3, label=f'SFS (Rank ≈ {OBSERVED_RANK})')
    ax.fill_between(range(OBSERVED_RANK, len(full_s_values)), 0, full_s_values[OBSERVED_RANK:], color='skyblue', alpha=0.4, label=f'Redundant/Noise Dims (≈ {len(full_s_values) - OBSERVED_RANK})')
    ax.axhline(y=RANK_TOLERANCE, color='red', linestyle='--', label=f'Rank Tolerance = {RANK_TOLERANCE}')
    
    ax.set_title('Full Singular Value Spectrum (300-dim Experiment)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Singular Value Index (sorted descending)', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    
    output_path = "sfs_hypothesis_results/spectrum_inspection_300dim_with_print.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✓ Spectrum plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()