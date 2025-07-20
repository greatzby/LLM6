#!/usr/bin/env python3
"""
dimension_contribution_analysis.py
分析哪些具体维度对0% vs 20%的差异贡献最大
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# [复用上面的辅助函数 get_checkpoint_path 和 load_weight_matrix]

def get_checkpoint_path(ratio, seed, iteration):
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No directory found for ratio={ratio}, seed={seed}")
    selected_dir = dirs[-1]
    return f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"

def load_weight_matrix(ratio, seed, iteration):
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

def analyze_dimension_contributions():
    """分析每个维度的贡献"""
    print("="*80)
    print("DIMENSION-LEVEL CONTRIBUTION ANALYSIS")
    print("="*80)
    
    # 加载最终模型
    W_0 = load_weight_matrix(0, 42, 50000)
    W_20 = load_weight_matrix(20, 42, 50000)
    
    # SVD分解
    U_0, S_0, Vt_0 = svd(W_0, full_matrices=False)
    U_20, S_20, Vt_20 = svd(W_20, full_matrices=False)
    
    # 1. 分析V空间每个维度的差异
    print("\n### V-Space (Thought Space) Dimension Analysis ###")
    
    n_dims = min(Vt_0.shape[0], 120)  # 分析所有维度
    v_angles = []
    
    for i in range(n_dims):
        v1 = Vt_0[i, :]  # 第i个右奇异向量
        v2 = Vt_20[i, :]
        
        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        v_angles.append(angle)
        
        # 打印前10个和后10个
        if i < 10 or i >= n_dims - 10:
            print(f"  Dim {i+1}: angle = {angle:.2f}°, energy = {S_0[i]:.3f}")
    
    # 2. 识别关键维度
    v_angles = np.array(v_angles)
    threshold = 30  # 30度以上认为是显著差异
    critical_dims = np.where(v_angles > threshold)[0]
    
    print(f"\n### Critical Dimensions (angle > {threshold}°) ###")
    print(f"Found {len(critical_dims)} critical dimensions: {critical_dims[:10]}...")
    
    # 3. 分析关键维度的模式
    print("\n### Pattern Analysis of Critical Dimensions ###")
    
    # 累积能量分析
    cumsum_energy_0 = np.cumsum(S_0) / np.sum(S_0)
    cumsum_energy_20 = np.cumsum(S_20) / np.sum(S_20)
    
    print(f"\nEnergy distribution:")
    for k in [10, 20, 30, 50, 70]:
        print(f"  Top {k} dims: 0% model = {cumsum_energy_0[k-1]:.2%}, "
              f"20% model = {cumsum_energy_20[k-1]:.2%}")
    
    # 4. 可视化
    plot_dimension_analysis(v_angles, S_0, S_20, critical_dims)
    
    # 5. 功能分析（需要具体的任务知识）
    print("\n### Functional Interpretation (Hypothesis) ###")
    print("Based on dimension indices and energy distribution:")
    print("- High-energy dims (1-15): Core structural patterns")
    print("- Mid-energy dims (16-50): Task-specific combinations")  
    print("- Low-energy dims (51+): Fine-grained variations")
    
    if len(critical_dims) > 0:
        critical_in_high = np.sum(critical_dims < 15)
        critical_in_mid = np.sum((critical_dims >= 15) & (critical_dims < 50))
        critical_in_low = np.sum(critical_dims >= 50)
        
        print(f"\nCritical dimensions distribution:")
        print(f"  In high-energy range: {critical_in_high}")
        print(f"  In mid-energy range: {critical_in_mid}")
        print(f"  In low-energy range: {critical_in_low}")
    
    return v_angles, critical_dims

def plot_dimension_analysis(v_angles, S_0, S_20, critical_dims):
    """可视化维度分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 每个维度的角度
    ax = axes[0, 0]
    ax.bar(range(len(v_angles)), v_angles, color='steelblue')
    ax.axhline(y=30, color='red', linestyle='--', label='30° threshold')
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('V-Space Angular Difference per Dimension')
    ax.legend()
    
    # 2. 奇异值分布对比
    ax = axes[0, 1]
    dims = range(min(len(S_0), 80))
    ax.semilogy(dims, S_0[:len(dims)], 'b-', label='0% model', linewidth=2)
    ax.semilogy(dims, S_20[:len(dims)], 'r--', label='20% model', linewidth=2)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title('Singular Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 角度vs能量散点图
    ax = axes[1, 0]
    ax.scatter(S_0[:len(v_angles)], v_angles, alpha=0.6)
    if len(critical_dims) > 0:
        ax.scatter(S_0[critical_dims], v_angles[critical_dims], 
                  color='red', s=100, label='Critical dims')
    ax.set_xlabel('Singular Value (0% model)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Angle vs Energy Relationship')
    ax.set_xscale('log')
    if len(critical_dims) > 0:
        ax.legend()
    
    # 4. 累积角度变化
    ax = axes[1, 1]
    cumulative_angle_change = np.cumsum(v_angles)
    ax.plot(cumulative_angle_change, 'g-', linewidth=2)
    ax.fill_between(range(len(v_angles)), 0, cumulative_angle_change, alpha=0.3)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Cumulative Angle Change')
    ax.set_title('Cumulative Angular Divergence')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimension_contribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'dimension_contribution_analysis.png'")

if __name__ == "__main__":
    analyze_dimension_contributions()