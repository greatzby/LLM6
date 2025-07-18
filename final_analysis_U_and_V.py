#!/usr/bin/env python3
"""
final_analysis_U_and_V.py

Performs a definitive, robust, and granular subspace similarity analysis
for BOTH the right (V, 'thought space') and left (U, 'language space')
singular vector subspaces.

Key features:
1.  Uses a list of fixed, data-driven truncation values 'k'.
2.  Averages results over multiple random seeds for generalization.
3.  Plots metrics over granular training iterations to show evolution dynamics.
4.  Calculates and plots standard deviations as error bars for stability.
5.  Directly compares U and V subspace evolution on the same plots.
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# --- 辅助函数 (保持不变) ---

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

def effective_rank(S):
    S_normalized = S / S.sum()
    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
    return np.exp(entropy)

# --- 核心计算函数 (已更新，同时计算U和V) ---

def compute_subspace_metrics(W1, W2, k):
    if W1 is None or W2 is None:
        return None
        
    # SVD for both models
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)

    # --- 1. V Subspace Analysis ('Thought Space') ---
    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T
    overlap_v = V1_k.T @ V2_k
    cos_angles_v = np.clip(svd(overlap_v, compute_uv=False), 0, 1)
    V2_proj_V1 = V1_k @ (V1_k.T @ V2_k)
    coverage_v = np.linalg.norm(V2_proj_V1, 'fro')**2 / np.linalg.norm(V2_k, 'fro')**2
    similarity_v = np.mean(cos_angles_v)

    # --- 2. U Subspace Analysis ('Language Space') ---
    U1_k, U2_k = U1[:, :k], U2[:, :k]
    overlap_u = U1_k.T @ U2_k
    cos_angles_u = np.clip(svd(overlap_u, compute_uv=False), 0, 1)
    U2_proj_U1 = U1_k @ (U1_k.T @ U2_k)
    coverage_u = np.linalg.norm(U2_proj_U1, 'fro')**2 / np.linalg.norm(U2_k, 'fro')**2
    similarity_u = np.mean(cos_angles_u)

    return {
        'v_similarity': similarity_v, 'v_coverage': coverage_v,
        'u_similarity': similarity_u, 'u_coverage': coverage_u,
        'er1': effective_rank(S1), 'er2': effective_rank(S2),
    }

# --- 主分析流程 (已更新，处理U和V的结果) ---

def analyze_granular_evolution(k_value):
    print("="*80)
    print(f"Granular & Robust Analysis for U and V (k = {k_value})")
    print("="*80)

    iterations = [3000, 10000, 20000, 30000, 40000, 50000]
    stages = [f'{i//1000}k' for i in iterations]
    seeds = [42, 123, 456]

    results_by_stage = {stage: [] for stage in stages}

    for i, stage in enumerate(stages):
        iter_val = iterations[i]
        print(f"\n### Analyzing Stage: {stage} (iteration {iter_val})")
        for seed in seeds:
            print(f"  - Seed: {seed}")
            W1 = load_weight_matrix(0, seed, iter_val)
            W2 = load_weight_matrix(20, seed, iter_val)
            metrics = compute_subspace_metrics(W1, W2, k=k_value)
            if metrics:
                results_by_stage[stage].append(metrics)

    summary = {}
    for stage, metrics_list in results_by_stage.items():
        if not metrics_list: continue
        summary[stage] = {
            'v_similarity_mean': np.mean([m['v_similarity'] for m in metrics_list]),
            'v_similarity_std': np.std([m['v_similarity'] for m in metrics_list]),
            'v_coverage_mean': np.mean([m['v_coverage'] for m in metrics_list]),
            'v_coverage_std': np.std([m['v_coverage'] for m in metrics_list]),
            'u_similarity_mean': np.mean([m['u_similarity'] for m in metrics_list]),
            'u_similarity_std': np.std([m['u_similarity'] for m in metrics_list]),
            'u_coverage_mean': np.mean([m['u_coverage'] for m in metrics_list]),
            'u_coverage_std': np.std([m['u_coverage'] for m in metrics_list]),
            'er1_mean': np.mean([m['er1'] for m in metrics_list]),
            'er2_mean': np.mean([m['er2'] for m in metrics_list]),
        }
    
    print("\n" + "="*80)
    print(f"SUMMARY (Averaged over seeds, k={k_value})")
    print("="*80)
    for stage, data in summary.items():
        print(f"\n{stage}:")
        print(f"  V-Similarity: {data['v_similarity_mean']:.4f} ± {data['v_similarity_std']:.4f}")
        print(f"  U-Similarity: {data['u_similarity_mean']:.4f} ± {data['u_similarity_std']:.4f}")
        print(f"  V-Coverage:   {data['v_coverage_mean']:.4f} ± {data['v_coverage_std']:.4f}")
        print(f"  U-Coverage:   {data['u_coverage_mean']:.4f} ± {data['u_coverage_std']:.4f}")

    return summary

# --- 可视化函数 (已更新，对比U和V) ---

def plot_granular_results(summary, k_value):
    stages = list(summary.keys())
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    fig.suptitle(f'Subspace Evolution Analysis: U (Language) vs. V (Thought) (k={k_value}, averaged over 3 seeds)', fontsize=22)

    # 1. 相似度演化 (U vs V)
    v_sim_means = [summary[s]['v_similarity_mean'] for s in stages]
    v_sim_stds = [summary[s]['v_similarity_std'] for s in stages]
    u_sim_means = [summary[s]['u_similarity_mean'] for s in stages]
    u_sim_stds = [summary[s]['u_similarity_std'] for s in stages]
    axes[0].errorbar(stages, v_sim_means, yerr=v_sim_stds, fmt='-o', capsize=5, label='V Similarity (Thought Space)', color='b')
    axes[0].errorbar(stages, u_sim_means, yerr=u_sim_stds, fmt='-o', capsize=5, label='U Similarity (Language Space)', color='r')
    axes[0].set_ylabel('Principal Angle Similarity')
    axes[0].set_xlabel('Training Iteration')
    axes[0].set_title('Subspace Similarity Evolution (U vs. V)', fontsize=16)
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 2. Coverage演化 (U vs V)
    v_cov_means = [summary[s]['v_coverage_mean'] for s in stages]
    v_cov_stds = [summary[s]['v_coverage_std'] for s in stages]
    u_cov_means = [summary[s]['u_coverage_mean'] for s in stages]
    u_cov_stds = [summary[s]['u_coverage_std'] for s in stages]
    axes[1].errorbar(stages, v_cov_means, yerr=v_cov_stds, fmt='-o', capsize=5, label='V Coverage (Thought Space)', color='green')
    axes[1].errorbar(stages, u_cov_means, yerr=u_cov_stds, fmt='-o', capsize=5, label='U Coverage (Language Space)', color='darkorange')
    axes[1].set_ylabel('Coverage')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_title('Subspace Coverage Evolution (U vs. V)', fontsize=16)
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. 有效秩演化 (保持不变)
    er1_means = [summary[s]['er1_mean'] for s in stages]
    er2_means = [summary[s]['er2_mean'] for s in stages]
    axes[2].plot(stages, er1_means, '-o', label='0% mix (Mean ER)', color='royalblue')
    axes[2].plot(stages, er2_means, '-o', label='20% mix (Mean ER)', color='darkorange')
    axes[2].set_xlabel('Training Iteration')
    axes[2].set_ylabel('Effective Rank')
    axes[2].set_title('Effective Rank Evolution', fontsize=16)
    axes[2].legend()
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'granular_analysis_U_and_V_k{k_value}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved as '{output_filename}'")

def main():
    # 在这里定义你要测试的k值
    # 建议先从一个k值开始，比如k=65，看看结果
    k_to_test = [60, 65, 70] 

    for k in k_to_test:
        summary_data = analyze_granular_evolution(k_value=k)
        if summary_data:
            plot_granular_results(summary_data, k_value=k)

if __name__ == "__main__":
    main()