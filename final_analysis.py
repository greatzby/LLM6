#!/usr/bin/env python3
"""
final_analysis.py

Performs the definitive, robust, and granular subspace similarity analysis.
Key features:
1. Uses a list of fixed, data-driven truncation values 'k' for sensitivity analysis.
2. Averages results over multiple random seeds for generalization.
3. Plots metrics over a granular set of training iterations to show evolution dynamics.
4. Calculates and plots standard deviations as error bars to show result stability.
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
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found for ratio={ratio}, seed={seed}")
    selected_dir = sorted(dirs)[-1]
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

# --- 核心计算函数 (保持不变) ---

def compute_subspace_metrics(W1, W2, k):
    if W1 is None or W2 is None:
        return None
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T
    overlap = V1_k.T @ V2_k
    cos_angles = np.clip(svd(overlap, compute_uv=False), 0, 1)
    V2_proj_V1 = V1_k @ (V1_k.T @ V2_k)
    coverage = np.linalg.norm(V2_proj_V1, 'fro')**2 / np.linalg.norm(V2_k, 'fro')**2
    similarity = np.mean(cos_angles)
    return {
        'similarity': similarity, 'coverage': coverage,
        'er1': effective_rank(S1), 'er2': effective_rank(S2),
        'cos_angles': cos_angles
    }

# --- 主分析流程 (已更新迭代点) ---

def analyze_granular_evolution(k_value):
    print("="*80)
    print(f"Granular & Robust Analysis (k = {k_value})")
    print("="*80)

    # *** 更新：使用更细粒度的迭代点 ***
    iterations = [3000, 10000, 20000, 30000, 40000, 50000]
    stages = [f'{i//1000}k' for i in iterations] # ['3k', '10k', '20k', ...]
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
            'similarity_mean': np.mean([m['similarity'] for m in metrics_list]),
            'similarity_std': np.std([m['similarity'] for m in metrics_list]),
            'coverage_mean': np.mean([m['coverage'] for m in metrics_list]),
            'coverage_std': np.std([m['coverage'] for m in metrics_list]),
            'er1_mean': np.mean([m['er1'] for m in metrics_list]),
            'er2_mean': np.mean([m['er2'] for m in metrics_list]),
        }
    
    print("\n" + "="*80)
    print(f"SUMMARY (Averaged over seeds, k={k_value})")
    print("="*80)
    for stage, data in summary.items():
        print(f"\n{stage}:")
        print(f"  Similarity: {data['similarity_mean']:.4f} ± {data['similarity_std']:.4f}")
        print(f"  Coverage:   {data['coverage_mean']:.4f} ± {data['coverage_std']:.4f}")

    return summary

# --- 可视化函数 (已更新以适应更多点) ---

def plot_granular_results(summary, k_value):
    stages = list(summary.keys())
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) # 改为1x3布局，更适合展示演化曲线
    fig.suptitle(f'Granular Evolution Analysis (k={k_value}, averaged over 3 seeds)', fontsize=20)

    # 1. 相似度演化 (带误差棒)
    sim_means = [summary[s]['similarity_mean'] for s in stages]
    sim_stds = [summary[s]['similarity_std'] for s in stages]
    axes[0].errorbar(stages, sim_means, yerr=sim_stds, fmt='-o', capsize=5, label='Mean Similarity', color='b')
    axes[0].set_ylabel('Principal Angle Similarity')
    axes[0].set_xlabel('Training Iteration')
    axes[0].set_title('Subspace Similarity Evolution')
    axes[0].legend()

    # 2. Coverage演化 (带误差棒)
    cov_means = [summary[s]['coverage_mean'] for s in stages]
    cov_stds = [summary[s]['coverage_std'] for s in stages]
    axes[1].errorbar(stages, cov_means, yerr=cov_stds, fmt='-o', color='green', capsize=5, label='Mean Coverage')
    axes[1].set_ylabel('Coverage')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_title('Coverage of 0% Subspace by 20% Subspace')
    axes[1].legend()

    # 3. 有效秩演化 (折线图)
    er1_means = [summary[s]['er1_mean'] for s in stages]
    er2_means = [summary[s]['er2_mean'] for s in stages]
    axes[2].plot(stages, er1_means, '-o', label='0% mix (Mean ER)', color='royalblue')
    axes[2].plot(stages, er2_means, '-o', label='20% mix (Mean ER)', color='darkorange')
    axes[2].set_xlabel('Training Iteration')
    axes[2].set_ylabel('Effective Rank')
    axes[2].set_title('Effective Rank Evolution')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = f'granular_analysis_k{k_value}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved as '{output_filename}'")

def main():
    # *** 更新：在这里定义你要测试的k值 ***
    k_to_test = [60, 65, 70] 

    for k in k_to_test:
        summary_data = analyze_granular_evolution(k_value=k)
        if summary_data:
            plot_granular_results(summary_data, k_value=k)

if __name__ == "__main__":
    main()