#!/usr/bin/env python3
"""
compare_coverage_directions.py

A crucial experiment to distinguish between "Drift" and "Reduction/Collapse"
by comparing coverage in both directions: Cov(0% -> 20%) and Cov(20% -> 0%).

- If Cov(20% -> 0%) is high (~1.0) while Cov(0% -> 20%) is lower, it proves
  a "Reduction" relationship (V_0 is a subset of V_20).
- If both are low, it suggests a "Drift" relationship.
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

# --- 核心计算函数 (已更新，计算双向Coverage) ---

def compute_bidirectional_coverage(W1, W2, k):
    """
    Computes similarity and coverage in both directions.
    W1 is assumed to be the 0% model, W2 is the 20% model.
    """
    if W1 is None or W2 is None:
        return None
        
    U1, _, Vt1 = svd(W1, full_matrices=False)
    U2, _, Vt2 = svd(W2, full_matrices=False)

    # --- V Subspace Analysis ('Thought Space') ---
    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T # V1_k from 0% model, V2_k from 20% model
    
    # Coverage(0% -> 20%): How much of the IDEAL (20%) space is covered by the DEGRADED (0%) space
    V2_proj_on_V1 = V1_k @ (V1_k.T @ V2_k)
    coverage_0_covers_20_v = np.linalg.norm(V2_proj_on_V1, 'fro')**2 / np.linalg.norm(V2_k, 'fro')**2

    # Coverage(20% -> 0%): How much of the DEGRADED (0%) space is covered by the IDEAL (20%) space
    V1_proj_on_V2 = V2_k @ (V2_k.T @ V1_k)
    coverage_20_covers_0_v = np.linalg.norm(V1_proj_on_V2, 'fro')**2 / np.linalg.norm(V1_k, 'fro')**2

    # --- U Subspace Analysis ('Language Space') ---
    U1_k, U2_k = U1[:, :k], U2[:, :k] # U1_k from 0% model, U2_k from 20% model

    # Coverage(0% -> 20%)
    U2_proj_on_U1 = U1_k @ (U1_k.T @ U2_k)
    coverage_0_covers_20_u = np.linalg.norm(U2_proj_on_U1, 'fro')**2 / np.linalg.norm(U2_k, 'fro')**2

    # Coverage(20% -> 0%)
    U1_proj_on_U2 = U2_k @ (U2_k.T @ U1_k)
    coverage_20_covers_0_u = np.linalg.norm(U1_proj_on_U2, 'fro')**2 / np.linalg.norm(U1_k, 'fro')**2

    return {
        'v_cov_0_covers_20': coverage_0_covers_20_v,
        'v_cov_20_covers_0': coverage_20_covers_0_v,
        'u_cov_0_covers_20': coverage_0_covers_20_u,
        'u_cov_20_covers_0': coverage_20_covers_0_u,
    }

# --- 主分析流程 ---

def analyze_coverage_directions(k_value):
    print("="*80)
    print(f"Bidirectional Coverage Analysis (k = {k_value})")
    print("="*80)

    iterations = [3000, 10000, 20000, 30000, 40000, 50000]
    stages = [f'{i//1000}k' for i in iterations]
    seeds = [42, 123, 456]

    results_by_stage = {stage: [] for stage in stages}

    for i, stage in enumerate(stages):
        iter_val = iterations[i]
        print(f"\n### Analyzing Stage: {stage} (iteration {iter_val})")
        stage_metrics = []
        for seed in seeds:
            # W1 is 0% model, W2 is 20% model
            W1 = load_weight_matrix(0, seed, iter_val)
            W2 = load_weight_matrix(20, seed, iter_val)
            metrics = compute_bidirectional_coverage(W1, W2, k=k_value)
            if metrics:
                stage_metrics.append(metrics)
        results_by_stage[stage] = stage_metrics

    summary = {}
    for stage, metrics_list in results_by_stage.items():
        if not metrics_list: continue
        summary[stage] = {
            'v_0_covers_20_mean': np.mean([m['v_cov_0_covers_20'] for m in metrics_list]),
            'v_0_covers_20_std': np.std([m['v_cov_0_covers_20'] for m in metrics_list]),
            'v_20_covers_0_mean': np.mean([m['v_cov_20_covers_0'] for m in metrics_list]),
            'v_20_covers_0_std': np.std([m['v_cov_20_covers_0'] for m in metrics_list]),
            'u_0_covers_20_mean': np.mean([m['u_cov_0_covers_20'] for m in metrics_list]),
            'u_0_covers_20_std': np.std([m['u_cov_0_covers_20'] for m in metrics_list]),
            'u_20_covers_0_mean': np.mean([m['u_cov_20_covers_0'] for m in metrics_list]),
            'u_20_covers_0_std': np.std([m['u_cov_20_covers_0'] for m in metrics_list]),
        }
    
    print("\n" + "="*80)
    print(f"SUMMARY (Averaged over seeds, k={k_value})")
    print("="*80)
    for stage, data in summary.items():
        print(f"\n{stage}:")
        print(f"  V-Space Cov(0% -> 20%): {data['v_0_covers_20_mean']:.4f} ± {data['v_0_covers_20_std']:.4f}")
        print(f"  V-Space Cov(20% -> 0%): {data['v_20_covers_0_mean']:.4f} ± {data['v_20_covers_0_std']:.4f}")
        print(f"  U-Space Cov(0% -> 20%): {data['u_0_covers_20_mean']:.4f} ± {data['u_0_covers_20_std']:.4f}")
        print(f"  U-Space Cov(20% -> 0%): {data['u_20_covers_0_mean']:.4f} ± {data['u_20_covers_0_std']:.4f}")

    return summary

# --- 可视化函数 ---

def plot_coverage_comparison(summary, k_value):
    stages = list(summary.keys())
    x = np.arange(len(stages))  # the label locations
    width = 0.35  # the width of the bars

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle(f'Bidirectional Coverage Analysis (k={k_value}, averaged over 3 seeds)', fontsize=22)

    # --- V-Space (Thought Space) Plot ---
    v_0_covers_20 = [summary[s]['v_0_covers_20_mean'] for s in stages]
    v_20_covers_0 = [summary[s]['v_20_covers_0_mean'] for s in stages]
    v_err_0_covers_20 = [summary[s]['v_0_covers_20_std'] for s in stages]
    v_err_20_covers_0 = [summary[s]['v_20_covers_0_std'] for s in stages]

    ax1.bar(x - width/2, v_0_covers_20, width, yerr=v_err_0_covers_20, label='Cov(0% -> 20%) [Degraded explains Ideal]', capsize=5, color='salmon')
    ax1.bar(x + width/2, v_20_covers_0, width, yerr=v_err_20_covers_0, label='Cov(20% -> 0%) [Ideal explains Degraded]', capsize=5, color='skyblue')
    
    ax1.set_ylabel('Coverage Ratio')
    ax1.set_xlabel('Training Iteration')
    ax1.set_title('V-Space (Thought Space) Coverage', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(0.7, 1.05) # Zoom in on the relevant range

    # --- U-Space (Language Space) Plot ---
    u_0_covers_20 = [summary[s]['u_0_covers_20_mean'] for s in stages]
    u_20_covers_0 = [summary[s]['u_20_covers_0_mean'] for s in stages]
    u_err_0_covers_20 = [summary[s]['u_0_covers_20_std'] for s in stages]
    u_err_20_covers_0 = [summary[s]['u_20_covers_0_std'] for s in stages]

    ax2.bar(x - width/2, u_0_covers_20, width, yerr=u_err_0_covers_20, label='Cov(0% -> 20%)', capsize=5, color='salmon')
    ax2.bar(x + width/2, u_20_covers_0, width, yerr=u_err_20_covers_0, label='Cov(20% -> 0%)', capsize=5, color='skyblue')

    ax2.set_xlabel('Training Iteration')
    ax2.set_title('U-Space (Language Space) Coverage', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'bidirectional_coverage_k{k_value}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved as '{output_filename}'")

def main():
    # 建议使用一个较大的k值，比如k=60或k=92，来全面评估子空间关系
    k_to_test = 60
    summary_data = analyze_coverage_directions(k_value=k_to_test)
    if summary_data:
        plot_coverage_comparison(summary_data, k_value=k_to_test)

if __name__ == "__main__":
    main()