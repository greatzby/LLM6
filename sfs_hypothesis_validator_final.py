#!/usr/bin/env python3
"""
sfs_hypothesis_validator_final_corrected_v2.py

该脚本用于验证“确定性知识子空间 (DKS) + 随机波动子空间 (SFS)”假说。
此版本修正了秩的计算逻辑，采用了用户原始脚本中正确的“子空间正交部分”的计算方法，
确保了结果的正确性。

预测: Rank(Orthogonal_Component(V2, V1)) ≈ Embedding_Dim - 92

运行: python sfs_hypothesis_validator_final_corrected_v2.py
"""
import os
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 配置 (CONFIG) - 已根据您的最终信息精确填写，无需修改
# ==============================================================================

ITERATIONS_TO_TEST = [5000, 30000, 50000]

CONFIG = {
    '200_embed': {
        'label': 'Embed Dim = 200',
        'predicted_rank': 200 - 92,
        'model_A_dir': 'out/composition_20250721_044634',
        'model_B_dir': 'out/composition_20250722_044016',
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
    '120_embed': {
        'label': 'Embed Dim = 120',
        'predicted_rank': 120 - 92,
        'model_A_dir': 'out/composition_mix5_seed42_20250705_151206',
        'model_B_dir': 'out/composition_mix5_seed123_20250705_153459',
        'ckpt_filename_template': 'ckpt_mix{ratio}_seed{seed}_iter{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
        'ratio': 5,
    },
    '60_embed': {
        'label': 'Embed Dim = 60 (Bottleneck)',
        'predicted_rank': 'N/A', # 在瓶颈情况下，预期全秩
        'model_A_dir': 'out/composition_20250721_115705',
        'model_B_dir': 'out/composition_20250722_041942',
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
}

OUTPUT_DIR = "sfs_hypothesis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 2. 核心函数 (Core Functions) - 已修正
# ==============================================================================

def load_weight_matrix(exp_config, iteration, seed):
    """根据配置加载权重矩阵 (无变化)"""
    model_dir = exp_config['model_A_dir'] if seed == exp_config['seed_A'] else exp_config['model_B_dir']
    filename = exp_config['ckpt_filename_template'].format(
        iter=iteration, seed=seed, ratio=exp_config.get('ratio'))
    ckpt_path = os.path.join(model_dir, filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"  Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    W = state_dict.get('lm_head.weight', None)
    if W is None:
        raise KeyError(f"Cannot find 'lm_head.weight' in {ckpt_path}")
    return W.float().numpy()

def analyze_subspace_difference_rank(W1, W2, rank_tolerance=0.1):
    """
    【已修正】计算子空间差异的秩。
    这使用了您原始脚本中的正确方法：计算V2中正交于V1空间的部分的秩。
    """
    # 1. 对两个权重矩阵进行SVD，获取列空间基向量V
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    V1 = Vt1.T
    V2 = Vt2.T

    # 2. 将V2投影到V1张成的子空间上
    V2_proj_on_V1 = V1 @ (V1.T @ V2)

    # 3. 计算V2中正交于V1子空间的部分 (这才是真正的“新维度”)
    V_diff_orthogonal = V2 - V2_proj_on_V1
    
    # 4. 计算这个正交部分的秩和奇异值
    #    我们使用您原始脚本中经过验证的容忍度 tol=0.1
    observed_rank = np.linalg.matrix_rank(V_diff_orthogonal, tol=rank_tolerance)
    _, s_values, _ = svd(V_diff_orthogonal, full_matrices=False)

    return observed_rank, s_values

# ==============================================================================
# 3. 主执行流程 (Main Execution) - 无需修改
# ==============================================================================

def run_analysis_and_plot():
    """执行所有实验的分析并生成最终的可视化图表。"""
    print("="*80)
    print("Starting SFS Hypothesis Validation (Final Corrected Version v2)")
    print("="*80)

    exp_order = ['200_embed', '120_embed', '60_embed']
    num_experiments = len(exp_order)
    fig, axes = plt.subplots(1, num_experiments, figsize=(7 * num_experiments, 6), sharey=True)
    if num_experiments == 1: axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ITERATIONS_TO_TEST)))

    for i, exp_key in enumerate(exp_order):
        exp_info = CONFIG[exp_key]
        ax = axes[i]
        print(f"\n--- Running Experiment: {exp_key} ---")
        
        for j, iteration in enumerate(ITERATIONS_TO_TEST):
            try:
                W1 = load_weight_matrix(exp_info, iteration, exp_info['seed_A'])
                W2 = load_weight_matrix(exp_info, iteration, exp_info['seed_B'])
                
                # 调用修正后的、正确的分析函数
                observed_rank, s_values = analyze_subspace_difference_rank(W1, W2)
                
                print(f"  Iteration: {iteration}")
                print(f"    Predicted SFS Rank: {exp_info['predicted_rank']}")
                print(f"    Observed Difference Rank: {observed_rank}")
                
                ax.plot(s_values, marker='o', linestyle='-', markersize=4,
                        label=f'Iter {iteration}k (Rank={observed_rank})', color=colors[j])

            except FileNotFoundError as e:
                print(f"  Skipping Iteration {iteration}: {e}")
            except Exception as e:
                print(f"  An error occurred at Iteration {iteration}: {e}")
        
        ax.set_title(exp_info['label'], fontsize=16, fontweight='bold')
        ax.set_xlabel("Singular Value Index of Orthogonal Component", fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        if isinstance(exp_info['predicted_rank'], int):
            ax.axvline(x=exp_info['predicted_rank'], color='red', linestyle='--', linewidth=2.5,
                       label=f"Predicted Rank = {exp_info['predicted_rank']}")
        
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Singular Value of Orthogonal Component", fontsize=12)
    fig.suptitle("SFS Hypothesis Validation: Singular Value Spectrum of Subspace Difference", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, "sfs_rank_validation_final_corrected_v2.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✓ Analysis complete! Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis_and_plot()