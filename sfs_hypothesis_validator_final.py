#!/usr/bin/env python3
"""
sfs_hypothesis_validator_final_corrected.py

该脚本用于验证“确定性知识子空间 (DKS) + 随机波动子空间 (SFS)”假说。
此版本已根据用户提供的最终、精确的路径、文件名格式以及维度更正进行配置，可直接运行。

预测: Rank(W_seed1 - W_seed2) ≈ Embedding_Dim - 92

运行: python sfs_hypothesis_validator_final_corrected.py
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

# 定义要进行比较的检查点迭代次数
ITERATIONS_TO_TEST = [5000, 30000, 50000]

# 定义每个实验的参数和模型路径信息
CONFIG = {
    '200_embed': {
        'label': 'Embed Dim = 200',
        'predicted_rank': 200 - 92,
        'model_A_dir': 'out/composition_20250721_044634', # Seed 42
        'model_B_dir': 'out/composition_20250722_044016', # Seed 123
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
    '120_embed': { # <--- 已从 100 更正为 120
        'label': 'Embed Dim = 120', # <--- 已更正
        'predicted_rank': 120 - 92, # <--- 已从 100-92=8 更正为 120-92=28
        'model_A_dir': 'out/composition_mix5_seed42_20250705_151206', # Seed 42
        'model_B_dir': 'out/composition_mix5_seed123_20250705_153459', # Seed 123
        'ckpt_filename_template': 'ckpt_mix{ratio}_seed{seed}_iter{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
        'ratio': 5, # 此实验特有的参数
    },
    '60_embed': {
        'label': 'Embed Dim = 60 (Bottleneck)',
        'predicted_rank': 'N/A',
        'model_A_dir': 'out/composition_20250721_115705', # Seed 42
        'model_B_dir': 'out/composition_20250722_041942', # Seed 123
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
}

# 输出目录
OUTPUT_DIR = "sfs_hypothesis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 2. 核心函数 (Core Functions) - 无需修改
# ==============================================================================

def load_weight_matrix(exp_config, iteration, seed):
    """
    根据实验配置、迭代次数和种子加载单个权重矩阵。
    """
    model_dir = exp_config['model_A_dir'] if seed == exp_config['seed_A'] else exp_config['model_B_dir']

    # 使用字典格式化，只填充模板中存在的占位符
    filename = exp_config['ckpt_filename_template'].format(
        iter=iteration,
        seed=seed,
        ratio=exp_config.get('ratio')
    )
    
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

def analyze_difference_rank(W1, W2, rank_tolerance=1e-3):
    """
    计算两个权重矩阵之差的奇异值和秩。
    """
    W_diff = W1 - W2
    _, S_diff, _ = svd(W_diff, full_matrices=False)
    rank = np.sum(S_diff > rank_tolerance)
    return rank, S_diff

# ==============================================================================
# 3. 主执行流程 (Main Execution) - 无需修改
# ==============================================================================

def run_analysis_and_plot():
    """
    执行所有实验的分析并生成最终的可视化图表。
    """
    print("="*80)
    print("Starting SFS Hypothesis Validation (Final Corrected Version)")
    print("="*80)

    # 重新排序以获得更直观的图表：200 -> 120 -> 60
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
                
                observed_rank, s_values = analyze_difference_rank(W1, W2)
                
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
        ax.set_xlabel("Singular Value Index", fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        if isinstance(exp_info['predicted_rank'], int):
            ax.axvline(x=exp_info['predicted_rank'], color='red', linestyle='--', linewidth=2.5,
                       label=f"Predicted Rank = {exp_info['predicted_rank']}")
        
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Singular Value of (W_A - W_B)", fontsize=12)
    fig.suptitle("SFS Hypothesis Validation: Singular Value Spectrum of Weight Difference", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, "sfs_rank_validation_final_corrected.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✓ Analysis complete! Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis_and_plot()