#!/usr/bin/env python3
"""
sfs_hypothesis_validator_all_dims.py

该脚本用于验证“确定性知识子空间 (DKS) + 随机波动子空间 (SFS)”假说。
此版本整合了60, 120, 150, 200, 300维度的所有实验，以全面测试
模型在不同“表示预算”下的知识编码策略。

运行: python sfs_hypothesis_validator_all_dims.py
"""
import os
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 配置 (CONFIG) - 已添加150维和300维实验
# ==============================================================================

# 定义要进行比较的检查点迭代次数
ITERATIONS_TO_TEST = [5000, 30000, 50000]

# 定义每个实验的参数和模型路径信息
CONFIG = {
    # <--- 新增实验：300维
    '300_embed': {
        'label': 'Embed Dim = 300',
        'predicted_rank': 300 - 109, # 预测采用“奢侈”的109维DKS策略
        'model_A_dir': 'out/composition_embed300_seed42_20250722_150525',
        'model_B_dir': 'out/composition_embed300_seed123_20250722_153809',
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
    '200_embed': {
        'label': 'Embed Dim = 200',
        'predicted_rank': 200 - 109, # 观测结果表明其采用109维DKS策略
        'model_A_dir': 'out/composition_20250721_044634',
        'model_B_dir': 'out/composition_20250722_044016',
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
    # <--- 新增实验：150维
    '150_embed': {
        'label': 'Embed Dim = 150',
        'predicted_rank': 150 - 109, # 预测其开始采用109维DKS策略
        'model_A_dir': 'out/composition_embed150_seed42_20250722_141518',
        'model_B_dir': 'out/composition_embed150_seed123_20250722_144021',
        'ckpt_filename_template': 'ckpt_{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
    },
    '120_embed': {
        'label': 'Embed Dim = 120',
        'predicted_rank': 120 - 92, # 观测结果表明其采用“紧凑”的92维DKS策略
        'model_A_dir': 'out/composition_mix5_seed42_20250705_151206',
        'model_B_dir': 'out/composition_mix5_seed123_20250705_153459',
        'ckpt_filename_template': 'ckpt_mix{ratio}_seed{seed}_iter{iter}.pt',
        'seed_A': 42,
        'seed_B': 123,
        'ratio': 5,
    },
    '60_embed': {
        'label': 'Embed Dim = 60 (Bottleneck)',
        'predicted_rank': 'N/A', # 预算严重不足，预测SFS为0
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
# 2. 核心函数 (Core Functions) - 无需修改
# ==============================================================================

def load_weight_matrix(exp_config, iteration, seed):
    """根据配置加载权重矩阵"""
    model_dir = exp_config['model_A_dir'] if seed == exp_config['seed_A'] else exp_config['model_B_dir']
    # 使用字典的.get()方法安全地获取ratio，如果不存在则为None
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
    """计算子空间差异的秩（V2中正交于V1空间的部分）"""
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    V1 = Vt1.T
    V2 = Vt2.T
    V2_proj_on_V1 = V1 @ (V1.T @ V2)
    V_diff_orthogonal = V2 - V2_proj_on_V1
    observed_rank = np.linalg.matrix_rank(V_diff_orthogonal, tol=rank_tolerance)
    _, s_values, _ = svd(V_diff_orthogonal, full_matrices=False)
    return observed_rank, s_values

# ==============================================================================
# 3. 主执行流程 (Main Execution) - 已更新以适应5个实验
# ==============================================================================

def run_analysis_and_plot():
    """执行所有实验的分析并生成最终的可视化图表。"""
    print("="*80)
    print("Starting SFS Hypothesis Validation (All 5 Dimensions)")
    print("="*80)

    # <--- 更新：按维度从小到大排序，以获得更直观的图表
    exp_order = ['60_embed', '120_embed', '150_embed', '200_embed', '300_embed']
    num_experiments = len(exp_order)
    
    # <--- 更新：使用2x3的网格布局，更适合展示5张图
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes_flat = axes.flatten() # 将2D坐标轴数组展平为1D，方便遍历

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ITERATIONS_TO_TEST)))

    for i, exp_key in enumerate(exp_order):
        exp_info = CONFIG[exp_key]
        ax = axes_flat[i] # <--- 更新：使用展平后的坐标轴
        print(f"\n--- Running Experiment: {exp_key} ---")
        
        for j, iteration in enumerate(ITERATIONS_TO_TEST):
            try:
                W1 = load_weight_matrix(exp_info, iteration, exp_info['seed_A'])
                W2 = load_weight_matrix(exp_info, iteration, exp_info['seed_B'])
                
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

    # <--- 更新：为第一列的两个图设置Y轴标签
    axes[0, 0].set_ylabel("Singular Value of Orthogonal Component", fontsize=12)
    axes[1, 0].set_ylabel("Singular Value of Orthogonal Component", fontsize=12)

    # <--- 更新：隐藏多余的第6个子图
    fig.delaxes(axes_flat[-1])

    fig.suptitle("SFS Hypothesis Validation: Singular Value Spectrum Across Embedding Dimensions", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, "sfs_validation_all_5_dims.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✓ Analysis complete! Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis_and_plot()