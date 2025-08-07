#!/usr/bin/env python3
"""
procrustes_analysis.py

本脚本使用普氏分析（Procrustes Analysis）来对齐并比较不同模型的解空间。

核心功能:
1.  加载任意两个指定模型的 lm_head 权重。
2.  提取它们的前 k 个 V-Space 向量 (V1_k, V2_k)。
3.  计算在对齐前的原始子空间相似度。
4.  使用 scipy.linalg.orthogonal_procrustes 计算最佳旋转矩阵 R，
    该矩阵能将 V2_k 旋转到与 V1_k 尽可能对齐。
5.  计算对齐后的新子空间相似度。
6.  清晰地展示“对齐前”与“对齐后”的相似度变化，从而量化旋转对
    模型解空间的影响，并揭示真实的结构性差异。

使用方法:
- 直接运行 `python procrustes_analysis.py` 即可。
- 可在主函数中自由修改 `COMPARISON_CONFIGS` 列表来定义新的比较任务。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
from collections import defaultdict

# --- 全局配置 ---
# 您可以自由修改这个列表，加入任何您想检查的迭代次数
ITERATION_TO_ANALYZE = 50000 
K_VALUE = 60 # 使用有效秩 k=60

# --- 核心工具函数 (从您之前的脚本中复用) ---

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """根据文件结构获取检查点路径。"""
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    if not os.path.exists(path):
        return None
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载指定模型的 lm_head 权重矩阵。"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None: 
        print(f"  [警告] 找不到模型: d={d_model}, ratio={ratio}, seed={seed}, iter={iteration}")
        return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        W = state_dict['lm_head.weight'].float().numpy()
        del checkpoint, state_dict
        return W
    except Exception as e:
        print(f"  [错误] 加载文件 '{path}' 失败: {e}")
        return None

def calculate_similarity(V1, V2):
    """一个通用的子空间相似度计算函数。"""
    # V1 和 V2 已经是 (d, k) 形状的子空间基
    overlap = V1.T @ V2
    # SVD 的奇异值即为两个子空间主角度的余弦值
    cos_theta = np.clip(svd(overlap, compute_uv=False), 0, 1)
    return np.mean(cos_theta)

# --- 主分析流程 ---

def run_single_comparison(config):
    """对单个配置进行完整的普氏分析。"""
    d_model = config['d_model']
    k = config['k']
    iter_val = config['iter']

    # --- 1. 加载权重 ---
    W1 = load_weight_matrix(d_model, config['ratio1'], config['seed1'], iter_val)
    W2 = load_weight_matrix(d_model, config['ratio2'], config['seed2'], iter_val)

    if W1 is None or W2 is None:
        print("-------> 跳过此项比较，因为无法加载一个或两个模型。\n")
        return

    # --- 2. SVD 并提取子空间 ---
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    
    # V 矩阵的形状是 (d_model, d_model)，每一列是一个基向量
    # 我们取前 k 个基向量构成子空间
    V1_k = Vt1[:k, :].T
    V2_k = Vt2[:k, :].T

    # --- 3. 计算对齐前的相似度 ---
    similarity_before = calculate_similarity(V1_k, V2_k)

    # --- 4. 执行普氏分析，找到最佳旋转矩阵 R ---
    # 我们要找到 R，使得 V2_k @ R 尽可能接近 V1_k
    R, _ = orthogonal_procrustes(V2_k, V1_k)
    
    # --- 5. 对 V2 进行旋转对齐 ---
    V2_k_aligned = V2_k @ R

    # --- 6. 计算对齐后的相似度 ---
    similarity_after = calculate_similarity(V1_k, V2_k_aligned)

    # --- 7. 打印结果 ---
    print(f"  模型 1: mix={config['ratio1']}%, seed={config['seed1']}")
    print(f"  模型 2: mix={config['ratio2']}%, seed={config['seed2']}")
    print("-" * 60)
    print(f"  V-Space 相似度 (对齐前): {similarity_before:.8f}")
    print(f"  V-Space 相似度 (对齐后): {similarity_after:.8f}")
    print(f"  对齐提升: {(similarity_after - similarity_before):.8f}")
    print("-" * 60 + "\n")


def main():
    """
    主函数，定义并执行所有比较任务。
    """
    # =========================================================================
    # --- 在这里配置您要运行的所有比较任务 ---
    # =========================================================================
    COMPARISON_CONFIGS = [
        # --- 场景 A: 控制实验 (相同 mix, 不同 seed) ---
        # 目标: 验证旋转对齐可以将相似度从 ~75% 恢复到接近 100%
        {
            'description': '控制组1: 0% mix, 不同种子',
            'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE,
            'ratio1': 0, 'seed1': 42,
            'ratio2': 0, 'seed2': 123
        },
        {
            'description': '控制组2: 20% mix, 不同种子',
            'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE,
            'ratio1': 20, 'seed1': 42,
            'ratio2': 20, 'seed2': 123
        },
        
        # --- 场景 B: 核心实验 (不同 mix, 相同 seed) ---
        # 目标: 探究在消除了旋转影响后，0%和20%模型的真实结构差异有多大
        {
            'description': '实验组1: 相同种子, 0% vs 20% mix',
            'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE,
            'ratio1': 0, 'seed1': 42,
            'ratio2': 20, 'seed2': 42
        },
        {
            'description': '实验组2: 相同种子, 0% vs 20% mix',
            'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE,
            'ratio1': 0, 'seed1': 123,
            'ratio2': 20, 'seed2': 123
        },
    ]

    print("="*80)
    print("      开始执行基于普氏分析的解空间对齐与比较")
    print(f"      (k={K_VALUE}, iteration={ITERATION_TO_ANALYZE})")
    print("="*80)

    for config in COMPARISON_CONFIGS:
        print(f"\n>>> 正在执行: {config['description']}")
        run_single_comparison(config)

    print("\n" + "#"*80)
    print("🎉🎉🎉 全部分析完毕！ 🎉🎉🎉")
    print("#"*80)
    print("\n结论解读:")
    print("  - 对于【控制组】(相同mix, 不同seed):")
    print("    如果“对齐后”相似度飙升至接近 1.0，则强力证明了随机种子的影响主要是全局旋转，")
    print("    并且我们的对齐方法是有效的。")
    print("  - 对于【实验组】(不同mix, 相同seed):")
    print("    “对齐后”的相似度是关键！它揭示了在排除了旋转后，0%和20%模型之间")
    print("    还剩下多少真实的、不可旋转的几何结构差异。这个数值越高，说明它们的")
    print("    核心空间结构越相似。")


if __name__ == "__main__":
    main()