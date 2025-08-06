#!/usr/bin/env python3
"""
interseed_stability_checker.py

本脚本使用修正后的、更严格的“基向量对齐度”方法来验证训练的收敛稳定性。

核心功能:
1.  针对同一实验条件（如 d=92, 0% mix），找出所有不同的随机种子。
2.  对这些种子进行两两配对，例如 (42 vs 123), (42 vs 456), (123 vs 456)。
3.  计算每一对模型在多个训练节点上 V-Space 和 U-Space 的基向量平均对齐度。

这个脚本将提供最强有力的证据，来证明您的实验环境是否“干净”和“稳定”。
我们期望在训练后期，这个值能非常接近 1.0。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import itertools

# --- 全局配置 ---
SEEDS_TO_CHECK = [42, 123, 456]
ITERATIONS_TO_CHECK = list(range(5000, 50001, 5000))
D_MODEL = 92

# --- 核心工具函数 ---

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """根据文件结构获取检查点路径。"""
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs: return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    if not os.path.exists(path): return None
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载指定模型的 lm_head 权重矩阵。"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None: return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        W = state_dict['lm_head.weight'].float().numpy()
        del checkpoint, state_dict
        return W
    except Exception as e:
        print(f"  [错误] 加载文件 '{path}' 失败: {e}")
        return None

# --- 【正确且严格的】相似度计算函数 ---
def compute_basis_alignment(W1, W2):
    """
    计算两个权重矩阵的基向量对齐度（平均余弦相似度）。
    这能正确地衡量空间的旋转程度。
    """
    if W1 is None or W2 is None: return None
    
    # SVD分解。Vt的每一行是一个V向量。
    U1, _, Vt1 = svd(W1, full_matrices=False)
    U2, _, Vt2 = svd(W2, full_matrices=False)
    
    # 计算V向量的平均对齐度
    # 对应行做点积，取绝对值（因为v和-v是等价的基向量），然后求平均
    v_cos_sims = np.abs(np.sum(Vt1 * Vt2, axis=1))
    v_alignment = np.mean(v_cos_sims)
    
    # 计算U向量的平均对齐度
    # U的每一列是一个U向量，所以需要先转置
    u_cos_sims = np.abs(np.sum(U1.T * U2.T, axis=1))
    u_alignment = np.mean(u_cos_sims)
    
    return {'v_alignment': v_alignment, 'u_alignment': u_alignment}

# --- 主执行流程 ---

def main():
    """
    主函数，定义并执行跨随机种子的稳定性分析。
    """
    # =========================================================================
    # --- 在这里配置您要运行的所有实验组 ---
    # =========================================================================
    CONFIGURATIONS_TO_RUN = [
        # 实验组 1: d=92, mix_ratio=0%。
        # 脚本将自动进行 3 组比较: (42 vs 123), (42 vs 456), (123 vs 456)
        { 'ratio': 0 },
        
        # 实验组 2: d=92, mix_ratio=20%。
        # 脚本同样会进行 3 组比较
        { 'ratio': 20 },
    ]

    print("="*80)
    print("      开始执行【严格的】跨随机种子收敛稳定性分析 (d=92 only)")
    print("="*80)

    # 遍历所有配置好的实验组
    for config in CONFIGURATIONS_TO_RUN:
        ratio = config['ratio']

        print(f"\n\n{'#'*70}")
        print(f"  分析实验组: d_model={D_MODEL}, mix_ratio={ratio}%")
        print(f"  种子池: {SEEDS_TO_CHECK}")
        print(f"{'#'*70}")

        seed_pairs = list(itertools.combinations(SEEDS_TO_CHECK, 2))
        print(f"\n将要执行 {len(seed_pairs)} 组比较: {seed_pairs}")

        for seed1, seed2 in seed_pairs:
            print(f"\n--- [比较开始] Seed {seed1} vs. Seed {seed2} ---")
            
            results_timeline = []
            for iteration in ITERATIONS_TO_CHECK:
                print(f"  正在检查迭代: {iteration}")
                W1 = load_weight_matrix(D_MODEL, ratio, seed1, iteration)
                W2 = load_weight_matrix(D_MODEL, ratio, seed2, iteration)

                if W1 is not None and W2 is not None:
                    metrics = compute_basis_alignment(W1, W2)
                    if metrics:
                        results_timeline.append({
                            'iteration': iteration,
                            **metrics
                        })
                else:
                    print(f"  跳过迭代 {iteration}，因为一个或两个检查点文件未找到。")

            if not results_timeline:
                print("\n未能成功分析任何检查点，请检查文件路径和迭代次数。")
                continue

            print("\n" + "="*70)
            print(f"  基向量对齐度总结: Seed {seed1} vs. Seed {seed2} (mix={ratio}%)")
            print("="*70)
            print(f"{'Iteration':<15} | {'V-Space Avg. Cosine Sim.':<30} | {'U-Space Avg. Cosine Sim.':<30}")
            print("-" * 70)
            for result in results_timeline:
                iter_str = f"{result['iteration']}"
                v_align_str = f"{result['v_alignment']:.8f}"
                u_align_str = f"{result['u_alignment']:.8f}"
                print(f"{iter_str:<15} | {v_align_str:<30} | {u_align_str:<30}")
            print("-" * 70)

    print("\n\n" + "#"*80)
    print("🎉🎉🎉 全部实验组分析完毕！ 🎉🎉🎉")
    print("#"*80)
    print("\n结论解读:")
    print("  - 这里的数值代表了基向量的平均对齐程度。")
    print("  - 如果在训练后期（例如50k次迭代），这些值能够稳定地**非常接近1.0**（如 > 0.999），")
    print("    那么就强有力地证明了：不同的随机初始化最终收敛到了**同一个解**。")
    print("  - 这证明了您的实验环境是“干净”且“稳定”的，结果是可复现的！")

if __name__ == "__main__":
    main()