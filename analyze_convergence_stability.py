#!/usr/bin/env python3
"""
analyze_convergence_stability.py

本脚本用于对一系列模型训练结果进行收敛稳定性分析。

核心功能:
1.  针对同一实验条件（如相同的混合比例和嵌入维度），自动找出所有不同的随机种子。
2.  对这些种子进行两两配对，生成所有可能的比较组合。
3.  比较每一对模型在多个训练节点（检查点）上的 V-Space 和 U-Space 相似度。
4.  智能处理带有时间戳的输出目录，确保总是分析最新的实验结果。

这能清晰地展示出在相同条件下，不同随机初始化是否能稳定地收敛到相同的解结构。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import itertools

# --- 在这里配置您要执行的检查 ---
# 您可以自由修改这个列表，加入任何您想检查的迭代次数
ITERATIONS_TO_CHECK = [10000, 20000, 30000, 40000, 50000]

# --- 核心函数 ---

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """
    根据文件结构获取检查点路径。
    该函数能够智能处理带有时间戳的目录。
    """
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    
    if not dirs:
        print(f"  [警告] 找不到目录，匹配模式为 '{pattern}'")
        return None
    
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    
    if not os.path.exists(path):
        print(f"  [警告] 检查点文件不存在于 '{path}'")
        return None
        
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载指定模型的 lm_head 权重矩阵。"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None:
        return None
        
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    W = state_dict['lm_head.weight'].float().numpy()
    del checkpoint, state_dict
    return W

def compute_subspace_similarity(W1, W2, k):
    """计算两个权重矩阵的U和V子空间相似度。"""
    if W1 is None or W2 is None:
        return None
        
    U1, _, Vt1 = svd(W1, full_matrices=False)
    U2, _, Vt2 = svd(W2, full_matrices=False)

    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T
    U1_k, U2_k = U1[:, :k], U2[:, :k]

    overlap_v = V1_k.T @ V2_k
    similarity_v = np.mean(np.clip(svd(overlap_v, compute_uv=False), 0, 1))

    overlap_u = U1_k.T @ U2_k
    similarity_u = np.mean(np.clip(svd(overlap_u, compute_uv=False), 0, 1))

    return {'v_similarity': similarity_v, 'u_similarity': similarity_u}

# --- 主执行流程 ---

def main():
    """
    主函数，定义并执行稳定性分析。
    您只需要修改下面的 CONFIGURATIONS_TO_RUN 列表即可。
    """
    # =========================================================================
    # --- 在这里配置您要运行的所有实验组 ---
    # 【已按您的最终要求修改】
    # 现在两个实验组都包含三个种子: 42, 123, 456
    # =========================================================================
    CONFIGURATIONS_TO_RUN = [
        # 实验组 1: d_model=92, mix_ratio=20%。
        # 脚本将自动进行 3 组比较: (42 vs 123), (42 vs 456), (123 vs 456)
        {
            'd_model': 92,
            'ratio': 20,
            'seeds': [42, 123, 456], 
            'k': 92
        },
        
        # 实验组 2: d_model=92, mix_ratio=0%。
        # 脚本同样会进行 3 组比较: (42 vs 123), (42 vs 456), (123 vs 456)
        {
            'd_model': 92,
            'ratio': 0,
            'seeds': [42, 123, 456],
            'k': 92
        },
    ]

    print("="*80)
    print("      开始执行多重随机种子收敛稳定性分析 (d_model=92 only)")
    print("="*80)

    # 遍历所有配置好的实验组
    for config in CONFIGURATIONS_TO_RUN:
        d_model = config['d_model']
        ratio = config['ratio']
        seeds = config['seeds']
        k_value = config['k']

        print(f"\n\n{'#'*70}")
        print(f"  分析实验组: d_model={d_model}, mix_ratio={ratio}%, k={k_value}")
        print(f"  种子池: {seeds}")
        print(f"{'#'*70}")

        if len(seeds) < 2:
            print("\n种子池中少于2个种子，无法进行比较。跳过此实验组。")
            continue

        seed_pairs = list(itertools.combinations(seeds, 2))
        print(f"\n将要执行 {len(seed_pairs)} 组比较: {seed_pairs}")

        for seed1, seed2 in seed_pairs:
            print(f"\n--- [比较开始] Seed {seed1} vs. Seed {seed2} ---")
            
            results_timeline = []
            for iteration in ITERATIONS_TO_CHECK:
                print(f"  正在检查迭代: {iteration}")
                W1 = load_weight_matrix(d_model, ratio, seed1, iteration)
                W2 = load_weight_matrix(d_model, ratio, seed2, iteration)

                if W1 is not None and W2 is not None:
                    metrics = compute_subspace_similarity(W1, W2, k=k_value)
                    if metrics:
                        results_timeline.append({
                            'iteration': iteration,
                            'v_similarity': metrics['v_similarity'],
                            'u_similarity': metrics['u_similarity']
                        })
                else:
                    print(f"  跳过迭代 {iteration}，因为一个或两个检查点文件未找到。")

            if not results_timeline:
                print("\n未能成功分析任何检查点，请检查文件路径和迭代次数。")
                continue

            print("\n" + "="*60)
            print(f"  稳定性分析总结: Seed {seed1} vs. Seed {seed2}")
            print("="*60)
            print(f"{'Iteration':<15} | {'V-Space (Thought) Sim.':<25} | {'U-Space (Language) Sim.':<25}")
            print("-" * 60)
            for result in results_timeline:
                iter_str = f"{result['iteration']}"
                v_sim_str = f"{result['v_similarity']:.8f}"
                u_sim_str = f"{result['u_similarity']:.8f}"
                print(f"{iter_str:<15} | {v_sim_str:<25} | {u_sim_str:<25}")
            print("="*60)

    print("\n\n" + "#"*80)
    print("🎉🎉🎉 全部实验组分析完毕！ 🎉🎉🎉")
    print("#"*80)
    print("\n结论解读:")
    print("  - 您可以看到，随着训练的进行（迭代次数增加），两个相似度值都应稳步接近 1.0。")
    print("  - 在训练后期，这些值如果能稳定在 0.999 以上，则强有力地证明了模型收敛的稳定性。")

if __name__ == "__main__":
    main()