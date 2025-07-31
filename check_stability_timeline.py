#!/usr/bin/env python3
"""
check_stability_timeline.py

本脚本用于分析模型稳定性随训练时间的变化。
它会比较两个不同随机种子（例如 42 vs. 123）训练出的模型，
在多个不同训练节点（例如 10000, 20000, ..., 50000次迭代）
的 V-Space 和 U-Space 相似度。

这能清晰地展示出两个模型是否以及何时收敛到相同的结构。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd

# --- 在这里配置您要执行的检查 ---
# 您可以自由修改这个列表，加入任何您想检查的迭代次数
ITERATIONS_TO_CHECK = [10000, 20000, 30000, 40000, 50000]

# --- 核心函数 (与上一版基本一致，但更加稳健) ---

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """根据文件结构获取检查点路径。"""
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        # 如果找不到目录，直接返回 None，让调用者处理
        print(f"警告: 找不到目录，匹配模式为 '{pattern}'")
        return None
    
    selected_dir = dirs[-1]
    path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(path):
        # 如果找不到文件，也返回 None
        print(f"警告: 检查点文件不存在于 '{path}'")
        return None
        
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载指定模型的 lm_head 权重矩阵。"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None:
        return None # 如果路径不存在，直接返回
        
    print(f"  > 正在加载: {path}")
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
    overlap_v = V1_k.T @ V2_k
    similarity_v = np.mean(np.clip(svd(overlap_v, compute_uv=False), 0, 1))

    U1_k, U2_k = U1[:, :k], U2[:, :k]
    overlap_u = U1_k.T @ U2_k
    similarity_u = np.mean(np.clip(svd(overlap_u, compute_uv=False), 0, 1))

    return {'v_similarity': similarity_v, 'u_similarity': similarity_u}

# --- 主执行流程 (已升级为循环检查和总结报告) ---

def main():
    """
    主函数，定义并执行稳定性时间线分析。
    """
    # --- 模型基本配置 ---
    D_MODEL = 92
    RATIO = 0
    K_VALUE = 92
    SEED1 = 42
    SEED2 = 123
    
    print("="*80)
    print("开始执行稳定性时间线分析 (Stability Timeline Analysis)")
    print(f"配置: d_model={D_MODEL}, mix_ratio={RATIO}%, k={K_VALUE}")
    print(f"目标: 比较 Seed {SEED1} vs. Seed {SEED2} 在多个检查点的表现")
    print(f"待检查节点: {ITERATIONS_TO_CHECK}")
    print("="*80)

    results_timeline = []

    # 遍历所有指定的检查点
    for iteration in ITERATIONS_TO_CHECK:
        print(f"\n--- 正在检查迭代: {iteration} ---")
        
        # 加载两个模型的权重
        W1 = load_weight_matrix(D_MODEL, RATIO, SEED1, iteration)
        W2 = load_weight_matrix(D_MODEL, RATIO, SEED2, iteration)

        # 如果两个文件都成功加载，则进行计算
        if W1 is not None and W2 is not None:
            metrics = compute_subspace_similarity(W1, W2, k=K_VALUE)
            if metrics:
                results_timeline.append({
                    'iteration': iteration,
                    'v_similarity': metrics['v_similarity'],
                    'u_similarity': metrics['u_similarity']
                })
                print(f"  分析完成。V-Sim: {metrics['v_similarity']:.6f}, U-Sim: {metrics['u_similarity']:.6f}")
        else:
            print("  跳过此迭代，因为一个或两个检查点文件未找到。")

    # --- 打印最终的总结报告 ---
    if not results_timeline:
        print("\n未能成功分析任何检查点，请检查文件路径和迭代次数是否正确。")
        return

    print("\n\n" + "="*60)
    print("              稳定性分析总结报告")
    print("="*60)
    # 打印表头，使用f-string进行格式化对齐
    print(f"{'Iteration':<15} | {'V-Space (Thought) Sim.':<25} | {'U-Space (Language) Sim.':<25}")
    print("-" * 60)
    
    # 打印每一行的数据
    for result in results_timeline:
        iter_str = f"{result['iteration']}"
        v_sim_str = f"{result['v_similarity']:.8f}"
        u_sim_str = f"{result['u_similarity']:.8f}"
        print(f"{iter_str:<15} | {v_sim_str:<25} | {u_sim_str:<25}")
        
    print("="*60)
    
    print("\n结论解读:")
    print("  - 您可以看到，随着训练的进行（迭代次数增加），两个相似度值都应稳步接近 1.0。")
    print("  - 在训练后期，这些值如果能稳定在 0.999 以上，则强有力地证明了模型收敛的稳定性。")


if __name__ == "__main__":
    main()