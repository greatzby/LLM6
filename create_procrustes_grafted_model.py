#!/usr/bin/env python3
"""
create_procrustes_grafted_model.py

本脚本基于普氏分析的结果，执行最先进的“普氏对齐嫁接”。
它取代了旧的、基于维度匹配的嫁接方法。

核心逻辑:
1. 加载 0% 和 20% 混合比例的模型。
2. 对二者的 lm_head.weight 进行SVD分解。
3. 使用普氏分析找到将 0% 模型的V空间旋转对齐到 20% 模型V空间的最佳旋转矩阵 R。
4. 使用 0% 模型的U空间、20% 模型的S奇异值、以及对齐后的0%模型的V空间，重建混合权重。
5. 保存这个理论上最优的嫁接模型。
"""

import torch
import numpy as np
import glob
import os
import argparse
from scipy.linalg import svd, orthogonal_procrustes

# --- 辅助函数 ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """自动查找给定ratio和seed的最新一次训练的最终模型路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
    
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")
    
    print(f"  > 定位到模型: {path}")
    return path

def create_procrustes_grafted_model(path_0, path_20):
    """
    执行普氏对齐嫁接的核心函数。
    """
    # 加载模型
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float().numpy()

    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float().numpy()

    # SVD分解 (使用scipy以便与之前的分析保持一致)
    U0, S0, V0t = svd(W0, full_matrices=False)
    U20, S20, V20t = svd(W20, full_matrices=False)
    V0 = V0t.T
    V20 = V20t.T

    # 普氏分析：找到将 V0 旋转到 V20 的最佳旋转矩阵 R
    # 注意：procrustes(A, B) 找到 R 使得 ||AR - B|| 最小
    print("  > 正在执行普氏分析以找到最佳旋转矩阵 R...")
    R, _ = orthogonal_procrustes(V0, V20)
    
    # 验证对齐效果 (可选，但建议)
    V0_aligned = V0 @ R
    alignment_score = np.mean(np.sum(V0_aligned * V20, axis=0))
    print(f"  > 空间对齐得分 (越高越好，理论上限1.0): {alignment_score:.4f}")

    # 重建混合权重矩阵
    # W_hybrid = U0 @ diag(S20) @ (V0 @ R).T
    # (V0 @ R).T = R.T @ V0.T = R.T @ V0t
    print("  > 正在重建混合权重矩阵...")
    W_hybrid_np = U0 @ np.diag(S20) @ (V0_aligned.T)
    W_hybrid = torch.from_numpy(W_hybrid_np)

    # 创建新的模型 state_dict 和 checkpoint
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    # 如果权重绑定，也需要更新 wte.weight
    if 'transformer.wte.weight' in state_hybrid:
        state_hybrid['transformer.wte.weight'] = W_hybrid

    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)

    return ckpt_hybrid

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行普氏对齐嫁接实验。")
    parser.add_argument('--seed', type=int, default=42, help='要操作的随机种子 (例如: 42, 123, 456)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed

    print("="*60)
    print("     🚀 开始执行“普氏对齐嫁接”实验 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}")
    print("="*60)
    
    # 1. 定位模型文件
    print("\n步骤 1: 定位基准模型文件...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)

    # 2. 执行嫁接
    print("\n步骤 2: 开始执行普氏对齐嫁接...")
    hybrid_ckpt = create_procrustes_grafted_model(path_0, path_20)

    # 3. 保存混合模型
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"hybrid_procrustes_grafted_seed{SEED_TO_TEST}.pt")
    torch.save(hybrid_ckpt, output_filename)
    print(f"\n✅ 成功保存嫁接模型到: {output_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 嫁接模型已成功生成！ 🎉🎉🎉")
    print("="*60)
    print("下一步：请运行评估脚本来测试其性能。")