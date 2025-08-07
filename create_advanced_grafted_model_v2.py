# create_advanced_grafted_model_v2.py

import torch
import numpy as np
import glob
import os
import argparse
from scipy.linalg import svd, orthogonal_procrustes

# --- 核心辅助函数 (包含您的路径查找器和我的诊断工具) ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """
    (来自您的脚本) 自动查找给定ratio和seed的最新一次训练的最终模型路径。
    """
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
    
    latest_dir = sorted(dirs)[-1]
    # 尝试多种可能的迭代数命名，以增加鲁棒性
    possible_iterations = [50000, 49999] 
    for iteration in possible_iterations:
        expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
        path = os.path.join(latest_dir, expected_filename)
        if os.path.exists(path):
            print(f"  > 定位到模型: {path}")
            return path
            
    # 如果都找不到，再尝试不带迭代数的通用名称
    generic_path = os.path.join(latest_dir, "ckpt.pt")
    if os.path.exists(generic_path):
        print(f"  > 定位到模型: {generic_path}")
        return generic_path

    raise FileNotFoundError(f"在目录 {latest_dir} 中未找到任何可识别的最终 checkpoint 文件。")

def diagnose_alignment(name, M0, M20, R):
    """
    (新功能) 提供详细的对齐诊断，揭示真实相似度。
    """
    M0_aligned = M0 @ R
    print(f"\n--- 诊断报告: {name} 空间 ---")
    
    diff_fro = np.linalg.norm(M0_aligned - M20, 'fro')
    print(f"[诊断1] 对齐后Frobenius距离 ||{name}0_aligned - {name}20||_F: {diff_fro:.6f} (越小越好)")

    M_cross = M0.T @ M20
    singular_values = svd(M_cross, compute_uv=False)
    subspace_similarity = np.mean(singular_values)
    print(f"[诊断2] 真实子空间相似度 (奇异值均值): {subspace_similarity:.6f} (这才是真实对齐分)")
    
    old_metric = np.mean(np.sum(M0_aligned * M20, axis=0))
    print(f"[诊断3] 旧对齐指标 (用于对比): {old_metric:.6f}")
    if np.allclose(old_metric, 1.0) and not np.allclose(subspace_similarity, 1.0):
        print("        ✅ 成功识别并绕开了1.00的假象！")
    print("--- 诊断结束 ---")
    return subspace_similarity

def create_advanced_grafted_model(path_0, path_20):
    """
    执行“双重对齐嫁接”的核心函数。
    """
    # 加载模型
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float().numpy()

    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float().numpy()

    # SVD分解
    print("  > 正在对权重矩阵 W_out 进行SVD分解...")
    U0, S0, V0t = svd(W0, full_matrices=False)
    U20, S20, V20t = svd(W20, full_matrices=False)
    V0, V20 = V0t.T, V20t.T

    # 双重对齐
    print("  > 正在执行双重空间对齐 (Procrustes Analysis)...")
    R_v = orthogonal_procrustes(V0, V20)
    v_similarity = diagnose_alignment("V", V0, V20, R_v)
    
    R_u = orthogonal_procrustes(U0, U20)
    u_similarity = diagnose_alignment("U", U0, U20, R_u)

    # 对齐后的空间
    V0_aligned = V0 @ R_v
    U0_aligned = U0 @ R_u
    
    # 重建混合权重矩阵 (使用新策略)
    print("\n  > 正在使用 '双重对齐' 策略重建混合权重矩阵...")
    W_hybrid_np = U0_aligned @ np.diag(S20) @ (V0_aligned.T)
    W_hybrid = torch.from_numpy(W_hybrid_np).float()
    print("    - W_grafted = U0_aligned @ diag(S20) @ V0_aligned.T  ...完成!")

    # 创建新的模型 state_dict
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    if 'transformer.wte.weight' in state_hybrid:
        print("  > 检测到权重绑定，同步更新 transformer.wte.weight...")
        state_hybrid['transformer.wte.weight'] = W_hybrid

    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)

    return ckpt_hybrid, v_similarity, u_similarity

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行高级“双重对齐嫁接”实验。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42, 123, 456)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed

    print("="*60)
    print("     🚀 开始执行高级“双重对齐嫁接”实验 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}")
    print("="*60)
    
    try:
        print("\n步骤 1: 定位基准模型文件...")
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)

        print("\n步骤 2: 开始执行双重对齐嫁接...")
        hybrid_ckpt, v_sim, u_sim = create_advanced_grafted_model(path_0, path_20)

        output_dir = "hybrid_models"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"hybrid_advanced_grafted_seed{SEED_TO_TEST}.pt")
        torch.save(hybrid_ckpt, output_filename)
        
        print("\n" + "="*60)
        print("🎉🎉🎉 高级嫁接模型已成功生成！ 🎉🎉🎉")
        print("="*60)
        print(f"模型已保存至: {output_filename}")
        print(f"真实 V 空间相似度: {v_sim:.4f}")
        print(f"真实 U 空间相似度: {u_sim:.4f}")
        print("\n下一步：请运行评估脚本来测试其性能。")

    except FileNotFoundError as e:
        print(f"\n🚨 致命错误: {e}")
        print("请检查您的 'out_d92' 目录结构和文件名是否正确。")
        exit(1)