# create_advanced_grafted_model.py

import torch
import numpy as np
import os
from scipy.linalg import svd
from scipy.spatial.transform import Rotation

def orthogonal_procrustes(A, B):
    """
    计算将A旋转到B的最佳旋转矩阵R。
    使得 ||A @ R - B||_F 最小。
    A, B: [n, k]
    """
    M = A.T @ B
    U, _, Vt = svd(M)
    R = U @ Vt
    return R

def get_model_weights(model_path, device='cpu'):
    """加载模型并提取W_out和W_in权重。"""
    ckpt = torch.load(model_path, map_location=device)
    # 假设模型结构与之前一致
    W_out = ckpt['model']['lm_head.weight'].cpu().numpy()
    W_in = ckpt['model']['transformer.wte.weight'].cpu().numpy()
    return W_out, W_in

def diagnose_alignment(name, M0, M20, R):
    """
    提供详细的对齐诊断。
    M0, M20: 待对齐的原始矩阵 (例如 U0, U20 或 V0, V20)
    R: 从 M0 到 M20 的旋转矩阵
    """
    M0_aligned = M0 @ R
    
    print(f"\n--- 诊断报告: {name} 空间 ---")
    
    # 1. 检查对齐后的Frobenius距离 (越小越好)
    # 这是衡量对齐后两个矩阵有多“接近”的真实物理距离
    diff_fro = np.linalg.norm(M0_aligned - M20, 'fro')
    print(f"[诊断1] 对齐后Frobenius距离 ||{name}0_aligned - {name}20||_F: {diff_fro:.6f}")

    # 2. 计算真实的子空间相似度 (SVD of M0.T @ M20)
    # 这是衡量两个空间内在重叠程度的稳健指标，不受完美旋转假象影响
    M_cross = M0.T @ M20
    singular_values = svd(M_cross, compute_uv=False)
    subspace_similarity = np.mean(singular_values)
    print(f"[诊断2] 真实子空间相似度 (奇异值均值): {subspace_similarity:.6f}")
    print(f"        (奇异值分布: Min={singular_values.min():.4f}, Max={singular_values.max():.4f})")

    # 3. 检查原始的“列向量对齐度”指标，以对比其与真实值的差异
    col_similarities = np.sum(M0_aligned * M20, axis=0)
    old_metric = np.mean(col_similarities)
    print(f"[诊断3] 旧对齐指标 (列向量内积均值): {old_metric:.6f}")
    if np.allclose(old_metric, 1.0):
        print("        ⚠️ 警告: 旧指标出现1.00的假象，已被[诊断2]的真实值替代。")

    # 4. 验证正交性
    is_orthogonal = np.allclose(M0_aligned.T @ M0_aligned, np.eye(M0_aligned.shape[1]))
    print(f"[诊断4] 对齐后矩阵 {name}0_aligned 是否保持正交: {is_orthogonal}")
    print("--- 诊断结束 ---")
    return subspace_similarity


def create_and_save_grafted_model(seed):
    """
    主函数：加载、分解、双重对齐、嫁接并保存模型。
    """
    print("="*80)
    print(f"🚀 开始处理 Seed = {seed}")
    print("="*80)

    # 定义模型路径
    path_0 = f'out/mix_0_d92_s{seed}/ckpt.pt'
    path_20 = f'out/mix_20_d92_s{seed}/ckpt.pt'
    output_dir = 'hybrid_models'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'hybrid_advanced_grafted_seed{seed}.pt')

    print(f"[*] 加载模型:\n    - 0% mix: {path_0}\n    - 20% mix: {path_20}")
    W_out_0, W_in_0 = get_model_weights(path_0)
    W_out_20, W_in_20 = get_model_weights(path_20)

    # 我们只对W_out进行操作，因为它是模型的核心计算层
    W0, W20 = W_out_0, W_out_20
    d_model = W0.shape[1] # 应该是92

    print(f"[*] 对权重矩阵 W_out (shape={W0.shape}) 进行SVD分解...")
    # 使用full_matrices=False确保得到经济尺寸的SVD
    U0, S0, V0t = svd(W0, full_matrices=False)
    U20, S20, V20t = svd(W20, full_matrices=False)
    V0, V20 = V0t.T, V20t.T

    print("[*] 开始进行双重空间对齐 (Procrustes Analysis)...")
    
    # 1. 对齐 V 空间 (输入/隐藏表示侧)
    print("\n[+] 对齐 V 空间...")
    R_v = orthogonal_procrustes(V0, V20)
    V0_aligned = V0 @ R_v
    v_similarity = diagnose_alignment("V", V0, V20, R_v)

    # 2. 对齐 U 空间 (输出/词汇表侧) - 这是关键创新！
    print("\n[+] 对齐 U 空间...")
    R_u = orthogonal_procrustes(U0, U20)
    U0_aligned = U0 @ R_u
    u_similarity = diagnose_alignment("U", U0, U20, R_u)

    print("\n[*] 使用 '双重对齐' 策略构建嫁接权重矩阵...")
    # 新策略: W_hybrid = U0_aligned @ S_20 @ V0_aligned.T
    W_grafted = U0_aligned @ np.diag(S20) @ V0_aligned.T
    print("    - W_grafted = U0_aligned @ diag(S20) @ V0_aligned.T  ...完成!")

    print("[*] 构建并保存新的嫁接模型...")
    # 加载0%模型的检查点作为模板
    ckpt_template = torch.load(path_0, map_location='cpu')
    
    # 将嫁接后的权重放回模板
    ckpt_template['model']['lm_head.weight'] = torch.from_numpy(W_grafted).float()
    
    # 重要：保持W_in和W_out同步
    # 在这个模型架构中，输入嵌入和输出投影是共享的（tied weights）
    # 因此，我们必须同时更新W_in (transformer.wte.weight)
    if 'transformer.wte.weight' in ckpt_template['model']:
        print("[*] 检测到权重绑定，同步更新 transformer.wte.weight...")
        ckpt_template['model']['transformer.wte.weight'] = torch.from_numpy(W_grafted).float()

    torch.save(ckpt_template, output_path)
    print(f"\n✅ 成功！高级嫁接模型已保存至: {output_path}")
    print(f"    - V空间真实相似度: {v_similarity:.4f}")
    print(f"    - U空间真实相似度: {u_similarity:.4f}")


if __name__ == '__main__':
    seeds = [42, 123, 456]
    for seed in seeds:
        create_and_save_grafted_model(seed)