# ----------- 文件: graft_advanced.py (v3.1 - 集成输出控制和alpha命名) -----------
# 核心修正：
# 1. 新增 --output_file 参数，允许外部脚本指定精确的输出路径。
# 2. 修改默认文件名生成逻辑，使其包含 alpha 值，避免实验结果被覆盖。
# =================================================================

import torch
import os
import argparse
import glob
import numpy as np
from scipy.linalg import orthogonal_procrustes, fractional_matrix_power

# 假设 model.py 在同一目录下
from model import GPTConfig, GPT

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于查找并验证模型检查点的路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")
    print(f"  > 定位到模型: {path}")
    return path

def create_graft_transplant(path_0, path_20, lam, seed, k_val, mode, alpha_val, alpha_clip, no_procrustes, output_file_path):
    """
    执行精准嫁接的核心函数，支持多种模式和控制。
    """
    print("[*] 正在加载原始模型权重...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')

    state_0 = ckpt_0['model']
    state_20 = ckpt_20['model']
    
    W0 = state_0['lm_head.weight'].float().cpu().numpy()
    
    if k_val == 0:
        print("[*] k=0: 触发“无操作”模式。")
        W_head_new = W0
    else:
        W20 = state_20['lm_head.weight'].float().cpu().numpy()
        print("[*] 正在对 lm_head 权重执行 SVD...")
        U0, S0, V0t = np.linalg.svd(W0, full_matrices=False)
        U2, S2, V2t = np.linalg.svd(W20, full_matrices=False)
        
        k = k_val
        print(f"[*] 使用指定的秩 k = {k}")

        U0_k, S0_k, V0_k = U0[:, :k], S0[:k], V0t[:k, :].T
        U0_tail, S0_tail, V0_tail = U0[:, k:], S0[k:], V0t[k:, :].T
        S2_k = S2[:k]
        
        W0_tail = U0_tail @ np.diag(S0_tail) @ V0_tail.T
        W0_k_comp = U0_k @ np.diag(S0_k) @ V0_k.T

        Rv = np.eye(k)
        if not no_procrustes:
            print("[*] 正在对 V 矩阵进行加权普氏对齐...")
            V2_k = V2t[:k, :].T
            weights = S2_k**2 
            weights /= weights.sum()
            W_diag = np.diag(np.sqrt(weights))
            try:
                Rv, _ = orthogonal_procrustes(V0_k @ W_diag, V2_k @ W_diag)
                print("    - V 矩阵对齐完成。")
            except Exception as e:
                print(f"    - 警告：普氏对齐失败: {e}。将使用单位矩阵作为旋转矩阵。")
        else:
            print("[*] 已跳过普氏对齐 (no_procrustes=True)。")

        print(f"[*] 进入融合模式: '{mode}'")
        if mode == 'spectral':
            if alpha_val is not None:
                alpha = alpha_val
                print(f"    - 使用手动指定的 alpha = {alpha:.4f}")
            else:
                alpha = np.linalg.norm(S0_k) / np.linalg.norm(S2_k)
                print(f"    - 自动计算得到 alpha = {alpha:.4f}")
                if alpha_clip:
                    alpha_clipped = np.clip(alpha, 0.5, 2.0)
                    if alpha != alpha_clipped:
                        print(f"    - alpha 已被裁剪至: {alpha_clipped:.4f}")
                    alpha = alpha_clipped
            
            V0_k_aligned = V0_k @ Rv
            W20_k_aligned_comp = U0_k @ np.diag(alpha * S2_k) @ V0_k_aligned.T
            W_head_mixed = (1 - lam) * W0_k_comp + lam * W20_k_aligned_comp
            W_head_new = W_head_mixed + W0_tail

        elif mode == 'projection':
            print("[*] 正在将 mix20 权重投影到 mix0 的子空间...")
            C = U0_k.T @ W20 @ V0_k
            if alpha_val is not None:
                scale_factor = alpha_val
                print(f"    - 使用手动指定的投影尺度因子 = {scale_factor:.4f}")
            else:
                scale_factor = np.linalg.norm(np.diag(S0_k)) / np.linalg.norm(C)
                print(f"    - 自动计算得到投影尺度因子 = {scale_factor:.4f}")
            
            C_scaled = C * scale_factor
            W20_proj_comp = U0_k @ C_scaled @ V0_k.T
            W_head_mixed = (1 - lam) * W0_k_comp + lam * W20_proj_comp
            W_head_new = W_head_mixed + W0_tail

        elif mode == 'rotation':
            tau = lam
            print(f"[*] 正在执行纯旋转，旋转比例 tau = {tau:.2f}")
            try:
                Rv_tau = fractional_matrix_power(Rv, tau)
                Rv_tau = Rv_tau.real
            except Exception as e:
                print(f"    - 警告：分数次幂计算失败: {e}。将使用线性插值近似。")
                Rv_tau = (1-tau) * np.eye(k) + tau * Rv

            V_rotated = V0_k @ Rv_tau
            W_head_new = U0_k @ np.diag(S0_k) @ V_rotated.T + W0_tail
        else:
            raise ValueError(f"未知的融合模式: {mode}")

    print("[*] 正在构建最终的混合模型 checkpoint...")
    ckpt_hybrid = ckpt_0.copy()
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = torch.from_numpy(W_head_new).float()
    ckpt_hybrid['model'] = state_hybrid
    model_args = ckpt_hybrid['model_args']
    model_args['tie_weights'] = False
    ckpt_hybrid['model_args'] = model_args

    # --- 修改: 输出路径控制 ---
    if output_file_path:
        output_path = output_file_path
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = "hybrid_models"
        # --- 修改: 自动生成的文件名现在包含alpha值 ---
        alpha_str = f"_alpha{alpha_val:.2f}" if alpha_val is not None else ""
        output_path = os.path.join(output_dir, f"grafted_mode-{mode}_k{k_val}_lam{lam:.2f}{alpha_str}_seed{seed}.pt")
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(ckpt_hybrid, output_path)
    print("-" * 60)
    print(f"✅ 精准嫁接模型已生成！")
    print(f"   - Mode: {mode}, Rank (k): {k_val}, Lambda/Tau: {lam:.2f}, Seed: {seed}")
    print(f"   - 保存路径: {output_path}")
    print("-" * 60)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行多模式、多参数的 lm_head 精准嫁接实验 (v3.1)。")
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--k', type=int, required=True, help='指定移植的秩k (k=0为无操作)。')
    parser.add_argument('--lam', type=float, default=1.0, help='融合比例 lambda (或旋转比例 tau)')
    parser.add_argument('--mode', type=str, default='spectral', choices=['spectral', 'projection', 'rotation'], help='融合模式')
    parser.add_argument('--alpha', type=float, default=None, help='手动指定缩放因子 alpha')
    parser.add_argument('--alpha_clip', action='store_true', help='对自动计算的 alpha 进行 [0.5, 2.0] 的裁剪')
    parser.add_argument('--no_procrustes', action='store_true', help='关闭普氏对齐，仅做谱/投影融合')
    # --- 新增: 允许从外部指定输出文件路径 ---
    parser.add_argument('--output_file', type=str, default=None, help='手动指定完整的输出文件路径')
    
    args = parser.parse_args()

    print("\n🔬 开始执行 lm_head 精准嫁接手术 (v3.1 - 已升级) 🔬\n")
    path_0 = get_final_checkpoint_path(0, args.seed)
    path_20 = get_final_checkpoint_path(20, args.seed)
    
    generated_file = create_graft_transplant(
        path_0, path_20, args.lam, args.seed, args.k, 
        args.mode, args.alpha, args.alpha_clip, args.no_procrustes,
        args.output_file  # --- 新增: 传递 output_file 参数 ---
    )
    
    print("💡 如何评估？请运行以下命令:")
    print(f"python evaluate_hybrid_model.py --model_path {generated_file} --data_dir data/simple_graph/composition_90\n")