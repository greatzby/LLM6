# ----------- 文件: graft_advanced.py (修正后) -----------
import torch
import os
import argparse
import glob
import numpy as np
from scipy.linalg import orthogonal_procrustes

# 从您修改后的 model.py 中导入
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

def k95(s):
    """计算捕获95%能量所需的奇异值数量(k)"""
    s_squared = s**2
    energy_cumsum = np.cumsum(s_squared)
    total_energy = energy_cumsum[-1]
    k = np.searchsorted(energy_cumsum, 0.95 * total_energy) + 1
    return int(k)

def create_graft_transplant(path_0, path_20, lam, seed):
    """
    执行精准的、解耦的、低秩的 lm_head 嫁接手术。
    """
    print("[*] 正在加载原始模型权重...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')

    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    W0 = state_0['lm_head.weight'].numpy()
    W20 = state_20['lm_head.weight'].numpy()

    print("[*] 正在对 lm_head 权重执行 SVD...")
    U0, S0, V0t = np.linalg.svd(W0, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(W20, full_matrices=False)

    print("[*] 计算保留95%能量的秩 k...")
    k = min(k95(S0), k95(S2))
    print(f"    - mix0 的 k@95% = {k95(S0)}")
    print(f"    - mix20 的 k@95% = {k95(S2)}")
    print(f"    - 将使用 k = {k}")

    U0_k, S0_k, V0_k = U0[:, :k], S0[:k], V0t[:k, :].T
    U0_tail, S0_tail, V0_tail = U0[:, k:], S0[k:], V0t[k:, :].T
    S2_k, V2_k = S2[:k], V2t[:k, :].T

    print("[*] 正在对 V 矩阵进行加权普氏对齐...")
    weights = S2_k**2
    weights /= weights.sum()
    W_diag = np.diag(np.sqrt(weights))
    Rv, _ = orthogonal_procrustes(V0_k @ W_diag, V2_k @ W_diag)
    V0_k_aligned = V0_k @ Rv
    print("    - V 矩阵对齐完成。")

    # --- 4. 重建新的 lm_head 权重 (修正后的逻辑) ---
    print("[*] 正在重建新的 lm_head 权重...")
    
    # 计算尺度校准因子
    alpha = np.linalg.norm(S0_k) / np.linalg.norm(S2_k)
    print(f"    - 计算得到尺度校准因子 alpha = {alpha:.4f}")

    # 组件A: mix0 的原始头部
    W0_k_comp = U0_k @ np.diag(S0_k) @ V0_k.T
    
    # 组件B: mix20 的对齐后头部 (在 mix0 的 U 基上表达)
    W20_k_aligned_comp = U0_k @ np.diag(alpha * S2_k) @ V0_k_aligned.T

    # 使用 lambda 线性插值这两个组件
    W_head_mixed = (1 - lam) * W0_k_comp + lam * W20_k_aligned_comp
    
    # 加上 mix0 的尾部
    W_tail = U0_tail @ np.diag(S0_tail) @ V0_tail.T
    W_head_new = W_head_mixed + W_tail
    
    print("    - 权重重建完成。")

    # --- 5. 构建新的、解耦的混合模型 ---
    print("[*] 正在构建最终的混合模型 checkpoint...")
    ckpt_hybrid = ckpt_0.copy()
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = torch.from_numpy(W_head_new).float()
    ckpt_hybrid['model'] = state_hybrid
    model_args = ckpt_hybrid['model_args']
    model_args['tie_weights'] = False
    ckpt_hybrid['model_args'] = model_args

    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"grafted_advanced_lam{lam:.2f}_seed{seed}.pt")
    torch.save(ckpt_hybrid, output_path)
    print("-" * 60)
    print(f"✅ 精准嫁接模型已生成！")
    print(f"   - Lambda (融合比例): {lam}")
    print(f"   - Seed (随机种子): {seed}")
    print(f"   - 保存路径: {output_path}")
    print("-" * 60)
    return output_path

# main 函数部分保持不变，无需修改
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行解耦的、低秩的、对齐的 lm_head 精准嫁接实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    parser.add_argument('--lam', type=float, default=0.5, help='Σ能谱的融合比例 lambda (0.0 to 1.0)')
    args = parser.parse_args()

    if not (0.0 <= args.lam <= 1.0):
        raise ValueError("lambda 参数必须在 0.0 和 1.0 之间")

    print("\n🔬 开始执行 lm_head 精准嫁接手术 🔬\n")
    path_0 = get_final_checkpoint_path(0, args.seed)
    path_20 = get_final_checkpoint_path(20, args.seed)
    
    generated_file = create_graft_transplant(path_0, path_20, args.lam, args.seed)
    
    print("💡 如何评估？请运行以下命令:")
    print(f"python evaluate_hybrid_model.py --model_path {generated_file} --data_dir data/simple_graph/composition_90\n")