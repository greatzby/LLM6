# ===================================================================
#           run_classic_svd_control.py
#
#  为“经典SVD对照实验”生成混合模型。
#  - 忽略维度匹配，直接按奇异值排序进行替换。
#  - 策略1: 替换能量最高的 Top-k 维度。
#  - 策略2: 替换能量最低的 Bottom-k 维度。
# ===================================================================

import torch
import numpy as np
import glob
import os
import argparse

# --- 辅助函数 (从你之前的脚本中复用) ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于自动查找给定ratio和seed的最终模型路径。"""
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

    print(f"  > 自动定位到模型: {path}")
    return path

# --- 核心嫁接函数 (为本实验特别重写) ---

def create_classic_svd_hybrid(path_0, path_20, strategy_name, k):
    """
    核心嫁接函数，执行“经典SVD”替换，不进行维度匹配。
    """
    # 加载模型
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float()

    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float()

    # SVD分解 (PyTorch的svd默认按奇异值从大到小排序)
    U0, S0, V0t = torch.linalg.svd(W0, full_matrices=False)
    U20, S20, V20t = torch.linalg.svd(W20, full_matrices=False)

    # 创建混合版本，先从0%模型完整克隆
    U_hybrid = U0.clone()
    S_hybrid = S0.clone()
    V_hybrid_t = V0t.clone()
    
    num_dims = S0.shape[0]
    print(f"  - 策略 '{strategy_name}', 剂量 k={k}: 准备替换 {k} 个维度...")

    if strategy_name == 'classic_svd_top':
        # 替换能量最高的 k 个维度
        indices_to_replace = range(k)
    elif strategy_name == 'classic_svd_bottom':
        # 替换能量最低的 k 个维度
        indices_to_replace = range(num_dims - k, num_dims)
    else:
        raise ValueError(f"未知的策略: {strategy_name}")

    for i in indices_to_replace:
        # 直接用20%模型中相同顺位的维度，覆盖0%模型的维度
        U_hybrid[:, i] = U20[:, i]
        S_hybrid[i] = S20[i]
        V_hybrid_t[i, :] = V20t[i, :]

    # 重构权重矩阵
    W_hybrid = U_hybrid @ torch.diag(S_hybrid) @ V_hybrid_t

    # 创建新的模型 state_dict 和 checkpoint
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    state_hybrid['transformer.wte.weight'] = W_hybrid

    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)

    return ckpt_hybrid


# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行经典SVD对照实验。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (e.g., 42, 123, 456)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    # 定义本次实验的两种新策略
    STRATEGIES = ['classic_svd_top', 'classic_svd_bottom']
    K_VALUES = range(1, 11)  # k = 1, 2, ..., 10

    print("="*60)
    print("     🔬 开始执行“经典SVD对照实验” 🔬     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}")
    print("="*60)
    
    # 1. 定位模型文件 (只执行一次)
    print("\n步骤 1: 定位基准模型文件...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        print("🚨 请仔细检查你的输出目录，确认最终模型文件确实存在并且命名符合预期。")
        exit(1)

    # 2. 循环执行所有实验组合
    print("\n步骤 2: 开始循环生成所有对照组混合模型...")
    total_models = len(STRATEGIES) * len(K_VALUES)
    model_counter = 0

    for strategy in STRATEGIES:
        print("\n" + "-"*50)
        print(f"处理策略: {strategy}")
        print("-"*50)
        
        for k in K_VALUES:
            model_counter += 1
            print(f"\n[{model_counter}/{total_models}] 正在生成: 策略={strategy}, 剂量 k={k}")
            
            # 执行嫁接
            hybrid_ckpt = create_classic_svd_hybrid(path_0, path_20, strategy, k)

            # 保存混合模型 (注意文件名中包含新策略名)
            output_filename = f"hybrid_model_seed{SEED_TO_TEST}_{strategy}_k{k}.pt"
            torch.save(hybrid_ckpt, output_filename)
            print(f"  ✅ 成功保存模型到: {output_filename}")

    print("\n" + "="*60)
    print(f"🎉🎉🎉 种子 {SEED_TO_TEST} 的全部 20 个对照模型均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print("下一步：请运行 'run_classic_svd_evaluation.sh' 脚本来批量评估这些模型。")