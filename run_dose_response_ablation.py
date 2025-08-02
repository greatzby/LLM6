# ===================================================================
#             run_dose_response_ablation.py
#
#  自动执行“剂量效应”维度嫁接实验。
#  - 遍历三种主要策略。
#  - 对每种策略，遍历嫁接的维度数量 k = 1 到 10。
#  - 为每个实验组合生成一个独立的混合模型。
# ===================================================================

import torch
import numpy as np
import json
import glob
import os
import argparse

# --- 辅助函数 (从你的脚本中提取并优化) ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """
    辅助函数，用于自动查找给定ratio和seed的最新一次训练的最终模型路径。
    这个版本更健壮，它会寻找最新的目录并返回其中的 'ckpt.pt' 文件。
    """
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
    
    # 选择最新的一个训练目录
    latest_dir = sorted(dirs)[-1]
    
    # 寻找最终的checkpoint文件，通常命名为 'ckpt.pt'
    path = os.path.join(latest_dir, 'ckpt.pt')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"在目录 {latest_dir} 中未找到最终的 checkpoint 文件 'ckpt.pt'")

    print(f"  > 自动定位到模型: {path}")
    return path

def create_hybrid_model(path_0, path_20, transplant_map, strategy_name, k):
    """
    核心嫁接函数。
    接收一个完整的移植地图，但只使用其中的前 k 个维度进行嫁接。
    """
    # 加载模型
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float()

    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float()

    # SVD分解
    U0, S0, V0t = torch.linalg.svd(W0, full_matrices=False)
    U20, S20, V20t = torch.linalg.svd(W20, full_matrices=False)

    # 创建混合版本，先从0%模型完整克隆
    U_hybrid = U0.clone()
    S_hybrid = S0.clone()
    V_hybrid_t = V0t.clone()

    # --- 核心修改：只选择前 k 个维度进行移植 ---
    # 将字典转换为可切片列表
    candidate_dims = list(transplant_map.items())
    dims_to_transplant = candidate_dims[:k]
    
    print(f"  - 策略 '{strategy_name}', 剂量 k={k}: 准备移植 {len(dims_to_transplant)} 个维度...")

    for dim0_idx, dim20_idx in dims_to_transplant:
        # 将20%模型中匹配维度的 U, S, V 子空间，完整地移植到0%模型对应的维度上
        U_hybrid[:, dim0_idx] = U20[:, dim20_idx]
        S_hybrid[dim0_idx] = S20[dim20_idx]
        V_hybrid_t[dim0_idx, :] = V20t[dim20_idx, :]

    # 重构权重矩阵
    W_hybrid = U_hybrid @ torch.diag(S_hybrid) @ V_hybrid_t

    # 创建新的模型 state_dict 和 checkpoint
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    # 同样需要更新 wte.weight (如果它们是绑定的)
    state_hybrid['transformer.wte.weight'] = W_hybrid

    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)

    return ckpt_hybrid


# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行维度嫁接的剂量效应实验。")
    parser.add_argument('--seed', type=int, default=42, help='要操作的随机种子 (42, 123, 或 456)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    STRATEGIES = ['by_angle_top20', 'by_energy_top20', 'control_stable_bottom20']
    K_VALUES = range(1, 11)  # k = 1, 2, ..., 10

    print("="*60)
    print("     🚀 开始执行“剂量效应”维度嫁接实验 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}")
    print("="*60)
    
    # 1. 定位模型文件 (只执行一次)
    print("\n步骤 1: 定位基准模型文件...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        print("🚨 请确保你已经为该种子训练了0%和20%的最终模型，并且 'ckpt.pt' 文件存在于最新的输出目录中。")
        exit(1)

    # 2. 加载候选维度列表 (只执行一次)
    print("\n步骤 2: 加载候选维度列表...")
    candidates_path = f'matched_dims_analysis/seed_{SEED_TO_TEST}_transplant_candidates.json'
    print(f"  > 从 {candidates_path} 加载...")
    try:
        with open(candidates_path, 'r') as f:
            all_candidates = json.load(f)
    except FileNotFoundError:
        print(f"\n🚨 错误: 候选维度文件 '{candidates_path}' 未找到!")
        print("🚨 请先运行 'analyze_matched_dimensions.py' 来生成这个文件。")
        exit(1)
    
    # 3. 循环执行所有实验组合
    print("\n步骤 3: 开始循环生成所有混合模型...")
    total_models = len(STRATEGIES) * len(K_VALUES)
    model_counter = 0

    for strategy in STRATEGIES:
        print("\n" + "-"*50)
        print(f"处理策略: {strategy}")
        print("-"*50)
        
        # 提取该策略的完整移植地图
        transplant_map_str_keys = all_candidates['transplant_candidates'][strategy]
        transplant_map = {int(k): v for k, v in transplant_map_str_keys.items()}

        for k in K_VALUES:
            model_counter += 1
            print(f"\n[{model_counter}/{total_models}] 正在生成: 策略={strategy}, 剂量 k={k}")
            
            # 执行嫁接
            hybrid_ckpt = create_hybrid_model(path_0, path_20, transplant_map, strategy, k)

            # 保存混合模型
            output_filename = f"hybrid_model_seed{SEED_TO_TEST}_{strategy}_k{k}.pt"
            torch.save(hybrid_ckpt, output_filename)
            print(f"  ✅ 成功保存模型到: {output_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 全部 30 个混合模型均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print("下一步：请运行 'run_dose_response_evaluation.sh' 脚本来批量评估这些模型。")