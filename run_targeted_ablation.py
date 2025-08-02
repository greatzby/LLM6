# ===================================================================
#                   run_targeted_ablation.py
#
#    执行“定点”维度嫁接实验，允许指定任意维度组合进行移植。
# ===================================================================

import torch
import numpy as np
import json
import glob
import os
import argparse

# --- 辅助函数 (保持不变) ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """
    自动查找给定ratio和seed的最新一次训练的最终模型路径。
    """
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

def create_targeted_hybrid_model(path_0, path_20, full_transplant_map, indices_to_transplant):
    """
    核心嫁接函数。
    接收一个完整的移植地图，但只使用指定的索引列表进行嫁接。
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

    # --- 核心修改：根据指定的索引列表选择维度 ---
    candidate_dims = list(full_transplant_map.items())
    
    print(f"  - 准备移植指定的 {len(indices_to_transplant)} 个维度...")

    for rank_idx in indices_to_transplant:
        # rank_idx 是用户指定的排名 (例如，第6个)，需要转换为0-based的列表索引
        list_idx = rank_idx - 1 
        if list_idx >= len(candidate_dims):
            print(f"警告: 排名 {rank_idx} 超出候选列表范围，跳过。")
            continue
            
        dim0_idx, dim20_idx = candidate_dims[list_idx]
        print(f"    - 嫁接排名第 {rank_idx} 的维度: 0%模型[{dim0_idx}] <--- 20%模型[{dim20_idx}]")
        
        U_hybrid[:, dim0_idx] = U20[:, dim20_idx]
        S_hybrid[dim0_idx] = S20[dim20_idx]
        V_hybrid_t[dim0_idx, :] = V20t[dim20_idx, :]

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
    parser = argparse.ArgumentParser(description="执行定点维度嫁接实验。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (e.g., 42)')
    parser.add_argument('--strategy', type=str, required=True, help='要使用的策略 (e.g., by_energy_top20)')
    parser.add_argument('--indices', type=str, required=True, help='要嫁接的维度排名，以逗号分隔 (e.g., "6,7,8")')
    args = parser.parse_args()

    # 解析参数
    SEED_TO_TEST = args.seed
    STRATEGY = args.strategy
    INDICES_TO_TRANSPLANT = [int(i) for i in args.indices.split(',')]
    INDICES_STR = "_".join(map(str, INDICES_TO_TRANSPLANT))

    print("="*60)
    print("        🎯 开始执行“定点”维度嫁接实验 🎯        ")
    print(f"操作种子: {SEED_TO_TEST} | 策略: {STRATEGY} | 目标维度排名: {INDICES_TO_TRANSPLANT}")
    print("="*60)
    
    # 1. 定位模型文件
    print("\n步骤 1: 定位基准模型文件...")
    path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
    path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)

    # 2. 加载候选维度列表
    print("\n步骤 2: 加载候选维度列表...")
    candidates_path = f'matched_dims_analysis/seed_{SEED_TO_TEST}_transplant_candidates.json'
    with open(candidates_path, 'r') as f:
        all_candidates = json.load(f)
    
    # 3. 提取该策略的完整移植地图
    full_transplant_map_str_keys = all_candidates['transplant_candidates'][STRATEGY]
    full_transplant_map = {int(k): v for k, v in full_transplant_map_str_keys.items()}

    # 4. 执行定点嫁接
    print("\n步骤 3: 执行定点维度嫁接...")
    hybrid_ckpt = create_targeted_hybrid_model(path_0, path_20, full_transplant_map, INDICES_TO_TRANSPLANT)

    # 5. 保存混合模型
    print("\n步骤 4: 保存生成的混合模型...")
    # 从策略名中去掉 '_top20' 或 '_bottom20' 以简化文件名
    strategy_short_name = STRATEGY.replace('_top20', '').replace('_bottom20', '')
    output_filename = f"hybrid_model_seed{SEED_TO_TEST}_{strategy_short_name}_indices_{INDICES_STR}.pt"
    torch.save(hybrid_ckpt, output_filename)
    
    print("\n" + "="*60)
    print("🎉 定点嫁接成功！🎉")
    print(f"新的混合模型已保存为: {output_filename}")
    print("="*60)