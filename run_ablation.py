"""
run_ablation.py

使用由 'analyze_matched_dimensions.py' 生成的候选维度列表，
执行精确的维度嫁接（消融）实验。

通过修改 CONFIG 部分的 SEED_TO_TEST 和 STRATEGY 来控制实验。
"""

import torch
import numpy as np
import json
import glob
import os

# ===================================================================
#                           实验配置中心
# ===================================================================
# --- 你只需要修改这里 ---
SEED_TO_TEST = 42  # 要操作的种子 (42, 123, 或 456)
STRATEGY = 'by_energy_top20'  # 选择移植策略:
                           # 'by_angle_top20': 移植思想重塑最剧烈的维度 (核心实验)
                           # 'by_energy_top20': 移植重要性提升最大的维度 (对比实验)
                           # 'control_stable_bottom20': 移植最稳定的维度 (控制组实验)
# --- 修改结束 ---
# ===================================================================


# --- 辅助函数 ---

def get_checkpoint_path(ratio, seed, iteration, checkpoint_dir="out_d92"):
    """辅助函数，用于自动查找正确的模型路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
    selected_dir = sorted(dirs)[-1] # 选择最新的时间戳
    path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    print(f"  > 找到模型路径: {path}")
    return path

def transplant_matched_dimensions(path_0, path_20, transplant_map):
    """
    将20%模型的匹配维度精确嫁接到0%模型。
    这是一个完整的“子空间移植”，包括U, S, V三个部分。
    """
    # 加载权重
    print("加载原始模型权重...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float()

    print("加载目标模型权重...")
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float()

    # SVD分解
    print("对两个模型进行SVD分解...")
    U0, S0, V0t = torch.linalg.svd(W0, full_matrices=False)
    U20, S20, V20t = torch.linalg.svd(W20, full_matrices=False)

    # 创建混合版本，先从0%模型完整克隆
    U_hybrid = U0.clone()
    S_hybrid = S0.clone()
    V_hybrid_t = V0t.clone()

    print(f"\n开始移植 {len(transplant_map)} 个维度 (策略: {STRATEGY})...")
    for dim0_idx, dim20_idx in transplant_map.items():
        print(f"  - 嫁接: 0%模型[维度 {dim0_idx}] <--- 20%模型[维度 {dim20_idx}]")
        # 将20%模型中匹配维度的 U, S, V 子空间，完整地移植到0%模型对应的维度上
        U_hybrid[:, dim0_idx] = U20[:, dim20_idx]
        S_hybrid[dim0_idx] = S20[dim20_idx]
        V_hybrid_t[dim0_idx, :] = V20t[dim20_idx, :]

    # 重构权重矩阵
    print("\n重构混合权重矩阵...")
    W_hybrid = U_hybrid @ torch.diag(S_hybrid) @ V_hybrid_t

    # 创建新的模型 state_dict 和 checkpoint
    print("创建新的混合模型checkpoint...")
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid

    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)

    return ckpt_hybrid


# --- 主执行逻辑 ---
if __name__ == "__main__":
    print("="*60)
    print("         执行精确维度嫁接 (Ablation Study)         ")
    print("="*60)
    
    # 1. 定义模型参数
    iteration = 50000
    
    # 2. 自动获取模型路径
    print("步骤 1: 定位模型文件...")
    path_0 = get_checkpoint_path(0, SEED_TO_TEST, iteration)
    path_20 = get_checkpoint_path(20, SEED_TO_TEST, iteration)

    # 3. 加载候选维度列表
    print("\n步骤 2: 加载候选维度列表...")
    candidates_path = f'matched_dims_analysis/seed_{SEED_TO_TEST}_transplant_candidates.json'
    print(f"  > 从 {candidates_path} 加载...")
    with open(candidates_path, 'r') as f:
        candidates = json.load(f)
    
    # 4. 根据策略选择移植地图
    transplant_map_str_keys = candidates['transplant_candidates'][STRATEGY]
    # 将JSON加载的字符串键转换为整数键
    transplant_map = {int(k): v for k, v in transplant_map_str_keys.items()}

    # 5. 执行嫁接
    print("\n步骤 3: 执行维度嫁接...")
    hybrid_ckpt = transplant_matched_dimensions(path_0, path_20, transplant_map)

    # 6. 保存混合模型
    print("\n步骤 4: 保存生成的混合模型...")
    output_filename = f"hybrid_model_seed{SEED_TO_TEST}_{STRATEGY}.pt"
    torch.save(hybrid_ckpt, output_filename)
    
    print("\n" + "="*60)
    print("🎉 实验成功完成！🎉")
    print(f"新的混合模型已保存为: {output_filename}")
    print("现在请使用你的评估流程来测试这个模型的性能。")
    print("="*60)