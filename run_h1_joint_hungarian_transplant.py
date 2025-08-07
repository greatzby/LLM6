import torch
import os
import argparse
import glob
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

# (此处省略您提供的完整模型代码，假设它已保存在一个名为 model.py 的文件中)
# from model import GPT, GPTConfig 

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于查找并验证模型检查点的路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path): raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")
    print(f"  > 定位到模型: {path}")
    return path

def joint_hungarian_neuron_transplant(state_0, state_20, k, layer_index=1):
    """
    在指定的transformer层(h.1)上，使用联合匈牙利算法进行神经元移植。
    """
    if k == 0:
        return state_0.copy()

    # --- 关键修改：将层级参数化 ---
    w_fc_key = f'transformer.h.{layer_index}.mlp.c_fc.weight'
    w_proj_key = f'transformer.h.{layer_index}.mlp.c_proj.weight'
    print(f"  🎯 目标层级: h.{layer_index}")

    W_fc_0, W_proj_0 = state_0[w_fc_key], state_0[w_proj_key]
    W_fc_20, W_proj_20 = state_20[w_fc_key], state_20[w_proj_key]

    print("      > 步骤1: 构建联合功能相似性成本矩阵...")
    input_sim = F.cosine_similarity(W_fc_0.unsqueeze(1), W_fc_20.unsqueeze(0), dim=-1)
    output_sim = F.cosine_similarity(W_proj_0.T.unsqueeze(1), W_proj_20.T.unsqueeze(0), dim=-1)
    total_sim = 0.5 * input_sim + 0.5 * output_sim
    cost_matrix = 1 - total_sim.cpu().numpy()

    print("      > 步骤2: 运行匈牙利算法进行最优化功能对齐...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print("      > 步骤3: 按mix20中匹配神经元的重要性排序...")
    matched_mix20_indices = col_ind
    neuron_importance = torch.norm(W_proj_20[:, matched_mix20_indices], p=2, dim=0)
    sorted_importance_indices = torch.argsort(neuron_importance, descending=True)

    indices_to_transplant = sorted_importance_indices[:k]
    
    state_hybrid = {key: val.clone() for key, val in state_0.items()}
    
    print(f"      > 步骤4: 移植最重要的 {k} 个对齐的神经元...")
    for i in indices_to_transplant:
        mix0_idx = row_ind[i]
        mix20_idx = col_ind[i]
        state_hybrid[w_fc_key][mix0_idx, :] = W_fc_20[mix20_idx, :]
        state_hybrid[w_proj_key][:, mix0_idx] = W_proj_20[:, mix20_idx]
        
    return state_hybrid

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在h.1 MLP层应用联合匈牙利算法进行神经元对齐移植。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    LAYER_TO_TEST = 1  # <--- 核心改变在这里！我们正在攻击h.1层
    K_VALUES = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80] # 扩大扫描范围

    print("="*60)
    print(f"     🚀 开始 h.{LAYER_TO_TEST} MLP层联合匈牙利对齐移植 (决战 v4.0) 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}, 扫描范围 k: {K_VALUES}")
    print("="*60)

    print("\n步骤 1: 加载基准模型...")
    path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
    path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    print("步骤 1 完成。")

    print("\n步骤 2: 开始循环生成所有混合模型...")
    generated_models = []
    output_dir = f"hybrid_models_h{LAYER_TO_TEST}_joint_hungarian_seed{SEED_TO_TEST}"
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        print(f"\n  - 正在生成 k={k} 的模型...")
        state_hybrid = joint_hungarian_neuron_transplant(state_0, state_20, k, layer_index=LAYER_TO_TEST)
        
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        output_path = os.path.join(output_dir, f"hybrid_h{LAYER_TO_TEST}_joint_hungarian_k{k}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        print(f"    ✅ 模型已保存至: {output_path}")

    print("\n步骤 3: 生成批量评估脚本...")
    script_content = f"#!/bin/bash\necho '🚀 开始批量评估h.{LAYER_TO_TEST}联合匈牙利对齐移植模型... 🚀'\n"
    for model_path in generated_models:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_h{LAYER_TO_TEST}_joint_hungarian_evaluation_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 新战场模型已生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请运行 './{script_filename}' 脚本。我们现在有充分的理由相信，这次的结果将会完全不同。")