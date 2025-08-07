import torch
import os
import argparse
import glob
import numpy as np

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

def transplant_neurons(state_0, state_20, k):
    """
    执行完整的神经元移植。
    一个神经元 = c_fc的一行 + c_proj的一列。
    (已移除对偏置项的处理，因为模型中不存在偏置项)
    """
    if k == 0:
        return state_0.copy()

    state_hybrid = {key: val.clone() for key, val in state_0.items()}
    
    # 定义MLP层的权重键
    w_fc_key = 'transformer.h.0.mlp.c_fc.weight'
    w_proj_key = 'transformer.h.0.mlp.c_proj.weight'

    W_fc_20, W_proj_20 = state_20[w_fc_key], state_20[w_proj_key]

    # 1. 识别mix20中最重要的k个神经元
    # 重要性定义为神经元输出权重(c_proj的列)的L2范数
    neuron_importance = torch.norm(W_proj_20, p=2, dim=0)
    indices_to_transplant = torch.topk(neuron_importance, k).indices
    print(f"      > 正在移植神经元 (索引): {indices_to_transplant.tolist()}")

    # 2. 执行完整的神经元移植
    for idx in indices_to_transplant:
        # 替换输入权重 (c_fc 的行)
        state_hybrid[w_fc_key][idx, :] = W_fc_20[idx, :]
        # 替换输出权重 (c_proj 的列)
        state_hybrid[w_proj_key][:, idx] = W_proj_20[:, idx]
        
    return state_hybrid

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对h.0 MLP层进行完整的神经元移植扫描。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    K_VALUES = [0, 1, 2, 5, 10, 20, 40, 80] 

    print("="*60)
    print("     🚀 开始 h.0 MLP层神经元移植扫描 (最终修正版 v2) 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}, 扫描范围 k: {K_VALUES}")
    print("="*60)

    # --- 步骤1: 加载基准模型 ---
    print("\n步骤 1: 加载基准模型...")
    path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
    path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    print("步骤 1 完成。")

    # --- 步骤2: 循环生成所有k值的混合模型 ---
    print("\n步骤 2: 开始循环生成所有混合模型...")
    generated_models = []
    output_dir = f"hybrid_models_neuron_transplant_seed{SEED_TO_TEST}"
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        print(f"\n  - 正在生成 k={k} 的模型...")
        state_hybrid = transplant_neurons(state_0, state_20, k)
        
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        output_path = os.path.join(output_dir, f"hybrid_neuron_k{k}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        print(f"    ✅ 模型已保存至: {output_path}")

    # --- 步骤3: 自动生成评估脚本 ---
    print("\n步骤 3: 生成批量评估脚本...")
    script_content = f"#!/bin/bash\necho '🚀 开始批量评估神经元移植模型 (v2)... 🚀'\n"
    for model_path in generated_models:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_neuron_transplant_evaluation_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 神经元移植模型均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请再次运行 './{script_filename}' 脚本。这次它应该能顺利完成了。")