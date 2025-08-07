import torch
import os
import argparse
import glob
import numpy as np

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于自动查找并验证模型检查点的路径。"""
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

def align_and_graft_layer(W0, W20, k):
    """
    对给定的权重矩阵W0和W20，执行旋转对齐、能量排序和维度嫁接。
    这是我们所有经验的最终结晶。
    注意: PyTorch的nn.Linear权重形状为 (out_features, in_features)。
    我们的目标是识别并嫁接最重要的 "输出特征"，即矩阵的 "行"。
    """
    if k == 0:
        return W0.clone()  # k=0时，不做任何操作，返回原始W0的克隆

    # 1. 旋转对齐 (Procrustes分析)，找到最佳旋转R使得 R @ W0 ≈ W20
    # 我们要对齐的是行空间，所以计算 W20 和 W0 的相关性
    correlation_matrix = W20 @ W0.T
    U_rot, _, Vh_rot = torch.svd(correlation_matrix)
    R = Vh_rot.T @ U_rot.T # 修正：正确的旋转矩阵公式
    W0_aligned = R @ W0

    # 2. 能量增益排序
    # 能量定义为每个输出特征（行向量）的L2范数的平方
    energy0_aligned = torch.sum(W0_aligned**2, dim=1)
    energy20 = torch.sum(W20**2, dim=1)
    energy_gain = energy20 - energy0_aligned  # 我们关心绝对增益

    sorted_indices = torch.argsort(energy_gain, descending=True)
    
    # 3. 维度嫁接
    indices_to_graft = sorted_indices[:k]
    W_hybrid = W0_aligned.clone()
    
    # 将W20中能量增益最高的 "行" 嫁接到对齐后的W0上
    print(f"      > 正在嫁接维度 (行索引): {indices_to_graft.tolist()}")
    for idx in indices_to_graft:
        W_hybrid[idx, :] = W20[idx, :]
        
    return W_hybrid

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对h.0 MLP层进行旋转对齐维度嫁接的最终扫描实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    K_VALUES = range(0, 11)  # 扫描 k=0 到 10

    print("="*60)
    print("     🚀 开始 h.0 MLP层旋转对齐嫁接的“剂量效应”扫描 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}, 扫描范围 k: {list(K_VALUES)}")
    print("="*60)

    # --- 步骤1: 加载基准模型 ---
    print("\n步骤 1: 加载基准模型...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)
        
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    print("步骤 1 完成。")

    # --- 步骤2: 循环生成所有k值的混合模型 ---
    print("\n步骤 2: 开始循环生成所有混合模型...")
    generated_models = []
    output_dir = f"hybrid_models_h0_final_seed{SEED_TO_TEST}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义h.0中需要嫁接的MLP层
    layers_to_graft = ['mlp.c_fc', 'mlp.c_proj']

    for k in K_VALUES:
        print(f"\n  - 正在生成 k={k} 的模型...")
        
        # 从 mix0 的状态字典开始构建，确保这是一个干净的起点
        state_hybrid = {key: val.clone() for key, val in state_0.items()}
        
        for layer_base_name in layers_to_graft:
            print(f"    - 处理层: {layer_base_name}")
            # 处理权重
            layer_weight_key = f'transformer.h.0.{layer_base_name}.weight'
            W0 = state_0[layer_weight_key]
            W20 = state_20[layer_weight_key]
            W_grafted = align_and_graft_layer(W0, W20, k)
            state_hybrid[layer_weight_key] = W_grafted

            # 处理偏置项 (bias)，直接使用mix20的
            # 偏置项与输出特征（行）直接相关，因此也需要嫁接
            layer_bias_key = f'transformer.h.0.{layer_base_name}.bias'
            if layer_bias_key in state_0 and layer_bias_key in state_20:
                b0_aligned = state_0[layer_bias_key].clone() # 假设对齐后的偏置项与原偏置项相似
                b20 = state_20[layer_bias_key]
                b_hybrid = b0_aligned
                # 嫁接与权重行对应的偏置项
                indices_to_graft = torch.argsort((b20 - b0_aligned)**2, descending=True)[:k]
                for idx in indices_to_graft:
                     b_hybrid[idx] = b20[idx]
                state_hybrid[layer_bias_key] = b_hybrid


        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        output_path = os.path.join(output_dir, f"hybrid_h0_rotated_k{k}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        print(f"    ✅ 模型已保存至: {output_path}")

    # --- 步骤3: 自动生成评估脚本 ---
    print("\n步骤 3: 生成批量评估脚本...")
    script_content = f"#!/bin/bash\necho '🚀 开始批量评估 h.0 旋转对齐嫁接模型(Final)的剂量效应... 🚀'\n"
    for model_path in generated_models:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_h0_rotation_evaluation_final_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 最终决战模型均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请运行 './{script_filename}' 脚本，见证最终的结果。")