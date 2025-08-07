import torch
import os
import argparse
import glob
import numpy as np
from model import GPT, GPTConfig

# --- 辅助函数 (与您的脚本一致) ---
def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
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

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化执行“旋转对齐+能量增益”策略的剂量效应扫描实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    K_VALUES = range(0, 11)  # 关键修改：从 k=0 扫描到 k=10

    print("="*60)
    print("     🚀 开始高级嫁接策略的“剂量效应”扫描实验 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}, 扫描范围 k: {list(K_VALUES)}")
    print("="*60)

    # --- 步骤1: 加载模型并执行一次性的对齐和识别 ---
    print("\n步骤 1: 加载模型并计算对齐矩阵和能量增益...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)

    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']

    # 1a. 对齐 c_fc 权重
    w0_fc = state_0['transformer.h.0.mlp.c_fc.weight']
    w20_fc = state_20['transformer.h.0.mlp.c_fc.weight']
    correlation_matrix_fc = w0_fc.T @ w20_fc
    U_fc, _, Vh_fc = torch.svd(correlation_matrix_fc)
    R_fc = Vh_fc.T @ U_fc.T
    w0_fc_aligned = w0_fc @ R_fc

    # 1b. 对齐 c_proj 权重
    w0_proj = state_0['transformer.h.0.mlp.c_proj.weight']
    w20_proj = state_20['transformer.h.0.mlp.c_proj.weight']
    correlation_matrix_proj = w0_proj.T @ w20_proj
    U_proj, _, Vh_proj = torch.svd(correlation_matrix_proj)
    R_proj = Vh_proj.T @ U_proj.T
    w0_proj_aligned = w0_proj @ R_proj

    # 1c. 识别能量增益最高的维度顺序 (只计算一次)
    energy0_aligned = torch.sum(w0_fc_aligned**2, dim=0)
    energy20 = torch.sum(w20_fc**2, dim=0)
    energy_gain = (energy20 - energy0_aligned) / (energy0_aligned + 1e-9)
    sorted_indices = torch.argsort(energy_gain, descending=True)
    print(f"    - 已识别出所有维度的能量增益顺序。")
    print("步骤 1 完成。")

    # --- 步骤2: 循环生成所有k值的混合模型 ---
    print("\n步骤 2: 开始循环生成所有混合模型...")
    generated_models = []
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        print(f"\n  - 正在生成 k={k} 的模型...")
        
        # 获取当前k值需要嫁接的维度
        indices_to_graft = sorted_indices[:k]
        
        # 创建混合模型状态
        state_hybrid = state_0.copy()

        # 移植稳定的“土壤”环境
        keys_to_transplant = [key for key in state_20.keys() if not key.startswith('transformer.h.0.mlp')]
        for key in keys_to_transplant:
            if key in state_20: state_hybrid[key] = state_20[key]

        # 对MLP进行精确嫁接
        w_hybrid_fc = w0_fc_aligned.clone()
        w_hybrid_proj = w0_proj_aligned.clone()
        
        if k > 0:
            print(f"    > 嫁接维度: {indices_to_graft.tolist()}")
            for idx in indices_to_graft:
                w_hybrid_fc[:, idx] = w20_fc[:, idx]
                w_hybrid_proj[idx, :] = w20_proj[idx, :]
        else:
            print("    > k=0, 不嫁接任何维度，仅保留旋转对齐后的状态。")

        state_hybrid['transformer.h.0.mlp.c_fc.weight'] = w_hybrid_fc
        state_hybrid['transformer.h.0.mlp.c_proj.weight'] = w_hybrid_proj
        state_hybrid['transformer.h.0.mlp.c_fc.bias'] = state_20['transformer.h.0.mlp.c_fc.bias']
        state_hybrid['transformer.h.0.mlp.c_proj.bias'] = state_20['transformer.h.0.mlp.c_proj.bias']
        
        # 保存模型
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        output_path = os.path.join(output_dir, f"hybrid_advanced_graft_k{k}_seed{SEED_TO_TEST}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        print(f"    ✅ 模型已保存至: {output_path}")

    # --- 步骤3: 自动生成评估脚本 ---
    print("\n步骤 3: 生成批量评估脚本...")
    script_content = "#!/bin/bash\necho '🚀 开始批量评估高级嫁接模型的剂量效应... 🚀'\n"
    for model_path in generated_models:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_advanced_grafting_evaluation_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 全部模型均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请运行 './{script_filename}' 脚本来批量评估这些模型。")