import torch
import os
import argparse
import glob
import numpy as np

# --- 辅助函数 (无需修改) ---
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
    parser = argparse.ArgumentParser(description="自动化执行基于旋转对齐的lm_head维度嫁接扫描实验 (V2 - 修正版)。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed
    K_VALUES = range(0, 11)

    print("="*60)
    print("     🚀 开始 lm_head 旋转对齐嫁接的“剂量效应”扫描 (V2) 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}, 扫描范围 k: {list(K_VALUES)}")
    print("="*60)

    # --- 步骤1: 加载模型并执行正确的对齐和分解 ---
    print("\n步骤 1: 加载模型，对齐整个权重空间，然后SVD分解...")
    try:
        path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
        path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)

    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight'].float()

    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight'].float()

    # --- 核心修正逻辑 ---
    # 1a. 直接对齐整个权重矩阵 W0 和 W20
    print("    - 正在计算旋转矩阵以对齐整个lm_head空间...")
    correlation_matrix = W0.T @ W20
    U_rot, _, Vh_rot = torch.svd(correlation_matrix)
    R = Vh_rot.T @ U_rot.T
    W0_aligned = W0 @ R
    print("    - 已将W0旋转对齐到W20的空间。")

    # 1b. 对对齐后的W0_aligned和原始的W20进行SVD分解
    print("    - 正在对对齐后的W0和原始W20进行SVD分解...")
    U0a, S0a, V0ta = torch.linalg.svd(W0_aligned, full_matrices=False)
    U20, S20, V20t = torch.linalg.svd(W20, full_matrices=False)
    
    # 1c. 识别能量增益最高的维度顺序
    energy0_aligned = S0a**2
    energy20 = S20**2
    energy_gain = (energy20 - energy0_aligned) / (energy0_aligned + 1e-9)
    sorted_indices = torch.argsort(energy_gain, descending=True)
    print(f"    - 已识别出所有维度的能量增益顺序。")
    print("步骤 1 完成。")

    # --- 步骤2: 循环生成所有k值的混合模型 ---
    print("\n步骤 2: 开始循环生成所有混合模型...")
    generated_models = []
    output_dir = "hybrid_models_lm_head_v2" # 使用新目录
    os.makedirs(output_dir, exist_ok=True)

    for k in K_VALUES:
        print(f"\n  - 正在生成 k={k} 的模型...")
        indices_to_graft = sorted_indices[:k]
        
        # 创建混合SVD组件，基础是已经对齐过的W0的组件
        U_hybrid = U0a.clone()
        S_hybrid = S0a.clone()
        V_hybrid_t = V0ta.clone()

        if k > 0:
            print(f"    > 嫁接维度: {indices_to_graft.tolist()}")
            for idx in indices_to_graft:
                # 嫁接W20对应维度的组件
                U_hybrid[:, idx] = U20[:, idx]
                S_hybrid[idx] = S20[idx]
                V_hybrid_t[idx, :] = V20t[idx, :]
        else:
            print("    > k=0, 不嫁接，仅使用W0的对齐版本。")
        
        W_hybrid = U_hybrid @ torch.diag(S_hybrid) @ V_hybrid_t
        state_hybrid = state_0.copy()
        state_hybrid['lm_head.weight'] = W_hybrid
        state_hybrid['transformer.wte.weight'] = W_hybrid

        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        output_path = os.path.join(output_dir, f"hybrid_lm_head_rotated_v2_k{k}_seed{SEED_TO_TEST}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        print(f"    ✅ 模型已保存至: {output_path}")

    # --- 步骤3: 自动生成评估脚本 ---
    print("\n步骤 3: 生成批量评估脚本...")
    script_content = "#!/bin/bash\necho '🚀 开始批量评估 lm_head 旋转对齐嫁接模型(V2)的剂量效应... 🚀'\n"
    for model_path in generated_models:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_lm_head_rotation_evaluation_v2_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 全部 lm_head 混合模型(V2)均已成功生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请运行 './{script_filename}' 脚本来批量评估这些模型。")