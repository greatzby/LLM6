import torch
import os
import argparse
import numpy as np
import glob

# --- (此脚本无需修改 model.py) ---

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

def create_interpolation_transplants(path_0, path_20, seed):
    """对 h.0 的权重进行线性插值，创建混合比例模型。"""
    print("[*] 正在加载原始模型权重...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']

    # 定义需要被插值的 h.0 权重 (不包括非参数的 attn.bias)
    h0_keys = [k for k in state_0.keys() if k.startswith('transformer.h.0.') and 'attn.bias' not in k]
    
    # 定义直接从 mix20 移植的基础组件 (即所有非 h.0 的组件)
    base_transplant_keys = [k for k in state_0.keys() if not k.startswith('transformer.h.0.')]

    # 定义一系列混合比例 alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
    
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    generated_models = []

    print(f"[*] 开始生成 h.0 权重插值模型 (Alpha: {alphas})...")
    
    for alpha in alphas:
        print(f"    - 正在处理 alpha = {alpha:.2f}")
        
        # 1. 创建一个基础模型，直接移植非 h.0 的所有组件
        state_hybrid = state_0.copy()
        for key in base_transplant_keys:
            state_hybrid[key] = state_20[key]
            
        # 2. 对 h.0 的权重进行线性插值
        for key in h0_keys:
            w_0, w_20 = state_0[key], state_20[key]
            state_hybrid[key] = (1 - alpha) * w_0 + alpha * w_20
            
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        
        output_path = os.path.join(output_dir, f"hybrid_interp_alpha{alpha:.2f}_seed{seed}.pt")
        torch.save(ckpt_hybrid, output_path)
        generated_models.append(output_path)
        
    print("    ✅ 所有插值模型已生成完毕。")
    return generated_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行 h.0 权重的线性插值实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    print("\n🔬 开始 h.0 权重插值实验 🔬\n")
    path_0 = get_final_checkpoint_path(0, args.seed)
    path_20 = get_final_checkpoint_path(20, args.seed)
    generated_files = create_interpolation_transplants(path_0, path_20, args.seed)
    
    # --- 自动生成评估脚本 ---
    script_content = "#!/bin/bash\necho '🚀 开始批量评估 h.0 权重插值模型... 🚀'\n"
    for model_path in generated_files:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    script_filename = "run_interpolation_evaluation.sh"
    with open(script_filename, 'w') as f: f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"\n🎉 插值模型已生成！请运行 ./{script_filename} 进行评估。")