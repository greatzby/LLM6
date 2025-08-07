import torch
import os
import argparse
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

def create_sub_block_transplants(path_0, path_20, seed):
    """创建对 h.0 内部组件进行精细化移植的模型。"""
    print("[*] 正在加载原始模型权重...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']

    # 定义基础组件（除h.0外所有需要从mix20移植的组件）
    base_transplant_keys = ['transformer.wte.weight', 'lm_head.weight', 'transformer.wpe.weight', 'transformer.ln_f.weight']
    
    # 定义 h.0 内部的子组件及其关联的LayerNorm
    h0_attn_keys = ['transformer.h.0.ln_1.weight', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_proj.weight']
    h0_mlp_keys = ['transformer.h.0.ln_2.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_proj.weight']
    
    strategies = [
        {'name': 'base_transplant_only', 'keys': base_transplant_keys, 'desc': '1. 仅移植IO/Pos/LN_f (对照组)'},
        {'name': 'plus_only_attention', 'keys': base_transplant_keys + h0_attn_keys, 'desc': '2. 在基础上，仅加上 h.0 的 Attention 部分'},
        {'name': 'plus_only_ffn', 'keys': base_transplant_keys + h0_mlp_keys, 'desc': '3. 在基础上，仅加上 h.0 的 FFN(MLP) 部分'},
        {'name': 'plus_full_h0', 'keys': base_transplant_keys + h0_attn_keys + h0_mlp_keys, 'desc': '4. 移植完整的h.0 (阳性对照)'},
    ]
    
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    generated_models = []

    for strategy in strategies:
        print("-" * 50)
        print(f"[*] 正在生成模型: {strategy['name']}")
        print(f"    描述: {strategy['desc']}")
        
        state_hybrid = state_0.copy()
        for key in strategy['keys']:
            if key in state_20:
                state_hybrid[key] = state_20[key]
        
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        
        output_path = os.path.join(output_dir, f"hybrid_sub_block_{strategy['name']}_seed{seed}.pt")
        torch.save(ckpt_hybrid, output_path)
        print(f"    ✅ 模型已保存至: {output_path}")
        generated_models.append(output_path)
        
    return generated_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行 h.0 内部组件的细粒度移植实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    print("\n🔬 开始 h.0 内部细粒度分析实验 🔬\n")
    path_0 = get_final_checkpoint_path(0, args.seed)
    path_20 = get_final_checkpoint_path(20, args.seed)
    generated_files = create_sub_block_transplants(path_0, path_20, args.seed)
    
    # --- 自动生成评估脚本 ---
    script_content = "#!/bin/bash\necho '🚀 开始批量评估 h.0 内部组件移植模型... 🚀'\n"
    for model_path in generated_files:
        script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
    script_filename = "run_sub_block_evaluation.sh"
    with open(script_filename, 'w') as f: f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"\n🎉 细粒度模型已生成！请运行 ./{script_filename} 进行评估。")