import torch
import os
import argparse
import glob

# --- (此脚本无需修改您提供的 model.py) ---

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """
    辅助函数，用于查找并验证模型检查点的路径。
    与之前的脚本完全相同，确保能够定位到正确的模型文件。
    """
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

def create_bundle_transplants(path_0, path_20, seed):
    """
    创建对 h.0 内部“功能捆绑包”进行移植的模型。
    这是本次实验的核心。
    """
    print("[*] 正在加载原始模型权重 (mix0 和 mix20)...")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    all_keys = state_0.keys()

    # --- 定义关键的权重集合 ---
    # 根据您提供的 model.py 结构，精确定义权重键名

    # 1. 定义 "Attention 捆绑包" (Attention Bundle)
    # 关键：将 ln_1 和 attn 视为一个不可分割的整体进行移植。
    attn_bundle_keys = [k for k in all_keys if k.startswith('transformer.h.0.ln_1.') or k.startswith('transformer.h.0.attn.')]
    
    # 2. 定义 "MLP 捆绑包" (MLP/FFN Bundle)
    # 关键：将 ln_2 和 mlp 视为一个不可分割的整体进行移植。
    mlp_bundle_keys = [k for k in all_keys if k.startswith('transformer.h.0.ln_2.') or k.startswith('transformer.h.0.mlp.')]

    # --- 定义我们的实验策略 ---
    
    strategies = [
        {
            'name': 'attn_bundle_transplant', 
            'keys_to_transplant': attn_bundle_keys, 
            'desc': '实验1: 移植 Attention Bundle (ln_1 + attn)，h.0的其余部分(ln_2, mlp)来自mix0'
        },
        {
            'name': 'mlp_bundle_transplant', 
            'keys_to_transplant': mlp_bundle_keys, 
            'desc': '实验2: 移植 MLP/FFN Bundle (ln_2 + mlp)，h.0的其余部分(ln_1, attn)来自mix0'
        },
    ]
    
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    generated_models = []

    for strategy in strategies:
        print("-" * 60)
        print(f"[*] 正在生成模型: {strategy['name']}")
        print(f"    描述: {strategy['desc']}")
        
        # 每次都从一个纯净的基线模型 (mix0) 开始
        state_hybrid = state_0.copy()
        
        # 从专家模型 (mix20) 中移植指定的权重 "捆绑包"
        transplant_count = 0
        for key in strategy['keys_to_transplant']:
            if key in state_20:
                state_hybrid[key] = state_20[key]
                transplant_count += 1
        print(f"    > 成功从 mix20 移植了 {transplant_count} 个权重张量。")

        # 注意：与之前的“整块移植”不同，这里我们没有移植 wte, wpe, ln_f, lm_head
        # 这是一个纯粹的“微创”手术，只在 h.0 内部操作。

        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        
        output_path = os.path.join(output_dir, f"hybrid_bundle_{strategy['name']}_seed{seed}.pt")
        torch.save(ckpt_hybrid, output_path)
        print(f"    ✅ 模型已保存至: {output_path}")
        generated_models.append(output_path)
        
    return generated_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行 h.0 内部“功能捆绑包” (Bundle) 的移植实验。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    print("\n🔬 开始 h.0 功能捆绑包 (Bundle) 移植实验 (路线图A) 🔬\n")
    
    try:
        path_0 = get_final_checkpoint_path(0, args.seed)
        path_20 = get_final_checkpoint_path(20, args.seed)
        generated_files = create_bundle_transplants(path_0, path_20, args.seed)
        
        # --- 自动生成评估脚本 ---
        script_content = "#!/bin/bash\n"
        script_content += "echo '🚀 开始批量评估 h.0 功能捆绑包移植模型... 🚀'\n"
        script_content += "echo '将使用确定性的贪心解码进行评估，确保结果稳定。'\n"
        
        for model_path in generated_files:
            script_content += f"\necho '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
            # 使用我们已经修正过的、采用贪心解码的评估脚本
            # 关键：在评估命令中加入 temperature=0
            script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90 --temperature 0\n"
        
        script_filename = "run_bundle_evaluation.sh"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        print("-" * 60)
        print(f"\n🎉 “功能捆绑包”模型已生成！请按以下步骤运行评估。")
        print(f"   评估脚本 '{script_filename}' 也已自动创建。")

    except FileNotFoundError as e:
        print(f"\n❌ 实验终止: {e}")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")