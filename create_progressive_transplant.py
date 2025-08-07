import torch
import os
import argparse
import glob

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于自动查找给定ratio和seed的最新一次训练的最终模型路径。"""
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
    print(f"  > 定位到模型: {path}")
    return path

def get_block_keys(state_dict, layer_index):
    """动态获取一个Transformer Block内的所有权重键。"""
    prefix = f'transformer.h.{layer_index}.'
    return [key for key in state_dict.keys() if key.startswith(prefix)]

def create_progressive_transplants(path_0, path_20, seed):
    """根据您的方案，创建一系列渐进式移植模型。"""
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model']
    state_20 = ckpt_20['model']

    # --- 定义渐进式替换策略 ---
    # 策略0: 仅替换输出/输入层 (我们已经知道会失败，作为基线)
    strategy_0_keys = ['lm_head.weight', 'transformer.wte.weight']
    
    # 策略1: 加上最终的LayerNorm
    strategy_1_keys = strategy_0_keys + ['transformer.ln_f.weight', 'transformer.ln_f.bias']
    
    # 策略2: 再加上最后一个Block (h.1)
    strategy_2_keys = strategy_1_keys + get_block_keys(state_20, 1)
    
    # 策略3: 再加上第一个Block (h.0)，即全脑移植
    strategy_3_keys = strategy_2_keys + get_block_keys(state_20, 0)

    strategies = [
        {'name': 'step1_io_only',         'keys': strategy_0_keys, 'desc': '1. 仅替换输入/输出层 (wte, lm_head)'},
        {'name': 'step2_plus_final_ln',   'keys': strategy_1_keys, 'desc': '2. 加上最终层归一化 (ln_f)'},
        {'name': 'step3_plus_last_block', 'keys': strategy_2_keys, 'desc': '3. 加上最后一个Transformer Block (h.1)'},
        {'name': 'step4_full_transplant', 'keys': strategy_3_keys, 'desc': '4. 替换所有Blocks (h.0, h.1) - 完全移植'},
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
        
        print(f"    总共替换了 {len(strategy['keys'])} 个权重张量。")
        
        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        
        output_path = os.path.join(output_dir, f"hybrid_progressive_{strategy['name']}_seed{seed}.pt")
        torch.save(ckpt_hybrid, output_path)
        print(f"    ✅ 模型已保存至: {output_path}")
        generated_models.append(output_path)
        
    return generated_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行渐进式分层嫁接实验，以找到能力的最小必要移植集。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 开始执行“最小必要移植集”定位实验 🔬")
    print(f"操作种子 (Seed): {args.seed}")
    print("="*60)

    try:
        path_0 = get_final_checkpoint_path(0, args.seed)
        path_20 = get_final_checkpoint_path(20, args.seed)
        
        generated_files = create_progressive_transplants(path_0, path_20, args.seed)
        
        # --- 自动生成评估脚本 ---
        script_content = "#!/bin/bash\n"
        script_content += "echo '=================================================='\n"
        script_content += "echo '🚀 开始批量评估渐进式移植模型... 🚀'\n"
        script_content += "echo '=================================================='\n\n"
        
        for model_path in generated_files:
            script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
            script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
            
        script_filename = "run_progressive_evaluation.sh"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        os.chmod(script_filename, 0o755) # 使脚本可执行
        
        print("\n" + "="*60)
        print("🎉 所有渐进式模型已生成完毕！")
        print(f"👉 我已为您自动创建了一个评估脚本: ./{script_filename}")
        print("   请直接在终端运行它，以获得所有结果。")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        print("🚨 请确认模型文件存在且路径正确。")
        exit(1)