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

def create_final_transplants(path_0, path_20, seed):
    """
    根据模型的真实结构 (n_layer=1, bias=False) 创建正确的渐进式移植模型。
    此版本不再做任何假设，而是动态地从模型文件中发现权重键。
    """
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model']
    state_20 = ckpt_20['model']
    all_keys = state_20.keys()

    # --- 动态地、基于真实情况发现权重键 ---
    # 1. 输入/输出层 (IO)
    io_keys = [k for k in all_keys if k in ['transformer.wte.weight', 'lm_head.weight']]
    
    # 2. 最终的LayerNorm层
    ln_f_keys = [k for k in all_keys if k.startswith('transformer.ln_f.')]
    
    # 3. 唯一的Transformer Block (h.0)
    #    注意：我们只关心可训练的权重，所以忽略'.attn.bias'这个buffer
    h0_keys = [k for k in all_keys if k.startswith('transformer.h.0.') and 'attn.bias' not in k]

    # --- 定义正确的、基于真实结构的策略 ---
    strategies = [
        {
            'name': 'step1_io_only',
            'keys': io_keys,
            'desc': '1. 仅替换输入/输出层 (wte, lm_head)'
        },
        {
            'name': 'step2_plus_final_ln',
            'keys': io_keys + ln_f_keys,
            'desc': '2. 加上最终层归一化 (ln_f)'
        },
        {
            'name': 'step3_full_transplant',
            'keys': io_keys + ln_f_keys + h0_keys,
            'desc': '3. 加上唯一的Transformer Block (h.0) - 完全移植'
        },
    ]
    
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    generated_models = []

    for strategy in strategies:
        print("-" * 50)
        print(f"[*] 正在生成模型: {strategy['name']}")
        print(f"    描述: {strategy['desc']}")
        
        num_keys = len(strategy['keys'])
        print(f"    总共将替换 {num_keys} 个权重张量。")

        state_hybrid = state_0.copy()
        for key in strategy['keys']:
            if key in state_20:
                state_hybrid[key] = state_20[key]
        
        # 验证移植后的总键数是否与原始键数一致
        assert len(state_hybrid.keys()) == len(state_0.keys())

        ckpt_hybrid = ckpt_0.copy()
        ckpt_hybrid['model'] = state_hybrid
        
        output_path = os.path.join(output_dir, f"hybrid_final_{strategy['name']}_seed{seed}.pt")
        torch.save(ckpt_hybrid, output_path)
        print(f"    ✅ 模型已保存至: {output_path}")
        generated_models.append(output_path)
        
    return generated_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据模型的真实结构，执行最终的、正确的渐进式嫁接实验。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 开始执行最终的、正确的“最小必要移植集”定位实验 🔬")
    print(f"操作种子 (Seed): {args.seed}")
    print("="*60)

    try:
        # 清理之前错误的模型文件
        old_files = glob.glob('hybrid_models/hybrid_progressive_*.pt')
        if old_files:
            print("[!] 正在清理之前生成的错误模型...")
            for f in old_files:
                os.remove(f)
            print(f"[!] 已删除 {len(old_files)} 个旧文件。")

        path_0 = get_final_checkpoint_path(0, args.seed)
        path_20 = get_final_checkpoint_path(20, args.seed)
        
        generated_files = create_final_transplants(path_0, path_20, args.seed)
        
        # --- 自动生成评估脚本 ---
        script_content = "#!/bin/bash\n"
        script_content += "echo '=================================================='\n"
        script_content += "echo '🚀 开始批量评估最终的、正确的移植模型... 🚀'\n"
        script_content += "echo '=================================================='\n\n"
        
        for model_path in generated_files:
            script_content += f"echo '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
            script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90\n"
            
        script_filename = "run_final_evaluation.sh"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        os.chmod(script_filename, 0o755) # 使脚本可执行
        
        print("\n" + "="*60)
        print("🎉 所有正确的渐进式模型已生成完毕！")
        print(f"👉 我已为您自动创建了一个新的评估脚本: ./{script_filename}")
        print("   请直接在终端运行它，以获得我们期待已久的、科学有效的结果。")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)