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
    return path

def analyze_model_keys(path_0, path_20):
    """加载两个模型并深入分析它们的state_dict键。"""
    
    print(f"[*] 正在加载 mix0 模型: {path_0}")
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0['model']
    keys_0 = set(state_0.keys())
    
    print(f"[*] 正在加载 mix20 模型: {path_20}")
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20['model']
    keys_20 = set(state_20.keys())
    
    print("\n" + "="*60)
    print("🔑 密钥分析报告 🔑")
    print("="*60)
    
    print(f"\n[1] 密钥数量统计:")
    print(f"  - mix0 模型有 {len(keys_0)} 个权重键。")
    print(f"  - mix20 模型有 {len(keys_20)} 个权重键。")
    
    if keys_0 == keys_20:
        print("  ✅ 两个模型的权重键完全一致。")
    else:
        print("  🚨 警告：两个模型的权重键不一致！")
        
        # 找出差异
        keys_only_in_0 = keys_0 - keys_20
        keys_only_in_20 = keys_20 - keys_0
        
        if keys_only_in_0:
            print(f"\n  - 只存在于 mix0 的键 ({len(keys_only_in_0)}个):")
            for key in sorted(list(keys_only_in_0)):
                print(f"    {key}")
                
        if keys_only_in_20:
            print(f"\n  - 只存在于 mix20 的键 ({len(keys_only_in_20)}个):")
            for key in sorted(list(keys_only_in_20)):
                print(f"    {key}")

    print("\n" + "="*60)
    print("[2] mix20 模型中 Transformer Block 的所有键 (Ground Truth):")
    print("="*60)
    
    block_keys_found = False
    for key in sorted(list(keys_20)):
        if 'transformer.h.' in key:
            print(f"  {key}")
            block_keys_found = True
            
    if not block_keys_found:
        print("  - 未找到任何 'transformer.h.' 相关的键！")

    print("\n" + "="*60)
    print("分析完成。请将以上输出提供给我，以便最终修复移植脚本。")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断并打印两个模型检查点的state_dict键，以找出差异。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42)')
    args = parser.parse_args()
    
    try:
        path_0 = get_final_checkpoint_path(0, args.seed)
        path_20 = get_final_checkpoint_path(20, args.seed)
        analyze_model_keys(path_0, path_20)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")