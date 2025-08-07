import torch
import os
import argparse
import glob

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于查找模型路径。"""
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

def print_all_keys(model_path):
    """
    加载一个模型并打印其 state_dict 中的所有键，不做任何过滤。
    这是我们需要的最终的、完整的“地面实况”。
    """
    print(f"[*] 正在加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    
    print("\n" + "="*60)
    print(f"🔑 模型 '{os.path.basename(model_path)}' 的完整密钥列表 🔑")
    print(f"   (总共 {len(state_dict.keys())} 个)")
    print("="*60)
    
    # 按照字母顺序打印所有键，以便查看
    for key in sorted(state_dict.keys()):
        print(key)
        
    print("="*60)
    print("诊断完成。请将这份完整的列表提供给我。")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="终极诊断工具：打印模型中所有的state_dict键。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42)')
    parser.add_argument('--ratio', type=int, default=20, help='要检查的模型比例 (0 或 20)')
    args = parser.parse_args()
    
    try:
        path = get_final_checkpoint_path(args.ratio, args.seed)
        print_all_keys(path)
    except FileNotFoundError as e:
        print(f"\n🚨 错误: {e}")
        exit(1)