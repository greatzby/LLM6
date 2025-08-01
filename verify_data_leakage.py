# ===================================================================
#                      verify_data_leakage.py
#
#  本脚本用于严格验证训练数据中是否存在测试集里的S1->S3泛化路径。
# ===================================================================
import os
import pickle
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify data leakage for compositionality experiments.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing train.bin, test.txt, meta.pkl, etc.")
    args = parser.parse_args()

    print("="*60)
    print("🕵️  开始验证数据泄露...")
    print(f"[*] 数据目录: {args.data_dir}")
    print("="*60)

    # --- 1. 加载元数据和分段信息 ---
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    block_size = meta['block_size']

    with open(os.path.join(args.data_dir, 'stage_info.pkl'), 'rb') as f:
        stages = pickle.load(f)['stages']
    S1, S2, S3 = stages

    # --- 2. 加载训练数据并转换为路径列表 ---
    print("[*] 正在加载并解析训练数据 (train_10.bin)... 这可能需要一点时间。")
    train_data = np.memmap(os.path.join(args.data_dir, 'train_10.bin'), dtype=np.uint16, mode='r')
    
    train_paths = []
    # 假设每个样本是 block_size+1, 我们只关心实际路径部分
    # 注意：这里的解析逻辑需要根据你数据生成的实际情况调整
    # 一个通用的方法是寻找 EOS token (假设为1)
    current_path = []
    for token_id in train_data:
        if token_id == 1: # 假设 1 是 EOS/分隔符
            if len(current_path) > 1:
                train_paths.append(tuple(current_path))
            current_path = []
        elif token_id != 0: # 忽略 padding
            try:
                # 将 token id 转回数字 node
                current_path.append(int(itos[token_id]))
            except (ValueError, KeyError):
                pass # 忽略非数字token
    if current_path:
        train_paths.append(tuple(current_path))
    
    # 使用 set 以获得极快的查找速度
    train_paths_set = set(train_paths)
    print(f"[+] 训练数据中解析出 {len(train_paths_set)} 条独立路径。")


    # --- 3. 加载测试数据并检查S1->S3路径 ---
    print("[*] 正在加载测试数据 (test.txt) 并检查S1->S3路径...")
    with open(os.path.join(args.data_dir, 'test.txt'), 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]

    s1_s3_paths_in_test = []
    for line in test_lines:
        parts = [int(p) for p in line.split()]
        source, target, path = parts[0], parts[1], tuple(parts[2:])
        if source in S1 and target in S3:
            full_path = tuple([source] + list(path) + [target])
            s1_s3_paths_in_test.append(full_path)
    
    print(f"[+] 测试集中共找到 {len(s1_s3_paths_in_test)} 条 S1->S3 路径。")

    # --- 4. 逐一比对，找出泄露 ---
    leaked_paths_count = 0
    for test_path in s1_s3_paths_in_test:
        if test_path in train_paths_set:
            print(f"  🚨 数据泄露警告! 测试路径 {test_path} 在训练数据中被发现!")
            leaked_paths_count += 1
            
    # --- 5. 报告最终结果 ---
    print("\n" + "="*60)
    print("                 🕵️  验证结果 🕵️                  ")
    print("="*60)
    if leaked_paths_count > 0:
        leak_percentage = (leaked_paths_count / len(s1_s3_paths_in_test)) * 100
        print(f"🔴 严重问题: 发现 {leaked_paths_count} 条泄露路径!")
        print(f"🔴 S1->S3 测试用例中，有 {leak_percentage:.2f}% 的路径已在训练集中出现。")
        print("🔴 这解释了为什么0% mix模型表现出超高的泛化能力。")
        print("🔴 实验前提不成立，必须先修复数据生成过程！")
    else:
        print("✅ 恭喜! 未发现数据泄露。")
        print("✅ 0% mix模型的高性能是真实泛化能力，而非数据泄露所致。")
    print("="*60)

if __name__ == "__main__":
    main()