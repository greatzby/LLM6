# data/simple_graph/prepare_composition.py (修改版)
import os
import pickle
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Prepare composition dataset from a specified directory.')
    # 关键改动：直接接收一个数据目录
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory (e.g., composition_90_alpine_random)')
    # 保留这些参数，因为它们可能被用于文件名或词汇表大小
    parser.add_argument('--total_nodes', type=int, default=90, help='Total number of nodes for vocab size calculation')
    parser.add_argument('--train_paths_per_pair', type=int, default=10, help='Suffix for the train file name (e.g., train_10.txt)')
    args = parser.parse_args()

    # 使用新的 data_dir 参数
    base_dir = args.data_dir
    print(f"Processing dataset in directory: {base_dir}")

    # 文件路径现在基于 data_dir 构建
    train_file_path = os.path.join(base_dir, f'train_{args.train_paths_per_pair}.txt')
    val_file_path = os.path.join(base_dir, 'test.txt') # 通常用 test.txt 作为验证集

    if not os.path.exists(train_file_path):
        print(f"Error: Training file not found at {train_file_path}")
        return
    if not os.path.exists(val_file_path):
        print(f"Error: Validation/Test file not found at {val_file_path}")
        return

    # --- 后续代码与您提供的版本完全相同 ---

    with open(train_file_path, 'r') as f:
        train_data = f.read()
    print(f"Train dataset length: {len(train_data):,} characters")

    with open(val_file_path, 'r') as f:
        val_data = f.read()
    print(f"Val dataset length: {len(val_data):,} characters")

    vocab_size = args.total_nodes + 2
    stoi = {str(i): i + 2 for i in range(args.total_nodes)}
    itos = {i + 2: str(i) for i in range(args.total_nodes)}
    stoi['[PAD]'] = 0
    itos[0] = '[PAD]'
    stoi['\n'] = 1
    itos[1] = '\n'

    def encode(s):
        ss = s.split(" ")
        return [stoi[ch] for ch in ss if ch in stoi]

    def get_max_len(s):
        return max(len(line.split(' ')) for line in s.strip().split('\n'))

    # 计算 block_size，确保能容纳最长的序列 + 2个特殊token (source, target) + 1个换行符
    # 这里的逻辑需要健壮，以处理格式 s t path...
    max_len_train = get_max_len(train_data)
    max_len_val = get_max_len(val_data)
    block_size = max(max_len_train, max_len_val)
    print(f"Max sequence length found: {block_size}")
    
    # 为了对齐，可以向上取整到32的倍数
    block_size = (block_size // 32 + 1) * 32
    print(f"Using block size: {block_size}")

    def process_data(s, block_size):
        lines = s.strip().split('\n')
        ids = []
        for line in lines:
            if line:
                encoded_line = encode(line) + [stoi['\n']] # 添加换行符
                padding = [stoi['[PAD]']] * (block_size - len(encoded_line))
                ids.extend(encoded_line + padding)
        return ids

    train_ids = process_data(train_data, block_size)
    val_ids = process_data(val_data, block_size)

    print(f"Train has {len(train_ids) // block_size} sequences.")
    print(f"Val has {len(val_ids) // block_size} sequences.")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(base_dir, f'train_{args.train_paths_per_pair}.bin'))
    # 将验证集保存为 val.bin，因为训练脚本默认会寻找这个文件
    val_ids.tofile(os.path.join(base_dir, 'val.bin'))

    meta = {
        'unreachable': False,
        'simple_format': True,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("\nDataset preparation complete!")
    print(f"Generated files in: {base_dir}")
    print(f" - train_{args.train_paths_per_pair}.bin")
    print(f" - val.bin")
    print(f" - meta.pkl")

if __name__ == "__main__":
    main()