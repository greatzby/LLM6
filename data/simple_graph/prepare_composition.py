# data/simple_graph/prepare_composition.py
import os
import pickle
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Prepare composition dataset')
parser.add_argument('--experiment_name', type=str, default='composition', help='Experiment name')
parser.add_argument('--total_nodes', type=int, default=90, help='Total number of nodes')
parser.add_argument('--train_paths_per_pair', type=int, default=10, help='Number of paths per pair')
args = parser.parse_args()

# 文件路径
base_dir = os.path.join(os.path.dirname(__file__), f'{args.experiment_name}_{args.total_nodes}')
train_file_path = os.path.join(base_dir, f'train_{args.train_paths_per_pair}.txt')
val_file_path = os.path.join(base_dir, 'test.txt')

# 读取数据
with open(train_file_path, 'r') as f:
    train_data = f.read()
print(f"Train dataset length: {len(train_data):,} characters")

with open(val_file_path, 'r') as f:
    val_data = f.read()
print(f"Val dataset length: {len(val_data):,} characters")

# 词汇表
vocab_size = args.total_nodes + 2
stoi = {}
itos = {}

for i in range(args.total_nodes):
    stoi[str(i)] = i + 2
    itos[i + 2] = str(i)

stoi['[PAD]'] = 0
itos[0] = '[PAD]'
stoi['\n'] = 1
itos[1] = '\n'

# 编码函数
def encode_string(s, stonum):
    ss = s.split(" ")
    return [stonum[ch] for ch in ss if ch in stonum]

def encode(s):
    return encode_string(s, stoi)

def process_reasoning(s, block_size):
    split_text = s.split('\n')
    ret = []
    for st in split_text:
        if st != "":
            enc_str = encode(st) + [1]  # 添加换行符
            # Padding到block_size+1
            ret += enc_str + [0] * (block_size + 1 - len(enc_str))
    return ret

def get_block_size(s):
    split_text = s.split('\n')
    bs = 0
    for st in split_text:
        if st != "":
            enc_str = encode(st) + [1]
            bs = max(bs, len(enc_str))
    return bs

# 计算block size
block_size = (max(get_block_size(train_data), get_block_size(val_data)) // 32 + 1) * 32
print(f"Block size: {block_size}")

# 编码数据
train_ids = process_reasoning(train_data, block_size)
val_ids = process_reasoning(val_data, block_size)

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")

# 保存二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(base_dir, f'train_{args.train_paths_per_pair}.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

# 保存元信息
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

print("Dataset preparation complete!")