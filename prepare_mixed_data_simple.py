# prepare_mixed_data_simple.py
import os
import shutil
import numpy as np
import pickle

def prepare_mixed_dataset(mixed_dir):
    """简单地准备混合数据集"""
    print(f"Preparing {mixed_dir}...")
    
    # 复制原始的meta.pkl
    original_dir = mixed_dir.replace('_mixed_5', '').replace('_mixed_10', '')
    
    # 读取训练数据
    with open(os.path.join(mixed_dir, 'train_10.txt'), 'r') as f:
        lines = f.readlines()
    
    # 加载编码器
    with open(os.path.join(original_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    block_size = meta['block_size']
    
    # 编码
    train_ids = []
    for line in lines:
        if line.strip():
            tokens = [stoi[t] for t in line.strip().split() if t in stoi]
            tokens.append(1)  # EOS
            # Padding
            tokens.extend([0] * (block_size + 1 - len(tokens)))
            train_ids.extend(tokens)
    
    # 保存
    train_ids = np.array(train_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(mixed_dir, 'train_10.bin'))
    
    print(f"  Created train_10.bin: {len(train_ids)} tokens")
    print(f"  Done!")

if __name__ == "__main__":
    # 准备5%数据
    prepare_mixed_dataset('data/simple_graph/composition_90_mixed_5')
    
    # 如果10%数据也创建了
    if os.path.exists('data/simple_graph/composition_90_mixed_10'):
        prepare_mixed_dataset('data/simple_graph/composition_90_mixed_10')