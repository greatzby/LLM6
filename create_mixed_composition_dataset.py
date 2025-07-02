# create_mixed_composition_dataset.py
import networkx as nx
import random
import os
import argparse
import numpy as np
import pickle
import shutil

def create_mixed_dataset(original_dir, s1_s3_ratio=0.05, output_suffix='mixed'):
    """基于原始数据创建混合训练集"""
    
    # 加载原始数据
    print(f"Loading original data from {original_dir}")
    
    # 加载图
    G = nx.read_graphml(os.path.join(original_dir, 'composition_graph.graphml'))
    
    # 加载阶段信息
    with open(os.path.join(original_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    S1, S2, S3 = stages
    
    # 读取原始训练数据
    train_file = os.path.join(original_dir, 'train_10.txt')
    with open(train_file, 'r') as f:
        original_lines = f.readlines()
    
    # 统计原始数据
    s1_s2_count = 0
    s2_s3_count = 0
    
    for line in original_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if source in S1 and target in S2:
                s1_s2_count += 1
            elif source in S2 and target in S3:
                s2_s3_count += 1
    
    print(f"Original training data: S1->S2: {s1_s2_count}, S2->S3: {s2_s3_count}")
    
    # 计算需要添加的S1->S3路径数量
    total_original = s1_s2_count + s2_s3_count
    num_s1_s3_to_add = int(total_original * s1_s3_ratio)
    
    print(f"Adding {num_s1_s3_to_add} S1->S3 paths ({s1_s3_ratio*100:.1f}% of original)")
    
    # 生成S1->S3路径
    s1_s3_pairs = [(s1, s3) for s1 in S1 for s3 in S3 
                   if nx.has_path(G, str(s1), str(s3))]
    
    # 随机选择要添加的配对
    selected_pairs = random.sample(s1_s3_pairs, 
                                  min(num_s1_s3_to_add // 10, len(s1_s3_pairs)))
    
    new_s1_s3_lines = []
    for source, target in selected_pairs:
        # 每个配对生成10条路径（与原始数据一致）
        for _ in range(10):
            path = nx.shortest_path(G, str(source), str(target))
            path_ints = [int(p) for p in path]
            line = ' '.join(str(x) for x in [source, target] + path_ints) + '\n'
            new_s1_s3_lines.append(line)
    
    # 限制到目标数量
    new_s1_s3_lines = new_s1_s3_lines[:num_s1_s3_to_add]
    
    # 创建新目录
    output_dir = f"{original_dir}_{output_suffix}_{int(s1_s3_ratio*100)}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制原始文件
    for file in ['composition_graph.graphml', 'stage_info.pkl', 'test.txt', 'val.bin', 'meta.pkl']:
        src = os.path.join(original_dir, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # 创建混合训练数据
    mixed_lines = original_lines + new_s1_s3_lines
    random.shuffle(mixed_lines)
    
    # 写入新的训练文件
    mixed_train_file = os.path.join(output_dir, 'train_10.txt')
    with open(mixed_train_file, 'w') as f:
        f.writelines(mixed_lines)
    
    # 统计混合数据
    print(f"\nMixed dataset statistics:")
    print(f"  Total paths: {len(mixed_lines)}")
    print(f"  Original S1->S2 + S2->S3: {len(original_lines)}")
    print(f"  Added S1->S3: {len(new_s1_s3_lines)}")
    
    print(f"\nMixed dataset created in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str, default='data/simple_graph/composition_90')
    parser.add_argument('--s1_s3_ratio', type=float, default=0.05, help='Ratio of S1->S3 paths to add')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    create_mixed_dataset(args.original_dir, args.s1_s3_ratio)