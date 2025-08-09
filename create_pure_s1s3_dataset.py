# ===================================================================
#           create_pure_s1s3_dataset.py
#
# 基于原始图生成逻辑，专门用于创建纯组合 (S1->S3) 数据集的脚本。
# 它确保了生成的数据与原始实验环境100%兼容。
# ===================================================================
import networkx as nx
import random
import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm

def generate_composition_graph(num_nodes_per_stage, edge_prob_within, edge_prob_between, num_stages=3):
    """(此函数与您的原始脚本完全相同，以确保生成完全一致的图)"""
    G = nx.DiGraph()
    total_nodes = num_nodes_per_stage * num_stages
    for i in range(total_nodes): G.add_node(str(i))
    stages = [list(range(s * num_nodes_per_stage, (s + 1) * num_nodes_per_stage)) for s in range(num_stages)]
    for stage_nodes in stages:
        for i in stage_nodes:
            for j in stage_nodes:
                if i < j and random.random() < edge_prob_within: G.add_edge(str(i), str(j))
    for s in range(num_stages - 1):
        for i in stages[s]:
            for j in stages[s+1]:
                if random.random() < edge_prob_between: G.add_edge(str(i), str(j))
    return G, stages

def create_s1s3_only_dataset(G, stages, num_paths=20000, val_split=0.1):
    """专门创建只包含 S1->S3 组合路径的数据集"""
    S1, S2, S3 = stages[0], stages[1], stages[2]
    
    print("[*] 正在从图中寻找所有有效的 S1->S3 配对...")
    s1_s3_pairs = []
    for s1 in tqdm(S1, desc="扫描S1节点"):
        for s3 in S3:
            if nx.has_path(G, str(s1), str(s3)):
                path = nx.shortest_path(G, str(s1), str(s3))
                path_ints = [int(p) for p in path]
                if any(node in S2 for node in path_ints[1:-1]):
                    s1_s3_pairs.append((s1, s3))

    print(f"[*] 找到了 {len(s1_s3_pairs)} 个有效的 S1->S3 配对。")
    if not s1_s3_pairs:
        raise ValueError("图中没有找到任何有效的 S1->S3 路径！请检查图生成参数。")

    print(f"[*] 正在生成 {num_paths} 条 S1->S3 路径用于训练和验证...")
    all_s1s3_paths = []
    for _ in tqdm(range(num_paths), desc="生成路径"):
        source, target = random.choice(s1_s3_pairs)
        path = nx.shortest_path(G, str(source), str(target))
        path_ints = [int(p) for p in path]
        all_s1s3_paths.append([source, target] + path_ints)

    random.shuffle(all_s1s3_paths)
    split_idx = int(len(all_s1s3_paths) * (1 - val_split))
    train_set = all_s1s3_paths[:split_idx]
    val_set = all_s1s3_paths[split_idx:]
    
    print(f"[*] 数据集分割完成: {len(train_set)} (训练), {len(val_set)} (验证)")
    return train_set, val_set

def format_data(data):
    return ' '.join(str(num) for num in data)

def write_dataset_txt(dataset, file_name):
    with open(file_name, "w") as file:
        for data in tqdm(dataset, desc=f"写入 {os.path.basename(file_name)}"):
            file.write(format_data(data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes_per_stage', type=int, default=30)
    parser.add_argument('--edge_prob_within', type=float, default=0.1)
    parser.add_argument('--edge_prob_between', type=float, default=0.3)
    parser.add_argument('--num_paths', type=int, default=20000, help="要生成的 S1->S3 路径总数")
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--experiment_name', type=str, default='composition_90_pure')
    parser.add_argument('--seed', type=int, default=42, help="必须与原始实验的种子一致")
    args = parser.parse_args()
    
    # 确保随机种子与您所有实验一致
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("🚀 开始生成纯组合 (S1->S3) 数据集...")
    print("="*60)
    
    G, stages = generate_composition_graph(args.nodes_per_stage, args.edge_prob_within, args.edge_prob_between)
    train_set, val_set = create_s1s3_only_dataset(G, stages, num_paths=args.num_paths, val_split=args.val_split)
    
    folder_name = os.path.join('data', 'simple_graph', args.experiment_name)
    os.makedirs(folder_name, exist_ok=True)
    
    # 将生成的路径写入 .txt 文件
    write_dataset_txt(train_set, os.path.join(folder_name, 'train_pure.txt'))
    write_dataset_txt(val_set, os.path.join(folder_name, 'val_pure.txt'))
    
    # 保存阶段信息，以便将来参考
    stage_info = {'stages': stages, 'nodes_per_stage': args.nodes_per_stage, 'num_stages': 3}
    with open(os.path.join(folder_name, 'stage_info.pkl'), 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\n[*] .txt 数据集已生成于 '{folder_name}' 目录!")
    print("="*60)