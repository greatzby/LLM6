# create_graph_composition_final.py
import networkx as nx
import random
import os
import argparse
import numpy as np
import pickle

def generate_composition_graph(num_nodes_per_stage, edge_prob_within, edge_prob_between, num_stages=3):
    """创建分层有向图"""
    G = nx.DiGraph()
    total_nodes = num_nodes_per_stage * num_stages
    
    # 添加所有节点
    for i in range(total_nodes):
        G.add_node(str(i))  # 注意：使用字符串作为节点名
    
    # 定义阶段
    stages = []
    for s in range(num_stages):
        stage_nodes = list(range(s * num_nodes_per_stage, (s + 1) * num_nodes_per_stage))
        stages.append(stage_nodes)
    
    # 添加边
    # 阶段内的边
    for stage_nodes in stages:
        for i in stage_nodes:
            for j in stage_nodes:
                if i < j and random.random() < edge_prob_within:
                    G.add_edge(str(i), str(j))
    
    # 阶段间的边（只从Si到Si+1）
    for s in range(num_stages - 1):
        current_stage = stages[s]
        next_stage = stages[s + 1]
        for i in current_stage:
            for j in next_stage:
                if random.random() < edge_prob_between:
                    G.add_edge(str(i), str(j))
    
    return G, stages

def create_composition_dataset(G, stages, test_samples_per_type=50, train_paths_per_pair=10):
    """创建平衡的数据集"""
    train_set = []
    test_set = []
    
    S1, S2, S3 = stages[0], stages[1], stages[2]
    
    # 收集所有可达的配对
    s1_s2_pairs = [(s1, s2) for s1 in S1 for s2 in S2 
                   if nx.has_path(G, str(s1), str(s2))]
    s2_s3_pairs = [(s2, s3) for s2 in S2 for s3 in S3 
                   if nx.has_path(G, str(s2), str(s3))]
    s1_s3_pairs = [(s1, s3) for s1 in S1 for s3 in S3 
                   if nx.has_path(G, str(s1), str(s3))]
    
    print(f"Available pairs: S1->S2: {len(s1_s2_pairs)}, S2->S3: {len(s2_s3_pairs)}, S1->S3: {len(s1_s3_pairs)}")
    
    # 训练集：只包含S1->S2和S2->S3
    print("\nGenerating training set...")
    for source, target in s1_s2_pairs:
        for _ in range(train_paths_per_pair):
            path = nx.shortest_path(G, str(source), str(target))
            path_ints = [int(p) for p in path]
            train_set.append([source, target] + path_ints)
    
    for source, target in s2_s3_pairs:
        for _ in range(train_paths_per_pair):
            path = nx.shortest_path(G, str(source), str(target))
            path_ints = [int(p) for p in path]
            train_set.append([source, target] + path_ints)
    
    # 测试集：平衡的三种类型
    print("\nGenerating balanced test set...")
    
    # S1->S2测试
    random.shuffle(s1_s2_pairs)
    for i in range(min(test_samples_per_type, len(s1_s2_pairs))):
        source, target = s1_s2_pairs[i]
        path = nx.shortest_path(G, str(source), str(target))
        path_ints = [int(p) for p in path]
        test_set.append([source, target] + path_ints)
    
    # S2->S3测试
    random.shuffle(s2_s3_pairs)
    for i in range(min(test_samples_per_type, len(s2_s3_pairs))):
        source, target = s2_s3_pairs[i]
        path = nx.shortest_path(G, str(source), str(target))
        path_ints = [int(p) for p in path]
        test_set.append([source, target] + path_ints)
    
    # S1->S3测试（组合能力）
    random.shuffle(s1_s3_pairs)
    for i in range(min(test_samples_per_type, len(s1_s3_pairs))):
        source, target = s1_s3_pairs[i]
        path = nx.shortest_path(G, str(source), str(target))
        path_ints = [int(p) for p in path]
        # 验证路径经过S2
        has_s2 = any(node in S2 for node in path_ints[1:-1])
        if has_s2:
            test_set.append([source, target] + path_ints)
    
    random.shuffle(train_set)
    
    return train_set, test_set

def format_data(data):
    """格式化为字符串"""
    return ' '.join(str(num) for num in data) + '\n'

def write_dataset(dataset, file_name):
    """写入文件"""
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes_per_stage', type=int, default=30)
    parser.add_argument('--edge_prob_within', type=float, default=0.1)
    parser.add_argument('--edge_prob_between', type=float, default=0.3)
    parser.add_argument('--train_paths_per_pair', type=int, default=10)
    parser.add_argument('--test_samples_per_type', type=int, default=50)
    parser.add_argument('--experiment_name', type=str, default='composition')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    total_nodes = args.nodes_per_stage * 3
    print(f"Creating graph with {total_nodes} nodes")
    
    # 生成图
    G, stages = generate_composition_graph(
        args.nodes_per_stage, 
        args.edge_prob_within, 
        args.edge_prob_between
    )
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 创建数据集
    train_set, test_set = create_composition_dataset(
        G, stages, 
        test_samples_per_type=args.test_samples_per_type,
        train_paths_per_pair=args.train_paths_per_pair
    )
    
    # 统计
    train_stats = {'S1->S2': 0, 'S2->S3': 0}
    test_stats = {'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0}
    
    S1, S2, S3 = stages
    
    for data in train_set:
        if data[0] in S1 and data[1] in S2:
            train_stats['S1->S2'] += 1
        elif data[0] in S2 and data[1] in S3:
            train_stats['S2->S3'] += 1
    
    for data in test_set:
        if data[0] in S1 and data[1] in S2:
            test_stats['S1->S2'] += 1
        elif data[0] in S2 and data[1] in S3:
            test_stats['S2->S3'] += 1
        elif data[0] in S1 and data[1] in S3:
            test_stats['S1->S3'] += 1
    
    print(f"\nTraining set: {train_stats}")
    print(f"Test set: {test_stats}")
    
    # 保存
    folder_name = f'{args.experiment_name}_{total_nodes}'
    os.makedirs(folder_name, exist_ok=True)
    
    write_dataset(train_set, os.path.join(folder_name, f'train_{args.train_paths_per_pair}.txt'))
    write_dataset(test_set, os.path.join(folder_name, 'test.txt'))
    nx.write_graphml(G, os.path.join(folder_name, 'composition_graph.graphml'))
    
    # 保存阶段信息
    stage_info = {
        'stages': stages,
        'nodes_per_stage': args.nodes_per_stage,
        'num_stages': 3
    }
    with open(os.path.join(folder_name, 'stage_info.pkl'), 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\nDataset saved to {folder_name}/")