# generate_alpine_from_graph.py - 严格遵循ALPINE论文规则
import networkx as nx
import random
import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm

def generate_random_path(G, source, target, reachability_cache):
    """
    生成一条从source到target的随机有效路径。
    这是新实验的核心逻辑。
    
    Args:
        G (nx.DiGraph): 图对象。
        source (str): 起始节点名称。
        target (str): 目标节点名称。
        reachability_cache (dict): 预计算的可达性缓存。
    """
    path = [source]
    current_node = source
    
    # 设置一个最大路径长度以防万一（例如在非DAG图中有环）
    max_path_length = G.number_of_nodes() * 2

    while current_node != target:
        if len(path) > max_path_length:
            # 路径过长，可能陷入了困境，重新尝试
            # 在我们的DAG中这几乎不会发生，但这是健壮的编程习惯
            return generate_random_path(G, source, target, reachability_cache)

        # 获取当前节点的所有直接后继
        successors = list(G.successors(current_node))
        
        # 关键步骤: 筛选出那些既是邻居又能最终到达目标target的节点
        # reachability_cache[target] 是一个包含所有能到达target的节点的集合
        valid_next_steps = [node for node in successors if node in reachability_cache[target]]
        
        if not valid_next_steps:
            # 这种情况在理论上不应该发生，因为我们是从一个已知的可达对开始的
            # 但为了代码的健壮性，如果发生，我们就返回None
            print(f"Warning: Stuck at node {current_node} when trying to reach {target}. No valid next step found.")
            return None 
            
        # 从所有有效的下一步中随机选择一个
        next_node = random.choice(valid_next_steps)
        path.append(next_node)
        current_node = next_node
        
    # 返回一个由整数组成的路径列表，符合您之前的数据格式
    return [int(p) for p in path]

def create_alpine_style_dataset(G, num_train_paths_per_pair, train_split_ratio, stage_info):
    """
    使用ALPINE方法从一个给定的图创建数据集。
    严格遵循ALPINE论文：直接边100%进训练集，其他按比例分配。
    """
    print("\n" + "="*70)
    print("ALPINE STRICT MODE - Following paper exactly")
    print("="*70)
    
    # 1. 预计算所有节点的可达性，这是为了极大地提高路径生成效率
    print("\nStep 1: Pre-computing reachability for all nodes...")
    reachability_cache = {}
    # G.nodes()返回的是字符串节点，这正是我们需要的
    for node in tqdm(G.nodes(), desc="Caching reachability"):
        # nx.ancestors(G, node) 返回所有能到达 node 的祖先节点集合
        ancestors = nx.ancestors(G, node)
        ancestors.add(node) # 节点本身也能"到达"自己
        reachability_cache[node] = ancestors

    # 2. 找出图中所有可达的 (source, target) 对
    print("\nStep 2: Finding all reachable (source, target) pairs...")
    all_reachable_pairs = []
    nodes = list(G.nodes())
    for source in tqdm(nodes, desc="Finding pairs"):
        # 使用预计算的缓存来找目标
        for target in nodes:
            if source != target and source in reachability_cache[target]:
                all_reachable_pairs.append((source, target))
    print(f"Found {len(all_reachable_pairs)} total reachable pairs.")

    # 3. 按ALPINE规则分割（直接边100%训练，其他按比例）
    print(f"\nStep 3: Splitting pairs (ALPINE rules: direct edges→train, others→{train_split_ratio:.0%})...")
    train_pairs = []
    test_pairs = []
    direct_edge_count = 0
    
    for source, target in all_reachable_pairs:
        # ALPINE规则：直接边必须在训练集
        if G.has_edge(source, target):
            train_pairs.append((source, target))
            direct_edge_count += 1
        # 其他对按train_split_ratio分配
        elif random.random() < train_split_ratio:
            train_pairs.append((source, target))
        else:
            test_pairs.append((source, target))
    
    print(f"  Direct edges (forced to training): {direct_edge_count}")
    print(f"  Final split - Training pairs: {len(train_pairs)}, Testing pairs: {len(test_pairs)}")

    # 添加统计信息：分析S1->S3的分布
    if stage_info:
        S1, S2, S3 = stage_info['stages']
        train_s1s2, train_s2s3, train_s1s3 = 0, 0, 0
        test_s1s2, test_s2s3, test_s1s3 = 0, 0, 0
        
        for source, target in train_pairs:
            src, tgt = int(source), int(target)
            if src in S1 and tgt in S2:
                train_s1s2 += 1
            elif src in S2 and tgt in S3:
                train_s2s3 += 1
            elif src in S1 and tgt in S3:
                train_s1s3 += 1
        
        for source, target in test_pairs:
            src, tgt = int(source), int(target)
            if src in S1 and tgt in S2:
                test_s1s2 += 1
            elif src in S2 and tgt in S3:
                test_s2s3 += 1
            elif src in S1 and tgt in S3:
                test_s1s3 += 1
        
        print("\n" + "-"*50)
        print("PAIR DISTRIBUTION (before path generation):")
        print("-"*50)
        print(f"  S1→S2: Train={train_s1s2}, Test={test_s1s2}")
        print(f"  S2→S3: Train={train_s2s3}, Test={test_s2s3}")
        print(f"  S1→S3: Train={train_s1s3}, Test={test_s1s3}")
        
        total_s1s3 = train_s1s3 + test_s1s3
        if total_s1s3 > 0:
            print(f"\n  🔍 KEY METRIC: S1→S3 in training = {train_s1s3}/{total_s1s3} = {train_s1s3/total_s1s3:.1%}")
            print(f"     (vs 0% in original experiment)")
        print("-"*50)

    train_set = []
    test_set = []

    # 4. 为训练集中的每个(source, target)对生成多条随机路径
    print(f"\nStep 4: Generating {num_train_paths_per_pair} random paths for each training pair...")
    
    # 对于直接边，添加直接路径（ALPINE特色）
    direct_paths_added = 0
    for source, target in tqdm(train_pairs, desc="Generating training data"):
        # 如果是直接边，先添加一条直接路径
        if G.has_edge(source, target):
            train_set.append([int(source), int(target), int(source), int(target)])
            direct_paths_added += 1
        
        # 然后生成指定数量的随机路径
        for _ in range(num_train_paths_per_pair):
            path = generate_random_path(G, source, target, reachability_cache)
            if path: # 确保路径成功生成
                train_set.append([int(source), int(target)] + path)
    
    print(f"  Added {direct_paths_added} direct paths for direct edges")

    # 5. 为测试集中的每个(source, target)对生成一条随机路径用于评估
    print("\nStep 5: Generating 1 random path for each testing pair...")
    for source, target in tqdm(test_pairs, desc="Generating testing data"):
        path = generate_random_path(G, source, target, reachability_cache)
        if path:
            test_set.append([int(source), int(target)] + path)
    
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    # 最终统计
    if stage_info:
        train_s1s2_samples, train_s2s3_samples, train_s1s3_samples = 0, 0, 0
        test_s1s2_samples, test_s2s3_samples, test_s1s3_samples = 0, 0, 0
        
        for data in train_set:
            source, target = data[0], data[1]
            if source in S1 and target in S2:
                train_s1s2_samples += 1
            elif source in S2 and target in S3:
                train_s2s3_samples += 1
            elif source in S1 and target in S3:
                train_s1s3_samples += 1
        
        for data in test_set:
            source, target = data[0], data[1]
            if source in S1 and target in S2:
                test_s1s2_samples += 1
            elif source in S2 and target in S3:
                test_s2s3_samples += 1
            elif source in S1 and target in S3:
                test_s1s3_samples += 1
        
        print("\n" + "="*70)
        print("FINAL DATASET STATISTICS")
        print("="*70)
        print(f"Training samples: {len(train_set)}")
        print(f"  S1→S2: {train_s1s2_samples}")
        print(f"  S2→S3: {train_s2s3_samples}")
        print(f"  S1→S3: {train_s1s3_samples}")
        print(f"\nTest samples: {len(test_set)}")
        print(f"  S1→S2: {test_s1s2_samples}")
        print(f"  S2→S3: {test_s2s3_samples}")
        print(f"  S1→S3: {test_s1s3_samples}")
        
        print("\n" + "="*70)
        print("🎯 CRITICAL COMPARISON:")
        print("="*70)
        print(f"Original 0% experiment: 0 S1→S3 paths in training")
        print(f"ALPINE strict mode: {train_s1s3_samples} S1→S3 paths in training")
        print(f"This difference should eliminate S2 transparency!")
        print("="*70)
    
    return train_set, test_set

def format_data(data):
    """将数据格式化为一行字符串"""
    return ' '.join(str(num) for num in data) + '\n'

def write_dataset(dataset, file_name):
    """将数据集写入文件"""
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset using ALPINE's method with strict paper rules")
    parser.add_argument('--input_graph', type=str, required=True, help='Path to the input graph file')
    parser.add_argument('--stage_info', type=str, required=True, help='Path to the stage_info.pkl file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the dataset')
    parser.add_argument('--train_paths_per_pair', type=int, default=20, help="Number of paths per training pair (ALPINE default: 20)")
    parser.add_argument('--train_split_ratio', type=float, default=0.5, help="Training ratio for non-edge pairs (ALPINE default: 0.5)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子以保证结果可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading graph from: {args.input_graph}")
    G = nx.read_graphml(args.input_graph)
    
    print(f"Loading stage info from: {args.stage_info}")
    with open(args.stage_info, 'rb') as f:
        stage_info = pickle.load(f)
        
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 使用ALPINE严格方法创建数据集
    train_set, test_set = create_alpine_style_dataset(
        G, 
        num_train_paths_per_pair=args.train_paths_per_pair,
        train_split_ratio=args.train_split_ratio,
        stage_info=stage_info
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义输出文件路径
    train_filename = os.path.join(args.output_dir, f'train_{args.train_paths_per_pair}.txt')
    test_filename = os.path.join(args.output_dir, 'test.txt')
    graph_filename = os.path.join(args.output_dir, 'composition_graph.graphml')
    stage_info_filename = os.path.join(args.output_dir, 'stage_info.pkl')
    
    # 写入新的训练集和测试集
    write_dataset(train_set, train_filename)
    write_dataset(test_set, test_filename)
    
    # 将原始的图和stage_info文件也复制到新目录中，保持完整性
    nx.write_graphml(G, graph_filename)
    with open(stage_info_filename, 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\n✅ Dataset successfully saved to: {args.output_dir}/")