#!/usr/bin/env python3
"""
检查图中的边连接情况，特别是S1→S3的直接边
"""

import networkx as nx
import pickle

# 加载图
print("Loading graph...")
G = nx.read_graphml('data/simple_graph/composition_90_alpine_strict/composition_graph.graphml')

# 加载stage信息
print("Loading stage info...")
with open('data/simple_graph/composition_90_alpine_strict/stage_info.pkl', 'rb') as f:
    stage_info = pickle.load(f)

S1, S2, S3 = stage_info['stages']
S1_set = set(S1)
S2_set = set(S2)
S3_set = set(S3)

print(f"\nStage sizes:")
print(f"  S1: {len(S1)} nodes")
print(f"  S2: {len(S2)} nodes")
print(f"  S3: {len(S3)} nodes")

# 统计各类边
edge_counts = {
    'S1→S1': 0,
    'S1→S2': 0,
    'S1→S3': 0,
    'S2→S1': 0,
    'S2→S2': 0,
    'S2→S3': 0,
    'S3→S1': 0,
    'S3→S2': 0,
    'S3→S3': 0
}

s1_to_s3_edges = []

for edge in G.edges():
    # 确保节点是整数
    if isinstance(edge[0], str):
        source = int(edge[0])
        target = int(edge[1])
    else:
        source = edge[0]
        target = edge[1]
    
    # 分类边
    if source in S1_set:
        if target in S1_set:
            edge_counts['S1→S1'] += 1
        elif target in S2_set:
            edge_counts['S1→S2'] += 1
        elif target in S3_set:
            edge_counts['S1→S3'] += 1
            s1_to_s3_edges.append((source, target))
    elif source in S2_set:
        if target in S1_set:
            edge_counts['S2→S1'] += 1
        elif target in S2_set:
            edge_counts['S2→S2'] += 1
        elif target in S3_set:
            edge_counts['S2→S3'] += 1
    elif source in S3_set:
        if target in S1_set:
            edge_counts['S3→S1'] += 1
        elif target in S2_set:
            edge_counts['S3→S2'] += 1
        elif target in S3_set:
            edge_counts['S3→S3'] += 1

# 打印结果
print("\n" + "="*60)
print("EDGE STATISTICS")
print("="*60)

for edge_type, count in edge_counts.items():
    print(f"{edge_type}: {count} edges")

print("\n" + "="*60)
print("CRITICAL FINDING")
print("="*60)

if edge_counts['S1→S3'] > 0:
    print(f"⚠️ Found {edge_counts['S1→S3']} direct S1→S3 edges!")
    print("\nFirst 10 S1→S3 edges:")
    for i, (s, t) in enumerate(s1_to_s3_edges[:10]):
        print(f"  {s} → {t}")
    if len(s1_to_s3_edges) > 10:
        print(f"  ... and {len(s1_to_s3_edges) - 10} more")
else:
    print("✅ NO direct S1→S3 edges found!")
    print("   All S1→S3 paths must go through S2.")

# 检查S1到S3的可达性（通过任何路径）
print("\n" + "="*60)
print("REACHABILITY CHECK")
print("="*60)

reachable_s1_to_s3 = 0
for s1 in S1[:5]:  # 检查前5个S1节点作为样本
    for s3 in S3[:5]:  # 检查前5个S3节点
        # 转换为字符串（graphml的节点是字符串）
        s1_str = str(s1)
        s3_str = str(s3)
        if nx.has_path(G, s1_str, s3_str):
            reachable_s1_to_s3 += 1
            # 找一条最短路径
            path = nx.shortest_path(G, s1_str, s3_str)
            path_ints = [int(p) for p in path]
            
            # 检查是否经过S2
            middle_nodes = path_ints[1:-1]
            passes_s2 = any(node in S2_set for node in middle_nodes)
            
            print(f"  {s1}→{s3}: path length {len(path)}, passes S2: {passes_s2}")
            if len(path) <= 3:  # 短路径，详细打印
                print(f"    Path: {' → '.join(map(str, path_ints))}")

print(f"\nSample reachability: {reachable_s1_to_s3}/25 pairs reachable")

# 反向边检查
print("\n" + "="*60)
print("REVERSE EDGES CHECK")
print("="*60)
print(f"S2→S1: {edge_counts['S2→S1']} edges")
print(f"S3→S2: {edge_counts['S3→S2']} edges")
print(f"S3→S1: {edge_counts['S3→S1']} edges")

if edge_counts['S2→S1'] + edge_counts['S3→S2'] + edge_counts['S3→S1'] == 0:
    print("✅ No backward edges - this is a proper DAG with forward flow!")