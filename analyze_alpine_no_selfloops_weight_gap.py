#!/usr/bin/env python3
"""
analyze_alpine_no_selfloops_weight_gap.py
分析ALPINE strict no selfloops模型的weight gap
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
from tqdm import tqdm
from datetime import datetime

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

# ==================== 1. 配置类 ====================

class AlpineNoSelfloopsConfig:
    """ALPINE No Selfloops模型配置类"""
    def __init__(self, d_model=92):
        self.d_model = d_model
        self.device = torch.device('cpu')
        
        # 模型参数 - 1层1头
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = d_model
        self.vocab_size = 92
        
        # 数据目录 - ALPINE no selfloops
        self.data_dir = 'data/simple_graph/composition_90_alpine_strict_no_selfloops'
        
        # Checkpoint目录
        self.checkpoint_dir = 'out/composition_20251005_051131'
        
        # 加载节点分组和图结构
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_stage_info(self):
        """加载节点分组信息"""
        stage_info_path = os.path.join(self.data_dir, 'stage_info.pkl')
        
        print(f"  Loading stage_info from: {stage_info_path}")
        
        with open(stage_info_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.S1, self.S2, self.S3 = stage_info['stages']
        
        # 转换为集合
        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)
        
        # 创建节点到token的映射
        self.node_to_token = {node: node + 2 for node in range(90)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}
        
        # S1, S2, S3的token索引
        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]
        
        print(f"  ✓ Loaded stage info:")
        print(f"    S1: {len(self.S1)} nodes")
        print(f"    S2: {len(self.S2)} nodes")
        print(f"    S3: {len(self.S3)} nodes")
    
    def load_graph_structure(self):
        """加载图结构"""
        graph_path = os.path.join(self.data_dir, 'composition_graph.graphml')
        
        print(f"  Loading graph from: {graph_path}")
        
        G = nx.read_graphml(graph_path)
        
        # 确保节点是整数
        if isinstance(list(G.nodes())[0], str):
            self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        else:
            self.G = G
        
        print(f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # 创建邻接矩阵
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        
        for edge in self.G.edges():
            source_token = self.node_to_token[edge[0]]
            target_token = self.node_to_token[edge[1]]
            self.A_true[source_token, target_token] = 1
        
        # 统计各类型边
        self.count_edges()
    
    def count_edges(self):
        """统计各类型边的数量"""
        s1_s2_edges = 0
        s2_s3_edges = 0
        s1_s3_edges = 0
        s1_s1_edges = 0
        s2_s2_edges = 0
        s3_s3_edges = 0
        
        for edge in self.G.edges():
            source, target = edge[0], edge[1]
            if source in self.S1_set:
                if target in self.S1_set:
                    s1_s1_edges += 1
                elif target in self.S2_set:
                    s1_s2_edges += 1
                elif target in self.S3_set:
                    s1_s3_edges += 1
            elif source in self.S2_set:
                if target in self.S2_set:
                    s2_s2_edges += 1
                elif target in self.S3_set:
                    s2_s3_edges += 1
            elif source in self.S3_set and target in self.S3_set:
                s3_s3_edges += 1
        
        print(f"  Edge statistics:")
        print(f"    S1->S2 edges: {s1_s2_edges}")
        print(f"    S2->S3 edges: {s2_s3_edges}")
        print(f"    S1->S3 edges: {s1_s3_edges}")
        print(f"    Self-loops - S1->S1: {s1_s1_edges}, S2->S2: {s2_s2_edges}, S3->S3: {s3_s3_edges}")
        
        if s1_s1_edges + s2_s2_edges + s3_s3_edges == 0:
            print(f"    ✅ No self-loops confirmed!")
        
        self.edge_stats = {
            'S1->S2': s1_s2_edges,
            'S2->S3': s2_s3_edges,
            'S1->S3': s1_s3_edges
        }

def find_checkpoint(iteration, config):
    """查找checkpoint"""
    checkpoint_path = os.path.join(config.checkpoint_dir, f'ckpt_{iteration}.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"    Found checkpoint: {checkpoint_path}")
        return checkpoint_path
    else:
        print(f"    ⚠️ No checkpoint found for iteration {iteration}")
        return None

def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵 - 1层1头版本"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        
        model_args = checkpoint.get('model_args', {})
        if not model_args:
            model_args = {
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'vocab_size': config.vocab_size,
                'block_size': 32,  # ALPINE no selfloops使用32
                'dropout': 0.0,
                'bias': False
            }
        
        model_args['vocab_size'] = config.vocab_size
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(config.device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        
        W_M_prime = []
        with torch.no_grad():
            for i in range(config.vocab_size):
                token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
                ffn_out = model.transformer.h[0].mlp(token_emb)
                combined = token_emb + ffn_out
                logits = model.lm_head(combined)
                W_M_prime.append(logits.squeeze().cpu().numpy()[:config.vocab_size])
        
        return np.array(W_M_prime)
        
    except Exception as e:
        print(f"    Error extracting W_M_prime: {e}")
        raise

def calculate_path_statistics(W_M_prime, config, path_type):
    """计算路径统计"""
    if path_type == 'S1->S2':
        source_tokens = config.S1_tokens
        target_tokens = config.S2_tokens
    elif path_type == 'S2->S3':
        source_tokens = config.S2_tokens
        target_tokens = config.S3_tokens
    elif path_type == 'S1->S3':
        source_tokens = config.S1_tokens
        target_tokens = config.S3_tokens
    else:
        raise ValueError(f"Invalid path_type: {path_type}")
    
    # 提取相关的子矩阵
    W_sub = W_M_prime[np.ix_(source_tokens, target_tokens)]
    A_sub = config.A_true[np.ix_(source_tokens, target_tokens)]
    
    # 创建掩码
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)
    
    stats = {
        'num_edges': np.sum(edge_mask),
        'num_non_edges': np.sum(non_edge_mask)
    }
    
    # 处理有边的情况
    if stats['num_edges'] > 0:
        stats['avg_edge_weight'] = np.mean(W_sub[edge_mask])
        stats['std_edge_weight'] = np.std(W_sub[edge_mask])
    else:
        stats['avg_edge_weight'] = 0
        stats['std_edge_weight'] = 0
    
    # 处理无边的情况
    if stats['num_non_edges'] > 0:
        stats['avg_non_edge_weight'] = np.mean(W_sub[non_edge_mask])
        stats['std_non_edge_weight'] = np.std(W_sub[non_edge_mask])
    else:
        stats['avg_non_edge_weight'] = 0
        stats['std_non_edge_weight'] = 0
    
    # 计算gap
    stats['gap'] = stats['avg_edge_weight'] - stats['avg_non_edge_weight']
    
    return stats

# ==================== 2. 数据收集和可视化 ====================

def collect_evolution_data(config, iterations):
    """收集演化数据"""
    print(f"\n📊 Processing ALPINE No Selfloops model...")
    
    model_data = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    found_checkpoints = 0
    for iteration in tqdm(iterations, desc="Loading checkpoints"):
        checkpoint_path = find_checkpoint(iteration, config)
        
        if checkpoint_path is None:
            for path_type in model_data.keys():
                model_data[path_type]['edge'].append(np.nan)
                model_data[path_type]['non_edge'].append(np.nan)
                model_data[path_type]['gap'].append(np.nan)
            continue
        
        try:
            W_M_prime = extract_W_M_prime(checkpoint_path, config)
            found_checkpoints += 1
            
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_path_statistics(W_M_prime, config, path_type)
                model_data[path_type]['edge'].append(stats['avg_edge_weight'])
                model_data[path_type]['non_edge'].append(stats['avg_non_edge_weight'])
                model_data[path_type]['gap'].append(stats['gap'])
                
        except Exception as e:
            print(f"  ⚠️ Error at iteration {iteration}: {e}")
            for path_type in model_data.keys():
                model_data[path_type]['edge'].append(np.nan)
                model_data[path_type]['non_edge'].append(np.nan)
                model_data[path_type]['gap'].append(np.nan)
    
    print(f"  ✓ Found {found_checkpoints}/{len(iterations)} checkpoints")
    return model_data

def plot_results(model_data, iterations, save_dir):
    """生成结果图"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('ALPINE No Selfloops - Weight Gap Analysis (1L-1H-92D)', 
                 fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'S1->S2': 'blue', 'S2->S3': 'red', 'S1->S3': 'green'}
    
    for i, path_type in enumerate(path_types):
        color = colors[path_type]
        
        # 第1列：Edge权重
        ax = axes[i, 0]
        
        edge_weights = model_data[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='o', color=color,
                   linewidth=2, markersize=5, alpha=0.8)
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color)
        
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=13)
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第2列：Non-edge权重
        ax = axes[i, 1]
        
        non_edge_weights = model_data[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', color=color,
                   linewidth=2, markersize=5, alpha=0.8, linestyle='--')
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color)
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第3列：Weight Gap
        ax = axes[i, 2]
        
        gaps = model_data[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='^', color=color,
                   linewidth=2.5, markersize=7)
            
            # 标注最终值
            final_gap = valid_gaps[-1]
            ax.annotate(f'{final_gap:.3f}', 
                      xy=(valid_iters[-1], final_gap),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=10, color=color, fontweight='bold')
            
            # 检查gap是否始终为正
            if min(valid_gaps) > 0:
                ax.text(0.95, 0.05, '✅ Always positive', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            else:
                ax.text(0.95, 0.05, f'⚠️ Min: {min(valid_gaps):.3f}', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 设置x轴
    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel('Training Iterations', fontsize=11)
            axes[i, j].set_xticks(iterations[::2])
            axes[i, j].set_xticklabels([f'{k//1000}k' for k in iterations[::2]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'alpine_no_selfloops_weight_gap_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.show()

def print_statistics(model_data, iterations, config):
    """打印统计信息"""
    print("\n" + "="*80)
    print("📊 ALPINE NO SELFLOOPS STATISTICS")
    print("="*80)
    
    print(f"\nDataset characteristics:")
    print(f"  • No self-loops (S1→S1, S2→S2, S3→S3 removed)")
    print(f"  • S1→S2 edges: {config.edge_stats['S1->S2']}")
    print(f"  • S2→S3 edges: {config.edge_stats['S2->S3']}")
    print(f"  • S1→S3 edges: {config.edge_stats['S1->S3']}")
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    for path_type in path_types:
        print(f"\n{path_type}:")
        print("-" * 40)
        
        gaps = model_data[path_type]['gap']
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        edge_weights = model_data[path_type]['edge']
        valid_edge = [w for w in edge_weights if not np.isnan(w)]
        
        non_edge_weights = model_data[path_type]['non_edge']
        valid_non_edge = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_gaps:
            print(f"  Gap statistics:")
            print(f"    Initial: {valid_gaps[0]:.4f}")
            print(f"    Final:   {valid_gaps[-1]:.4f}")
            print(f"    Min:     {min(valid_gaps):.4f}")
            print(f"    Max:     {max(valid_gaps):.4f}")
            
            # 特别检查各种路径
            if min(valid_gaps) > 0:
                print(f"    ✅ Always positive!")
            else:
                negative_count = sum(1 for g in valid_gaps if g < 0)
                print(f"    ⚠️ Negative {negative_count}/{len(valid_gaps)} times")
        
        if valid_edge:
            print(f"  Edge weight: {valid_edge[-1]:.4f} (final)")
        
        if valid_non_edge:
            print(f"  Non-edge weight: {valid_non_edge[-1]:.4f} (final)")

    # 特别分析
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS:")
    print("="*80)
    
    # 检查所有gap是否为正
    all_positive = True
    for path_type in path_types:
        gaps = model_data[path_type]['gap']
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        if valid_gaps and min(valid_gaps) <= 0:
            all_positive = False
            break
    
    if all_positive:
        print("✅ ALL WEIGHT GAPS ARE POSITIVE!")
        print("   This indicates strong compositionality without S2 transparency.")
    else:
        print("⚠️ Some weight gaps are negative or zero.")
        print("   Model may have partial compositionality issues.")
    
    # S1->S3的特殊性
    s1s3_gaps = [g for g in model_data['S1->S3']['gap'] if not np.isnan(g)]
    if s1s3_gaps:
        print(f"\n📌 S1→S3 Analysis:")
        print(f"   Final gap: {s1s3_gaps[-1]:.4f}")
        print(f"   This dataset includes S1→S3 paths (~31%), unlike Track A (0%).")

# ==================== 3. 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 ALPINE NO SELFLOOPS WEIGHT GAP ANALYSIS")
    print("="*80)
    
    # 配置
    iterations = list(range(5000, 51000, 5000))
    save_dir = 'alpine_no_selfloops_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  • Dataset: ALPINE strict no selfloops")
    print(f"  • Model: 1 Layer, 1 Head, 92D")
    print(f"  • Checkpoint dir: out/composition_20251005_051131")
    print(f"  • Iterations: {iterations}")
    print(f"  • Output directory: {save_dir}")
    
    # 初始化配置
    print("\n" + "="*60)
    print("Initializing configuration...")
    print("="*60)
    
    config = AlpineNoSelfloopsConfig(d_model=92)
    
    # 收集数据
    print("\n" + "="*60)
    print("Collecting data from checkpoints...")
    print("="*60)
    
    model_data = collect_evolution_data(config, iterations)
    
    # 保存原始数据
    with open(os.path.join(save_dir, 'alpine_no_selfloops_data.pkl'), 'wb') as f:
        pickle.dump({
            'model_data': model_data,
            'iterations': iterations,
            'edge_stats': config.edge_stats
        }, f)
    print(f"✅ Raw data saved to: {save_dir}/alpine_no_selfloops_data.pkl")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    
    plot_results(model_data, iterations, save_dir)
    
    # 打印统计
    print_statistics(model_data, iterations, config)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {save_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()