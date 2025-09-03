# analyze_alpine_weights_detailed.py

"""
Detailed weight gap evolution analysis for ALPINE experiments.
Shows edge weights, non-edge weights, and gaps with comprehensive visualization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
import argparse
from tqdm import tqdm
import json

# 确保模型定义文件存在
try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'. Please ensure it's in the same directory.")
    exit()

# ==================== 1. 配置与辅助函数 ====================

class Config:
    """配置类，负责加载所有资源。"""
    def __init__(self, experiment_type="ALPINE", checkpoint_dir=None):
        self.experiment_type = experiment_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        
        # 根据实验类型设置路径
        if experiment_type == "ALPINE":
            self.data_dir = "data/simple_graph/composition_90_alpine_strict"
            self.checkpoint_dir = checkpoint_dir or "out"
        else:  # "0%" original
            self.data_dir = "data/simple_graph/composition_90"
            self.checkpoint_dir = checkpoint_dir or "out_d92"
        
        # 加载元数据和图
        self.load_metadata_and_graph()

    def load_metadata_and_graph(self):
        """加载元数据和图结构"""
        meta_paths = [
            os.path.join(self.data_dir, 'meta.pkl'),
            os.path.join(self.data_dir, '../composition_90/meta.pkl'),
        ]
        
        meta_loaded = False
        for meta_path in meta_paths:
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f: 
                    meta = pickle.load(f)
                self.vocab_size = meta['vocab_size']
                meta_loaded = True
                print(f"✓ Loaded meta from: {meta_path}")
                break
        
        if not meta_loaded:
            print("⚠️ Warning: meta.pkl not found, using default vocab_size=92")
            self.vocab_size = 92

        graph_paths = [
            os.path.join(self.data_dir, 'composition_graph.graphml'),
            os.path.join(self.data_dir, '../composition_90/composition_graph.graphml'),
        ]
        
        graph_loaded = False
        for graph_path in graph_paths:
            if os.path.exists(graph_path):
                G = nx.read_graphml(graph_path)
                self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
                graph_loaded = True
                print(f"✓ Loaded graph from: {graph_path}")
                break
        
        if not graph_loaded:
            raise FileNotFoundError("Cannot find composition_graph.graphml")
        
        # 生成 Ground Truth 邻接矩阵
        nodelist = sorted(self.G.nodes())
        A_true_90 = nx.to_numpy_array(self.G, nodelist=nodelist)
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        if self.vocab_size >= 92:
            self.A_true[2:92, 2:92] = A_true_90
        else:
            self.A_true = A_true_90[:self.vocab_size, :self.vocab_size]
        print(f"✓ Ground Truth Adjacency Matrix created (vocab_size={self.vocab_size})")

def find_checkpoint(checkpoint_dir, experiment_type, iteration):
    """查找checkpoint文件"""
    if experiment_type == "ALPINE":
        patterns = [
            os.path.join(checkpoint_dir, f"ckpt_{iteration}.pt"),
            os.path.join(checkpoint_dir, f"ckpt_*_{iteration}.pt"),
        ]
    else:
        patterns = [
            os.path.join(checkpoint_dir, f"composition_mix0_seed42_*/ckpt_mix0_seed42_iter{iteration}.pt"),
        ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    
    return None

def load_model_and_extract_W_M_prime(checkpoint_path, config):
    """加载模型并提取W'_M矩阵"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    
    model_args = checkpoint.get('model_args', {})
    if not model_args:
        model_args = {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size,
            'block_size': 512,
            'dropout': 0.0,
            'bias': False
        }
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            ffn_out_emb = model.transformer.h[0].mlp(token_emb)
            combined_emb = token_emb + ffn_out_emb
            logits = model.lm_head(combined_emb)
            W_M_prime.append(logits.squeeze().cpu().numpy())
            
    return np.array(W_M_prime)

def calculate_weight_stats(W_M_prime, A_true, path_type):
    """计算指定路径类型的权重统计信息"""
    if path_type == 'S1->S2':
        rows, cols = np.s_[2:32], np.s_[32:62]
    elif path_type == 'S2->S3':
        rows, cols = np.s_[32:62], np.s_[62:92]
    elif path_type == 'S1->S3':
        rows, cols = np.s_[2:32], np.s_[62:92]
    else:
        raise ValueError("Invalid path_type")

    W_sub = W_M_prime[rows, cols]
    A_sub = A_true[rows, cols]

    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)

    num_edges = np.sum(edge_mask)
    num_non_edges = np.sum(non_edge_mask)
    
    if num_edges > 0:
        avg_edge_weight = np.mean(W_sub[edge_mask])
    else:
        avg_edge_weight = np.nan
        
    if num_non_edges > 0:
        avg_non_edge_weight = np.mean(W_sub[non_edge_mask])
    else:
        avg_non_edge_weight = 0
    
    if num_edges > 0:
        gap = avg_edge_weight - avg_non_edge_weight
    else:
        gap = np.nan
    
    return {
        'avg_edge_weight': avg_edge_weight,
        'avg_non_edge_weight': avg_non_edge_weight,
        'gap': gap,
        'num_edges': num_edges,
        'num_non_edges': num_non_edges
    }

# ==================== 2. 主要分析函数 ====================

def collect_evolution_data(config, iterations=None):
    """收集演化数据"""
    if iterations is None:
        iterations = list(range(1000, 51000, 1000))
    
    model_data = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    print(f"\n📊 Processing {config.experiment_type} experiment...")
    found_checkpoints = 0
    
    for iteration in tqdm(iterations, desc=f"Loading checkpoints"):
        checkpoint_path = find_checkpoint(config.checkpoint_dir, config.experiment_type, iteration)
        
        if checkpoint_path is None:
            for path_type in model_data.keys():
                model_data[path_type]['edge'].append(np.nan)
                model_data[path_type]['non_edge'].append(np.nan)
                model_data[path_type]['gap'].append(np.nan)
            continue
        
        found_checkpoints += 1
        
        try:
            W_M_prime = load_model_and_extract_W_M_prime(checkpoint_path, config)
            
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_weight_stats(W_M_prime, config.A_true, path_type)
                model_data[path_type]['edge'].append(stats['avg_edge_weight'])
                model_data[path_type]['non_edge'].append(stats['avg_non_edge_weight'])
                model_data[path_type]['gap'].append(stats['gap'])
                
        except Exception as e:
            print(f"  ⚠️ Error loading {checkpoint_path}: {e}")
            for path_type in model_data.keys():
                model_data[path_type]['edge'].append(np.nan)
                model_data[path_type]['non_edge'].append(np.nan)
                model_data[path_type]['gap'].append(np.nan)
    
    print(f"  ✓ Found {found_checkpoints}/{len(iterations)} checkpoints")
    return model_data, iterations

def plot_detailed_evolution(model_data, iterations, save_dir, experiment_name):
    """绘制详细的演化图（3x3布局）"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'{experiment_name} - Weight Gap Evolution During Training', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    
    for i, path_type in enumerate(path_types):
        # 第1列：平均边权重
        ax = axes[i, 0]
        edge_weights = model_data[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights and path_type != 'S1->S3':  # S1->S3没有真实边
            ax.plot(valid_iters, valid_weights, marker='o', linewidth=2, markersize=6, color='blue')
        ax.set_ylabel(f'{path_type}\nAverage Weight', fontsize=11)
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第2列：平均非边权重
        ax = axes[i, 1]
        non_edge_weights = model_data[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', linewidth=2, markersize=6, color='red')
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 第3列：权重差距
        ax = axes[i, 2]
        gaps = model_data[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps and path_type != 'S1->S3':  # S1->S3没有gap
            ax.plot(valid_iters, valid_gaps, marker='^', linewidth=2, markersize=6, color='green')
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # S1->S3注释
        if path_type == 'S1->S3':
            axes[i, 0].text(0.5, 0.5, 'No true edges', transform=axes[i, 0].transAxes, 
                          ha='center', va='center', fontsize=14, color='gray')
            axes[i, 2].text(0.5, 0.5, 'N/A', transform=axes[i, 2].transAxes, 
                          ha='center', va='center', fontsize=14, color='gray')
    
    # 设置x轴标签
    for j in range(3):
        axes[2, j].set_xlabel('Training Steps', fontsize=11)
        step = max(1, len(valid_iters) // 10) if valid_iters else 1
        if valid_iters:
            axes[2, j].set_xticks(valid_iters[::step])
            axes[2, j].set_xticklabels([f'{k//1000}k' for k in valid_iters[::step]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(save_dir, f'{experiment_name.lower()}_detailed_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed plot saved to: {save_path}")
    
    return save_path

def plot_simplified_evolution(model_data, iterations, save_dir, experiment_name):
    """绘制简化版演化图（2x2布局）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{experiment_name} - Weight Gap Evolution (Simplified)', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3']
    
    for i, path_type in enumerate(path_types):
        # 左列：Edge和Non-edge权重
        ax = axes[i, 0]
        edge_weights = model_data[path_type]['edge']
        non_edge_weights = model_data[path_type]['non_edge']
        
        valid_iters_edge = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights_edge = [w for w in edge_weights if not np.isnan(w)]
        valid_iters_nonedge = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights_nonedge = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights_edge:
            ax.plot(valid_iters_edge, valid_weights_edge, marker='o', label='Edge',
                   color='blue', linewidth=2, markersize=5)
        if valid_weights_nonedge:
            ax.plot(valid_iters_nonedge, valid_weights_nonedge, marker='s', label='Non-Edge',
                   color='red', linewidth=2, markersize=5, linestyle='--', alpha=0.7)
        
        ax.set_ylabel(f'{path_type}\nAverage Weight', fontsize=11)
        ax.set_title('Edge vs Non-Edge Weights', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # 右列：Weight Gap
        ax = axes[i, 1]
        gaps = model_data[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='^', color='green', linewidth=2.5, markersize=7)
            
            # 添加关键点标注
            if valid_gaps:
                # 标注最小值和最大值
                min_gap = min(valid_gaps)
                max_gap = max(valid_gaps)
                min_idx = valid_gaps.index(min_gap)
                max_idx = valid_gaps.index(max_gap)
                
                ax.annotate(f'Min: {min_gap:.3f}', 
                          xy=(valid_iters[min_idx], min_gap),
                          xytext=(5, -10), textcoords='offset points',
                          fontsize=8, color='red')
                ax.annotate(f'Max: {max_gap:.3f}', 
                          xy=(valid_iters[max_idx], max_gap),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, color='green')
        
        ax.set_ylabel('Weight Gap (Edge - Non-edge)', fontsize=11)
        ax.set_title('Weight Gap Evolution', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加说明文字
        if path_type == 'S2->S3' and valid_gaps and min(valid_gaps) > 0:
            ax.text(0.95, 0.05, '✅ Always positive!', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 设置x轴
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel('Training Steps', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(save_dir, f'{experiment_name.lower()}_simplified_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Simplified plot saved to: {save_path}")
    
    return save_path

def print_detailed_table(model_data, iterations, experiment_name):
    """打印详细的表格"""
    print("\n" + "="*80)
    print(f"📋 DETAILED EVOLUTION DATA - {experiment_name}:")
    print("="*80)
    
    # S1->S2表格
    print("\n" + "="*60)
    print("S1→S2 Evolution:")
    print("="*60)
    print(f"{'Steps':<10} {'Edge Weight':<12} {'Non-Edge':<12} {'Gap':<10}")
    print("-" * 46)
    
    for i, iter_val in enumerate(iterations):
        edge = model_data['S1->S2']['edge'][i]
        non_edge = model_data['S1->S2']['non_edge'][i]
        gap = model_data['S1->S2']['gap'][i]
        if not np.isnan(edge):
            print(f"{iter_val:<10} {edge:>11.4f} {non_edge:>11.4f} {gap:>9.4f}")
    
    # S2->S3表格
    print("\n" + "="*60)
    print("S2→S3 Evolution:")
    print("="*60)
    print(f"{'Steps':<10} {'Edge Weight':<12} {'Non-Edge':<12} {'Gap':<10}")
    print("-" * 46)
    
    for i, iter_val in enumerate(iterations):
        edge = model_data['S2->S3']['edge'][i]
        non_edge = model_data['S2->S3']['non_edge'][i]
        gap = model_data['S2->S3']['gap'][i]
        if not np.isnan(edge):
            print(f"{iter_val:<10} {edge:>11.4f} {non_edge:>11.4f} {gap:>9.4f}")
    
    # S1->S3表格（只有非边权重）
    print("\n" + "="*60)
    print("S1→S3 Evolution (No true edges in graph):")
    print("="*60)
    print(f"{'Steps':<10} {'Avg Weight':<12}")
    print("-" * 22)
    
    for i, iter_val in enumerate(iterations):
        avg_weight = model_data['S1->S3']['non_edge'][i]
        if not np.isnan(avg_weight):
            print(f"{iter_val:<10} {avg_weight:>11.4f}")

# ==================== 3. 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Detailed weight gap analysis for ALPINE experiments.")
    parser.add_argument('--alpine_dir', type=str, required=True, 
                       help='Directory containing ALPINE checkpoints')
    parser.add_argument('--save_dir', type=str, default='alpine_detailed_analysis', 
                       help='Directory to save analysis results')
    parser.add_argument('--max_iter', type=int, default=50000, 
                       help='Maximum iteration to analyze')
    parser.add_argument('--step', type=int, default=5000, 
                       help='Step size between iterations')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔬 ALPINE DETAILED WEIGHT GAP ANALYSIS")
    print("="*80)
    print(f"Checkpoint directory: {args.alpine_dir}")
    print(f"Output directory: {args.save_dir}")
    
    # 初始化配置
    config = Config(experiment_type="ALPINE", checkpoint_dir=args.alpine_dir)
    
    # 设置迭代范围
    iterations = list(range(args.step, args.max_iter + 1, args.step))
    
    # 收集演化数据
    print("\n" + "="*80)
    print("📈 COLLECTING EVOLUTION DATA...")
    print("="*80)
    
    model_data, iterations = collect_evolution_data(config, iterations)
    
    # 打印详细表格
    print_detailed_table(model_data, iterations, "ALPINE")
    
    # 计算关键统计
    print("\n" + "="*80)
    print("📊 KEY STATISTICS:")
    print("="*80)
    
    for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
        print(f"\n{path_type}:")
        
        if path_type != 'S1->S3':
            # 有边的情况
            gaps = model_data[path_type]['gap']
            edge_weights = model_data[path_type]['edge']
            non_edge_weights = model_data[path_type]['non_edge']
            
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            valid_edges = [e for e in edge_weights if not np.isnan(e)]
            valid_non_edges = [n for n in non_edge_weights if not np.isnan(n)]
            
            if valid_gaps:
                print(f"  Gap Range: [{min(valid_gaps):.4f}, {max(valid_gaps):.4f}]")
                print(f"  Final Gap: {valid_gaps[-1]:.4f}")
                print(f"  Edge Weight Range: [{min(valid_edges):.4f}, {max(valid_edges):.4f}]")
                print(f"  Non-edge Weight Range: [{min(valid_non_edges):.4f}, {max(valid_non_edges):.4f}]")
                
                if path_type == 'S2->S3' and min(valid_gaps) > 0:
                    print(f"  ✅ Gap is ALWAYS POSITIVE - S2 information preserved!")
                elif path_type == 'S2->S3':
                    print(f"  ⚠️ Gap had negative values - S2 might be suppressed at times")
        else:
            # S1->S3只有非边
            non_edge_weights = model_data[path_type]['non_edge']
            valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
            
            if valid_weights:
                print(f"  Average Weight Range: [{min(valid_weights):.4f}, {max(valid_weights):.4f}]")
                print(f"  Final Weight: {valid_weights[-1]:.4f}")
    
    # 生成图表
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS...")
    print("="*80)
    
    # 生成详细图（3x3）
    plot_detailed_evolution(model_data, iterations, args.save_dir, "ALPINE")
    
    # 生成简化图（2x2）
    plot_simplified_evolution(model_data, iterations, args.save_dir, "ALPINE")
    
    # 保存数据到JSON
    save_data = {
        'experiment': 'ALPINE',
        'iterations': iterations,
        'S1->S2': {
            'edge': [float(x) if not np.isnan(x) else None for x in model_data['S1->S2']['edge']],
            'non_edge': [float(x) if not np.isnan(x) else None for x in model_data['S1->S2']['non_edge']],
            'gap': [float(x) if not np.isnan(x) else None for x in model_data['S1->S2']['gap']]
        },
        'S2->S3': {
            'edge': [float(x) if not np.isnan(x) else None for x in model_data['S2->S3']['edge']],
            'non_edge': [float(x) if not np.isnan(x) else None for x in model_data['S2->S3']['non_edge']],
            'gap': [float(x) if not np.isnan(x) else None for x in model_data['S2->S3']['gap']]
        },
        'S1->S3': {
            'edge': [float(x) if not np.isnan(x) else None for x in model_data['S1->S3']['edge']],
            'non_edge': [float(x) if not np.isnan(x) else None for x in model_data['S1->S3']['non_edge']],
            'gap': [float(x) if not np.isnan(x) else None for x in model_data['S1->S3']['gap']]
        }
    }
    
    json_path = os.path.join(args.save_dir, 'alpine_detailed_data.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n💾 Data saved to: {json_path}")
    
    # 最终总结
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY:")
    print("="*80)
    
    # 检查S2->S3的gap
    s23_gaps = model_data['S2->S3']['gap']
    valid_s23_gaps = [g for g in s23_gaps if not np.isnan(g)]
    
    if valid_s23_gaps and min(valid_s23_gaps) > 0:
        print("\n✅ SUCCESS: S2→S3 weight gap is ALWAYS POSITIVE!")
        print("   This confirms that ALPINE training preserves S2 information.")
        print("   The model learns to use S2 as an intermediate step for S1→S3 paths.")
    else:
        print("\n⚠️ OBSERVATION: S2→S3 weight gap had some variations.")
        print("   Further analysis may be needed.")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()