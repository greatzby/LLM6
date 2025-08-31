# generate_weight_gap_evolution.py

"""
Script to generate weight gap evolution plots across training steps.
Shows how the model learns adjacency information over time.
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
    print("❌ 错误：无法导入'model.py'。请确保该文件与此脚本在同一目录下。")
    exit()

# ==================== 1. 配置与辅助函数 ====================

class Config:
    """配置类，负责加载所有资源。"""
    def __init__(self, seed=42, checkpoint_dir="out_d92"):
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        
        # 数据目录
        self.data_dir = "data/simple_graph/composition_90"
        
        # 加载元数据和图
        self.load_metadata_and_graph()

    def load_metadata_and_graph(self):
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f: 
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']

        graph_path = os.path.join(self.data_dir, 'composition_graph.graphml')
        G = nx.read_graphml(graph_path)
        self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        
        # 生成 Ground Truth 邻接矩阵
        nodelist = sorted(self.G.nodes())
        A_true_90 = nx.to_numpy_array(self.G, nodelist=nodelist)
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        self.A_true[2:92, 2:92] = A_true_90
        print("✓ Ground Truth Adjacency Matrix (A_true) created.")

def get_checkpoint_path(checkpoint_dir, ratio, seed, iteration):
    """获取特定迭代步数的checkpoint路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: 
        raise FileNotFoundError(f"Directory not found: {pattern}")
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt")
    if not os.path.exists(path):
        print(f"⚠️ Warning: Checkpoint not found at {path}")
        return None
    return path

def load_model_and_extract_W_M_prime(checkpoint_path, config):
    """加载模型并提取W'_M矩阵。"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model_args = checkpoint.get('model_args', {})
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            ffn_out_emb = model.transformer.h[0].mlp(token_emb)
            combined_emb = token_emb + ffn_out_emb  # 残差连接
            logits = model.lm_head(combined_emb)
            W_M_prime.append(logits.squeeze().cpu().numpy())
            
    return np.array(W_M_prime)

def calculate_weight_stats(W_M_prime, A_true, path_type):
    """计算指定路径类型的权重统计信息。"""
    if path_type == 'S1->S2':
        rows, cols = np.s_[2:32], np.s_[32:62]
    elif path_type == 'S2->S3':
        rows, cols = np.s_[32:62], np.s_[62:92]
    elif path_type == 'S1->S3':
        rows, cols = np.s_[2:32], np.s_[62:92]
    else:
        raise ValueError("Invalid path_type")

    # 提取对应子矩阵
    W_sub = W_M_prime[rows, cols]
    A_sub = A_true[rows, cols]

    # 创建掩码
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)

    # 计算统计量
    num_edges = np.sum(edge_mask)
    num_non_edges = np.sum(non_edge_mask)
    
    if num_edges > 0:
        avg_edge_weight = np.mean(W_sub[edge_mask])
    else:
        avg_edge_weight = np.nan  # 用NaN表示没有边
        
    if num_non_edges > 0:
        avg_non_edge_weight = np.mean(W_sub[non_edge_mask])
    else:
        avg_non_edge_weight = 0
    
    # 计算gap（只在有边的情况下）
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

def collect_evolution_data(config, mix_ratios=[0, 20], iterations=None):
    """收集所有checkpoint的权重差距演化数据。"""
    if iterations is None:
        iterations = list(range(5000, 55000, 5000))  # 5k到50k
    
    evolution_data = {}
    
    for ratio in mix_ratios:
        print(f"\n📊 Processing {ratio}% Mix Model...")
        model_data = {
            'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
            'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
            'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
        }
        
        for iteration in tqdm(iterations, desc=f"Loading checkpoints"):
            checkpoint_path = get_checkpoint_path(config.checkpoint_dir, ratio, config.seed, iteration)
            if checkpoint_path is None:
                # 如果checkpoint不存在，跳过
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
                continue
            
            # 加载模型并提取W'_M
            W_M_prime = load_model_and_extract_W_M_prime(checkpoint_path, config)
            
            # 计算每个路径类型的统计量
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_weight_stats(W_M_prime, config.A_true, path_type)
                model_data[path_type]['edge'].append(stats['avg_edge_weight'])
                model_data[path_type]['non_edge'].append(stats['avg_non_edge_weight'])
                model_data[path_type]['gap'].append(stats['gap'])
        
        evolution_data[f"{ratio}% Model"] = model_data
    
    return evolution_data, iterations

def plot_evolution(evolution_data, iterations, save_dir, config):
    """绘制权重差距演化图。"""
    # 创建3x3的子图（3个路径类型 x 3个指标）
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Weight Gap Evolution During Training', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'0% Model': 'blue', '20% Model': 'red'}
    
    for i, path_type in enumerate(path_types):
        # 第1列：平均边权重
        ax = axes[i, 0]
        for model_name, model_data in evolution_data.items():
            edge_weights = model_data[path_type]['edge']
            if not all(np.isnan(edge_weights)):  # 只在有数据时绘制
                ax.plot(iterations, edge_weights, marker='o', label=model_name, 
                       color=colors[model_name], linewidth=2, markersize=6)
        ax.set_ylabel(f'{path_type}\nAverage Weight', fontsize=11)
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        if path_type != 'S1->S3':  # S1->S3没有边，不显示legend
            ax.legend(loc='best')
        
        # 第2列：平均非边权重
        ax = axes[i, 1]
        for model_name, model_data in evolution_data.items():
            non_edge_weights = model_data[path_type]['non_edge']
            ax.plot(iterations, non_edge_weights, marker='s', label=model_name,
                   color=colors[model_name], linewidth=2, markersize=6)
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 第3列：权重差距
        ax = axes[i, 2]
        for model_name, model_data in evolution_data.items():
            gaps = model_data[path_type]['gap']
            if path_type != 'S1->S3' and not all(np.isnan(gaps)):  # S1->S3没有gap
                ax.plot(iterations, gaps, marker='^', label=model_name,
                       color=colors[model_name], linewidth=2, markersize=6)
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        if path_type != 'S1->S3':
            ax.legend(loc='best')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 对S1->S3添加注释
        if path_type == 'S1->S3':
            ax.text(0.5, 0.95, 'Note: S1->S3 has no true edges', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 设置x轴标签（只在最后一行）
    for j in range(3):
        axes[2, j].set_xlabel('Training Steps', fontsize=11)
        # 设置x轴刻度
        axes[2, j].set_xticks(iterations[::2])  # 每隔一个显示
        axes[2, j].set_xticklabels([f'{k//1000}k' for k in iterations[::2]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    save_path = os.path.join(save_dir, f'weight_gap_evolution_seed{config.seed}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Evolution plot saved to: {save_path}")
    plt.show()
    
    return save_path

def plot_simplified_evolution(evolution_data, iterations, save_dir, config):
    """绘制简化版的演化图（教授要求的格式）。"""
    # 创建2x2的子图（S1->S2和S2->S3）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Gap Evolution During Training (As Requested)', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3']
    
    for i, path_type in enumerate(path_types):
        # 左列：分别显示边和非边权重
        ax = axes[i, 0]
        for model_name, model_data in evolution_data.items():
            edge_weights = model_data[path_type]['edge']
            non_edge_weights = model_data[path_type]['non_edge']
            
            # 用不同的线型区分边和非边
            color = 'blue' if '0%' in model_name else 'red'
            ax.plot(iterations, edge_weights, marker='o', label=f'{model_name} - Edge',
                   color=color, linewidth=2, markersize=5, linestyle='-')
            ax.plot(iterations, non_edge_weights, marker='s', label=f'{model_name} - Non-Edge',
                   color=color, linewidth=2, markersize=5, linestyle='--', alpha=0.7)
        
        ax.set_ylabel(f'{path_type}\nAverage Weight', fontsize=11)
        ax.set_title('Edge vs Non-Edge Weights', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # 右列：权重差距
        ax = axes[i, 1]
        for model_name, model_data in evolution_data.items():
            gaps = model_data[path_type]['gap']
            color = 'blue' if '0%' in model_name else 'red'
            ax.plot(iterations, gaps, marker='^', label=model_name,
                   color=color, linewidth=2.5, markersize=7)
        
        ax.set_ylabel('Weight Gap', fontsize=11)
        ax.set_title('Weight Gap Evolution', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 设置x轴标签
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel('Training Steps', fontsize=11)
            axes[i, j].set_xticks(iterations[::2])
            axes[i, j].set_xticklabels([f'{k//1000}k' for k in iterations[::2]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    save_path = os.path.join(save_dir, f'weight_gap_evolution_simplified_seed{config.seed}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Simplified plot saved to: {save_path}")
    plt.show()
    
    return save_path

# ==================== 3. 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Generate weight gap evolution plots.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='out_d92', help='Directory containing model checkpoints')
    parser.add_argument('--save_dir', type=str, default='evolution_analysis', help='Directory to save analysis results')
    parser.add_argument('--include_s1s3', action='store_true', help='Include S1->S3 analysis in plots')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results for all checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化配置
    config = Config(seed=args.seed, checkpoint_dir=args.checkpoint_dir)
    config.seed = args.seed
    
    # 收集演化数据
    print("\n" + "="*80)
    print("📈 COLLECTING WEIGHT GAP EVOLUTION DATA...")
    print("="*80)
    
    iterations = list(range(5000, 55000, 5000))
    evolution_data, iterations = collect_evolution_data(config, mix_ratios=[0, 20], iterations=iterations)
    
    # 打印详细的演化数据
    print("\n" + "="*80)
    print("📋 DETAILED EVOLUTION DATA:")
    print("="*80)
    
    for model_name, model_data in evolution_data.items():
        print(f"\n{'='*40}")
        print(f"{model_name}")
        print(f"{'='*40}")
        
        # 创建表格
        print("\nS1->S2 Evolution:")
        print(f"{'Steps':<10} {'Edge Weight':<12} {'Non-Edge':<12} {'Gap':<10}")
        print("-" * 46)
        for i, iter_val in enumerate(iterations):
            edge = model_data['S1->S2']['edge'][i]
            non_edge = model_data['S1->S2']['non_edge'][i]
            gap = model_data['S1->S2']['gap'][i]
            if not np.isnan(gap):
                print(f"{iter_val:<10} {edge:>11.4f} {non_edge:>11.4f} {gap:>9.4f}")
        
        print("\nS2->S3 Evolution:")
        print(f"{'Steps':<10} {'Edge Weight':<12} {'Non-Edge':<12} {'Gap':<10}")
        print("-" * 46)
        for i, iter_val in enumerate(iterations):
            edge = model_data['S2->S3']['edge'][i]
            non_edge = model_data['S2->S3']['non_edge'][i]
            gap = model_data['S2->S3']['gap'][i]
            if not np.isnan(gap):
                print(f"{iter_val:<10} {edge:>11.4f} {non_edge:>11.4f} {gap:>9.4f}")
        
        print("\nS1->S3 Evolution (No true edges):")
        print(f"{'Steps':<10} {'Avg Weight':<12}")
        print("-" * 22)
        for i, iter_val in enumerate(iterations):
            avg_weight = model_data['S1->S3']['non_edge'][i]
            if not np.isnan(avg_weight):
                print(f"{iter_val:<10} {avg_weight:>11.4f}")
    
    # 生成对比表格
    print("\n" + "="*80)
    print("📊 WEIGHT GAP COMPARISON TABLE:")
    print("="*80)
    
    print("\n┌─────────┬─────────────────────────┬─────────────────────────┐")
    print("│  Steps  │      0% Model Gap       │      20% Model Gap      │")
    print("│         │   S1→S2   │   S2→S3    │   S1→S2   │   S2→S3    │")
    print("├─────────┼───────────┼────────────┼───────────┼────────────┤")
    
    for i, iter_val in enumerate(iterations):
        gap_0_s12 = evolution_data['0% Model']['S1->S2']['gap'][i]
        gap_0_s23 = evolution_data['0% Model']['S2->S3']['gap'][i]
        gap_20_s12 = evolution_data['20% Model']['S1->S2']['gap'][i]
        gap_20_s23 = evolution_data['20% Model']['S2->S3']['gap'][i]
        
        if not any(np.isnan([gap_0_s12, gap_0_s23, gap_20_s12, gap_20_s23])):
            print(f"│ {iter_val:>7} │ {gap_0_s12:>9.4f} │ {gap_0_s23:>10.4f} │ {gap_20_s12:>9.4f} │ {gap_20_s23:>10.4f} │")
    
    print("└─────────┴───────────┴────────────┴───────────┴────────────┘")
    
    # 计算并显示趋势
    print("\n" + "="*80)
    print("📈 TREND ANALYSIS:")
    print("="*80)
    
    for path_type in ['S1->S2', 'S2->S3']:
        print(f"\n{path_type} Gap Evolution:")
        for model_name in ['0% Model', '20% Model']:
            gaps = evolution_data[model_name][path_type]['gap']
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            if len(valid_gaps) >= 2:
                initial_gap = valid_gaps[0]
                final_gap = valid_gaps[-1]
                change = final_gap - initial_gap
                change_pct = (change / initial_gap * 100) if initial_gap != 0 else 0
                print(f"  {model_name}: {initial_gap:.4f} → {final_gap:.4f} (Change: {change:+.4f}, {change_pct:+.1f}%)")
    
    print("\nS1->S3 Average Weight Evolution (No true edges):")
    for model_name in ['0% Model', '20% Model']:
        weights = evolution_data[model_name]['S1->S3']['non_edge']
        valid_weights = [w for w in weights if not np.isnan(w)]
        if len(valid_weights) >= 2:
            initial_weight = valid_weights[0]
            final_weight = valid_weights[-1]
            change = final_weight - initial_weight
            print(f"  {model_name}: {initial_weight:.4f} → {final_weight:.4f} (Change: {change:+.4f})")
    
    # 生成图表
    print("\n" + "="*80)
    print("📊 GENERATING PLOTS...")
    print("="*80)
    
    # 生成简化版图表
    plot_simplified_evolution(evolution_data, iterations, args.save_dir, config)
    
    # 如果需要，生成完整版
    if args.include_s1s3:
        plot_evolution(evolution_data, iterations, args.save_dir, config)
    
    # 最终总结
    print("\n" + "="*80)
    print("📋 FINAL RESULTS SUMMARY (at 50k steps):")
    print("="*80)
    
    for model_name, model_data in evolution_data.items():
        print(f"\n{model_name}:")
        for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
            final_gap = model_data[path_type]['gap'][-1]
            final_edge = model_data[path_type]['edge'][-1]
            final_non_edge = model_data[path_type]['non_edge'][-1]
            
            if path_type == 'S1->S3':
                print(f"  {path_type}: No true edges, Avg weight = {final_non_edge:.3f}")
            else:
                print(f"  {path_type}: Gap = {final_gap:.3f} (Edge: {final_edge:.3f}, Non-edge: {final_non_edge:.3f})")
    
    # 保存数据到文件
    # 转换数据为可序列化格式
    save_data = {}
    for model_name, model_data in evolution_data.items():
        save_data[model_name] = {}
        for path_type, path_data in model_data.items():
            save_data[model_name][path_type] = {
                'iterations': iterations,
                'edge': [float(x) if not np.isnan(x) else None for x in path_data['edge']],
                'non_edge': [float(x) if not np.isnan(x) else None for x in path_data['non_edge']],
                'gap': [float(x) if not np.isnan(x) else None for x in path_data['gap']]
            }
    
    json_path = os.path.join(args.save_dir, f'evolution_data_seed{config.seed}.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n💾 Data saved to: {json_path}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()