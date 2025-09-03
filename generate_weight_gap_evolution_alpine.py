# generate_weight_gap_evolution_alpine.py

"""
Script to generate weight gap evolution plots for ALPINE experiments.
Compares original 0% training with ALPINE training method.
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
        
        # 模型参数（两个实验都使用相同配置）
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
        # 尝试多个可能的meta文件位置
        meta_paths = [
            os.path.join(self.data_dir, 'meta.pkl'),
            os.path.join(self.data_dir, '../composition_90/meta.pkl'),  # 如果ALPINE使用原始meta
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
            print("⚠️ Warning: meta.pkl not found, using default vocab_size=94")
            self.vocab_size = 94  # 默认值

        # 加载图
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
        self.A_true[2:92, 2:92] = A_true_90
        print(f"✓ Ground Truth Adjacency Matrix created (vocab_size={self.vocab_size})")

def find_checkpoint(checkpoint_dir, experiment_type, iteration):
    """查找checkpoint文件（更灵活的查找方式）"""
    # 对于ALPINE实验
    if experiment_type == "ALPINE":
        # 方式1: 直接在checkpoint_dir下
        patterns = [
            os.path.join(checkpoint_dir, f"ckpt_{iteration}.pt"),
            os.path.join(checkpoint_dir, f"ckpt_*_{iteration}.pt"),
            os.path.join(checkpoint_dir, f"**/ckpt_{iteration}.pt"),
        ]
        
        # 如果checkpoint_dir包含时间戳目录
        if not os.path.exists(checkpoint_dir):
            # 尝试找最新的输出目录
            parent_dir = os.path.dirname(checkpoint_dir) if os.path.dirname(checkpoint_dir) else "out"
            subdirs = glob.glob(os.path.join(parent_dir, "composition_*"))
            if subdirs:
                checkpoint_dir = sorted(subdirs)[-1]
                print(f"  Using directory: {checkpoint_dir}")
    
    else:  # Original 0% experiment
        patterns = [
            os.path.join(checkpoint_dir, f"composition_mix0_seed42_*/ckpt_mix0_seed42_iter{iteration}.pt"),
            os.path.join(checkpoint_dir, f"*/ckpt_*{iteration}.pt"),
        ]
    
    # 搜索checkpoint
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    
    return None

def load_model_and_extract_W_M_prime(checkpoint_path, config):
    """加载模型并提取W'_M矩阵（您的计算方式是正确的！）"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    
    # 获取模型配置
    model_args = checkpoint.get('model_args', {})
    if not model_args:
        # 如果没有model_args，使用默认配置
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

    # 提取W'_M矩阵 - 您的方法完全正确！
    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            # token embedding
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            # FFN output
            ffn_out_emb = model.transformer.h[0].mlp(token_emb)
            # 残差连接（这正是论文中的定义）
            combined_emb = token_emb + ffn_out_emb
            # 通过lm_head得到logits
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
        avg_edge_weight = np.nan
        
    if num_non_edges > 0:
        avg_non_edge_weight = np.mean(W_sub[non_edge_mask])
    else:
        avg_non_edge_weight = 0
    
    # 计算gap
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

def collect_evolution_data(configs, iterations=None):
    """收集所有实验的权重差距演化数据"""
    if iterations is None:
        iterations = list(range(1000, 51000, 1000))  # 1k到50k，每1k一个点
    
    evolution_data = {}
    
    for exp_name, config in configs.items():
        print(f"\n📊 Processing {exp_name}...")
        model_data = {
            'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
            'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
            'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
        }
        
        found_checkpoints = 0
        for iteration in tqdm(iterations, desc=f"Loading checkpoints for {exp_name}"):
            checkpoint_path = find_checkpoint(config.checkpoint_dir, config.experiment_type, iteration)
            
            if checkpoint_path is None:
                # 如果checkpoint不存在，用NaN填充
                for path_type in model_data.keys():
                    model_data[path_type]['edge'].append(np.nan)
                    model_data[path_type]['non_edge'].append(np.nan)
                    model_data[path_type]['gap'].append(np.nan)
                continue
            
            found_checkpoints += 1
            
            # 加载模型并提取W'_M
            try:
                W_M_prime = load_model_and_extract_W_M_prime(checkpoint_path, config)
                
                # 计算每个路径类型的统计量
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
        evolution_data[exp_name] = model_data
    
    return evolution_data, iterations

def plot_comparison(evolution_data, iterations, save_dir):
    """生成对比图（ALPINE vs Original）"""
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Gap Evolution: ALPINE vs Original Training', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3']
    colors = {'Original (0%)': 'blue', 'ALPINE': 'red'}
    
    for i, path_type in enumerate(path_types):
        # 左列：Edge和Non-edge权重
        ax = axes[i, 0]
        for exp_name, model_data in evolution_data.items():
            color = colors.get(exp_name, 'gray')
            edge_weights = model_data[path_type]['edge']
            non_edge_weights = model_data[path_type]['non_edge']
            
            # 找到有效数据点
            valid_iters_edge = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
            valid_weights_edge = [w for w in edge_weights if not np.isnan(w)]
            valid_iters_nonedge = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
            valid_weights_nonedge = [w for w in non_edge_weights if not np.isnan(w)]
            
            if valid_weights_edge:
                ax.plot(valid_iters_edge, valid_weights_edge, marker='o', 
                       label=f'{exp_name} - Edge', color=color, linewidth=2, markersize=4)
            if valid_weights_nonedge:
                ax.plot(valid_iters_nonedge, valid_weights_nonedge, marker='s', 
                       label=f'{exp_name} - Non-Edge', color=color, linewidth=2, 
                       markersize=4, linestyle='--', alpha=0.7)
        
        ax.set_ylabel(f'{path_type}\nAverage Weight', fontsize=11)
        ax.set_title('Edge vs Non-Edge Weights', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # 右列：Weight Gap
        ax = axes[i, 1]
        for exp_name, model_data in evolution_data.items():
            color = colors.get(exp_name, 'gray')
            gaps = model_data[path_type]['gap']
            
            # 找到有效数据点
            valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
            valid_gaps = [g for g in gaps if not np.isnan(g)]
            
            if valid_gaps:
                ax.plot(valid_iters, valid_gaps, marker='^', label=exp_name,
                       color=color, linewidth=2.5, markersize=7)
        
        ax.set_ylabel('Weight Gap (Edge - Non-edge)', fontsize=11)
        ax.set_title('Weight Gap Evolution', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加文字说明
        if path_type == 'S2->S3':
            ax.text(0.95, 0.05, 'Key: Positive gap = S2 preserved', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 设置x轴标签
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel('Training Steps', fontsize=11)
            # 只显示部分刻度
            step = max(1, len(iterations) // 10)
            axes[i, j].set_xticks(iterations[::step])
            axes[i, j].set_xticklabels([f'{k//1000}k' for k in iterations[::step]], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    save_path = os.path.join(save_dir, 'weight_gap_comparison_alpine.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {save_path}")
    plt.show()
    
    return save_path

def print_summary_table(evolution_data, iterations):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("📊 WEIGHT GAP COMPARISON TABLE:")
    print("="*80)
    
    # 选择关键迭代点
    key_iters = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    available_iters = [it for it in key_iters if it in iterations]
    
    print("\n┌─────────┬────────────────────────────┬────────────────────────────┐")
    print("│  Steps  │    Original (0%) Gap       │      ALPINE Gap            │")
    print("│         │   S1→S2    │    S2→S3     │   S1→S2    │    S2→S3     │")
    print("├─────────┼────────────┼──────────────┼────────────┼──────────────┤")
    
    for iter_val in available_iters:
        idx = iterations.index(iter_val) if iter_val in iterations else -1
        if idx >= 0:
            # Original gaps
            orig_s12 = evolution_data.get('Original (0%)', {}).get('S1->S2', {}).get('gap', [])[idx] if 'Original (0%)' in evolution_data else np.nan
            orig_s23 = evolution_data.get('Original (0%)', {}).get('S2->S3', {}).get('gap', [])[idx] if 'Original (0%)' in evolution_data else np.nan
            # ALPINE gaps
            alp_s12 = evolution_data.get('ALPINE', {}).get('S1->S2', {}).get('gap', [])[idx] if 'ALPINE' in evolution_data else np.nan
            alp_s23 = evolution_data.get('ALPINE', {}).get('S2->S3', {}).get('gap', [])[idx] if 'ALPINE' in evolution_data else np.nan
            
            # 格式化输出
            orig_s12_str = f"{orig_s12:>10.4f}" if not np.isnan(orig_s12) else "    N/A   "
            orig_s23_str = f"{orig_s23:>12.4f}" if not np.isnan(orig_s23) else "     N/A    "
            alp_s12_str = f"{alp_s12:>10.4f}" if not np.isnan(alp_s12) else "    N/A   "
            alp_s23_str = f"{alp_s23:>12.4f}" if not np.isnan(alp_s23) else "     N/A    "
            
            print(f"│ {iter_val:>7} │ {orig_s12_str} │ {orig_s23_str} │ {alp_s12_str} │ {alp_s23_str} │")
    
    print("└─────────┴────────────┴──────────────┴────────────┴──────────────┘")
    
    # S1->S3分析（无真实边）
    print("\n" + "="*80)
    print("📊 S1→S3 AVERAGE WEIGHTS (No true edges in graph):")
    print("="*80)
    
    print("\n┌─────────┬──────────────┬──────────────┐")
    print("│  Steps  │ Original (0%)│    ALPINE    │")
    print("├─────────┼──────────────┼──────────────┤")
    
    for iter_val in available_iters:
        idx = iterations.index(iter_val) if iter_val in iterations else -1
        if idx >= 0:
            orig_w = evolution_data.get('Original (0%)', {}).get('S1->S3', {}).get('non_edge', [])[idx] if 'Original (0%)' in evolution_data else np.nan
            alp_w = evolution_data.get('ALPINE', {}).get('S1->S3', {}).get('non_edge', [])[idx] if 'ALPINE' in evolution_data else np.nan
            
            orig_str = f"{orig_w:>12.4f}" if not np.isnan(orig_w) else "     N/A    "
            alp_str = f"{alp_w:>12.4f}" if not np.isnan(alp_w) else "     N/A    "
            
            print(f"│ {iter_val:>7} │ {orig_str} │ {alp_str} │")
    
    print("└─────────┴──────────────┴──────────────┘")

# ==================== 3. 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Generate weight gap evolution plots for ALPINE experiments.")
    parser.add_argument('--alpine_dir', type=str, default=None, 
                       help='Directory containing ALPINE checkpoints (e.g., out/composition_20241126_123456)')
    parser.add_argument('--original_dir', type=str, default='out_d92', 
                       help='Directory containing original 0% checkpoints')
    parser.add_argument('--save_dir', type=str, default='alpine_weight_analysis', 
                       help='Directory to save analysis results')
    parser.add_argument('--max_iter', type=int, default=50000, 
                       help='Maximum iteration to analyze')
    parser.add_argument('--step', type=int, default=1000, 
                       help='Step size between iterations')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 如果没有指定ALPINE目录，尝试自动查找
    if args.alpine_dir is None:
        possible_dirs = glob.glob("out/composition_*")
        if possible_dirs:
            args.alpine_dir = sorted(possible_dirs)[-1]
            print(f"📁 Auto-detected ALPINE directory: {args.alpine_dir}")
        else:
            args.alpine_dir = "out"
    
    print("\n" + "="*80)
    print("🔬 ALPINE WEIGHT GAP EVOLUTION ANALYSIS")
    print("="*80)
    print(f"ALPINE checkpoints: {args.alpine_dir}")
    print(f"Original checkpoints: {args.original_dir}")
    print(f"Output directory: {args.save_dir}")
    
    # 初始化配置
    configs = {}
    
    # 尝试加载原始实验（如果存在）
    if os.path.exists(args.original_dir):
        configs['Original (0%)'] = Config(experiment_type="0%", checkpoint_dir=args.original_dir)
        print("✓ Original (0%) experiment configuration loaded")
    
    # 加载ALPINE实验
    configs['ALPINE'] = Config(experiment_type="ALPINE", checkpoint_dir=args.alpine_dir)
    print("✓ ALPINE experiment configuration loaded")
    
    # 设置迭代范围
    iterations = list(range(args.step, args.max_iter + 1, args.step))
    
    # 收集演化数据
    print("\n" + "="*80)
    print("📈 COLLECTING WEIGHT GAP EVOLUTION DATA...")
    print("="*80)
    
    evolution_data, iterations = collect_evolution_data(configs, iterations)
    
    # 打印汇总表格
    print_summary_table(evolution_data, iterations)
    
    # 生成对比图
    print("\n" + "="*80)
    print("📊 GENERATING COMPARISON PLOTS...")
    print("="*80)
    
    plot_comparison(evolution_data, iterations, args.save_dir)
    
    # 计算并显示关键发现
    print("\n" + "="*80)
    print("🔍 KEY FINDINGS:")
    print("="*80)
    
    for exp_name in evolution_data.keys():
        print(f"\n{exp_name}:")
        
        # S2->S3 gap分析（最重要）
        s23_gaps = evolution_data[exp_name]['S2->S3']['gap']
        valid_gaps = [g for g in s23_gaps if not np.isnan(g)]
        
        if valid_gaps:
            initial_gap = valid_gaps[0] if len(valid_gaps) > 0 else np.nan
            final_gap = valid_gaps[-1] if len(valid_gaps) > 0 else np.nan
            min_gap = min(valid_gaps)
            max_gap = max(valid_gaps)
            
            print(f"  S2→S3 Gap Evolution:")
            print(f"    Initial: {initial_gap:.4f}")
            print(f"    Final:   {final_gap:.4f}")
            print(f"    Range:   [{min_gap:.4f}, {max_gap:.4f}]")
            
            if min_gap > 0:
                print(f"    ✅ Always positive! S2 information preserved.")
            elif final_gap > 0:
                print(f"    ⚠️ Eventually positive, but had negative period.")
            else:
                print(f"    ❌ Negative gap detected - S2 might be suppressed.")
    
    # 保存数据到JSON
    save_data = {}
    for exp_name, model_data in evolution_data.items():
        save_data[exp_name] = {}
        for path_type, path_data in model_data.items():
            save_data[exp_name][path_type] = {
                'iterations': iterations,
                'edge': [float(x) if not np.isnan(x) else None for x in path_data['edge']],
                'non_edge': [float(x) if not np.isnan(x) else None for x in path_data['non_edge']],
                'gap': [float(x) if not np.isnan(x) else None for x in path_data['gap']]
            }
    
    json_path = os.path.join(args.save_dir, 'alpine_evolution_data.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n💾 Data saved to: {json_path}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📌 Please check the following:")
    print("1. S2→S3 gap in ALPINE - should stay positive")
    print("2. S1→S2 gap in ALPINE - might be lower than original")
    print("3. Overall learning dynamics - ALPINE might be slower but more stable")

if __name__ == "__main__":
    main()