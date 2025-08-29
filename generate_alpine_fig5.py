# generate_alpine_fig5.py

"""
Focused Analysis Script to Generate ALPINE Figure 5-style Weight Gap Plots.
This script directly addresses the professor's feedback by providing clear,
unambiguous evidence of whether the model learns adjacency information.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
import argparse

# 确保模型定义文件存在
try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ 错误：无法导入'model.py'。请确保该文件与此脚本在同一目录下。")
    exit()

# ==================== 1. 配置与模型加载 (复用你的优秀代码) ====================

class Config:
    """配置类，负责加载所有资源。"""
    def __init__(self, seed=42, checkpoint_dir="out_d92"):
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92 # 假设是92维
        
        # 数据目录
        self.data_dir = "data/simple_graph/composition_90"
        
        # 加载元数据和图
        self.load_metadata_and_graph()

    def load_metadata_and_graph(self):
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f: meta = pickle.load(f)
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

def get_final_checkpoint_path(checkpoint_dir, ratio, seed):
    """查找最终的checkpoint文件路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: raise FileNotFoundError(f"Directory not found: {pattern}")
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f"ckpt_mix{ratio}_seed{seed}_iter50000.pt")
    if not os.path.exists(path):
        available = glob.glob(os.path.join(latest_dir, f"ckpt_mix{ratio}_seed{seed}_iter*.pt"))
        if not available: raise FileNotFoundError(f"No checkpoint found in {latest_dir}")
        path = sorted(available, key=os.path.getmtime)[-1]
    return path

def load_model_and_extract_W_M_prime(checkpoint_path, config):
    """加载模型并提取W'_M矩阵。"""
    print(f"Loading model from {os.path.basename(checkpoint_path)}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model_args = checkpoint.get('model_args', {})
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("Extracting W'_M matrix...")
    W_M_prime = []
    with torch.no_grad():
        wte = model.transformer.wte.weight
        ffn_in = model.transformer.h[0].mlp.c_fc.weight
        ffn_out = model.transformer.h[0].mlp.c_proj.weight
        
        # ALPINE论文中的精确计算公式: FFN(e_i) @ W_o + e_i @ W_o
        # 这是一个简化的、但功能等价的近似
        # W'_M ≈ W_in @ W_out
        # 我们这里使用你脚本中的方法，它更完整
        for i in range(config.vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            ffn_out_emb = model.transformer.h[0].mlp(token_emb)
            combined_emb = token_emb + ffn_out_emb # 残差连接
            logits = model.lm_head(combined_emb)
            W_M_prime.append(logits.squeeze().cpu().numpy())
            
    return np.array(W_M_prime)

# ==================== 2. 核心分析与绘图函数 (全新编写) ====================

def calculate_weight_gap(W_M_prime, A_true, path_type):
    """
    计算指定路径类型的平均权重差距。
    这是整个脚本的核心，直接回答教授的问题。
    """
    if path_type == 'S1->S2':
        rows, cols = np.s_[2:32], np.s_[32:62]
    elif path_type == 'S2->S3':
        rows, cols = np.s_[32:62], np.s_[62:90] # 修正S3的范围
    elif path_type == 'S1->S3':
        rows, cols = np.s_[2:32], np.s_[62:90] # 修正S3的范围
    else:
        raise ValueError("Invalid path_type")

    # 提取对应子矩阵
    W_sub = W_M_prime[rows, cols]
    A_sub = A_true[rows, cols]

    # 创建掩码
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)

    # 计算平均权重
    # 加上一个极小值防止除以零
    avg_edge_weight = np.mean(W_sub[edge_mask]) if np.sum(edge_mask) > 0 else 0
    avg_non_edge_weight = np.mean(W_sub[non_edge_mask]) if np.sum(non_edge_mask) > 0 else 0

    # 计算差距
    gap = avg_edge_weight - avg_non_edge_weight
    
    return {
        'avg_edge_weight': avg_edge_weight,
        'avg_non_edge_weight': avg_non_edge_weight,
        'gap': gap
    }

def plot_weight_gap_analysis(results, save_path):
    """
    生成清晰的条形图，可视化权重差距。
    这张图可以直接放入你的报告或邮件中。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('Analysis of Adjacency Information in W_M\' Matrix', fontsize=16, fontweight='bold')

    model_types = ['0% Mix Model', '20% Mix Model']
    path_types = list(results['0% Mix Model'].keys())
    
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        data = results[model_type]
        
        gaps = [data[pt]['gap'] for pt in path_types]
        avg_edges = [data[pt]['avg_edge_weight'] for pt in path_types]
        avg_non_edges = [data[pt]['avg_non_edge_weight'] for pt in path_types]
        
        x = np.arange(len(path_types))
        width = 0.35

        rects1 = ax.bar(x - width/2, avg_edges, width, label='Avg. Edge Weight', color='royalblue')
        rects2 = ax.bar(x + width/2, avg_non_edges, width, label='Avg. Non-Edge Weight', color='lightcoral')

        ax.set_ylabel('Average Weight in W_M\'')
        ax.set_title(model_type, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(path_types)
        ax.legend()
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 在条形图上标注差距值
        for j, pt in enumerate(path_types):
            gap_value = data[pt]['gap']
            ax.text(j, max(avg_edges[j], avg_non_edges[j]) + 0.5, f'Gap = {gap_value:.2f}', 
                    ha='center', va='bottom', fontweight='bold', color='green' if gap_value > 0 else 'red')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis plot saved to: {save_path}")
    plt.show()


# ==================== 3. 主执行函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Generate ALPINE Figure 5-style weight gap plots.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='out_d92', help='Directory containing model checkpoints')
    parser.add_argument('--save_dir', type=str, default='alpine_gap_analysis', help='Directory to save analysis results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 初始化配置
    config = Config(seed=args.seed, checkpoint_dir=args.checkpoint_dir)

    # 2. 加载模型并提取矩阵
    model_paths = {
        '0% Mix Model': get_final_checkpoint_path(args.checkpoint_dir, 0, args.seed),
        '20% Mix Model': get_final_checkpoint_path(args.checkpoint_dir, 20, args.seed)
    }
    
    W_M_primes = {}
    for name, path in model_paths.items():
        W_M_primes[name] = load_model_and_extract_W_M_prime(path, config)

    # 3. 对每个模型和路径类型进行分析
    all_results = {}
    path_types_to_analyze = ['S1->S2', 'S2->S3', 'S1->S3']

    print("\n" + "="*80)
    print("📊 CALCULATING AVERAGE WEIGHT GAPS...")
    print("="*80)

    for model_name, W_M_prime in W_M_primes.items():
        print(f"\n--- Analyzing: {model_name} ---")
        model_results = {}
        for path_type in path_types_to_analyze:
            gap_data = calculate_weight_gap(W_M_prime, config.A_true, path_type)
            model_results[path_type] = gap_data
            print(f"  {path_type}:")
            print(f"    Avg. Edge Weight:     {gap_data['avg_edge_weight']:.4f}")
            print(f"    Avg. Non-Edge Weight: {gap_data['avg_non_edge_weight']:.4f}")
            print(f"    >> Average Gap:        {gap_data['gap']:.4f}")
        all_results[model_name] = model_results
        
    # 4. 可视化结果
    save_path = os.path.join(args.save_dir, f'weight_gap_analysis_seed{args.seed}.png')
    plot_weight_gap_analysis(all_results, save_path)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE.")
    print("="*80)

if __name__ == "__main__":
    main()