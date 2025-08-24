"""
ALPINE Matrix Extraction and Comparison for Compositional Learning Analysis
最终稳定版 - 使用循环提取矩阵，并包含鲁棒的Ground Truth验证（权重差距法）
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import glob
import networkx as nx
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

# 导入你的模型定义
try:
    from model import GPTConfig, GPT
except ImportError:
    print("错误：无法导入'model.py'。请确保该文件与此脚本在同一目录下。")
    exit()

# ==================== 配置部分 (与之前相同) ====================
class Config:
    def __init__(self, seed=42, checkpoint_dir="out_d92"):
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        self.data_dir_0 = "data/simple_graph/composition_90"
        self.model_0_path = self.get_final_checkpoint_path(0, seed)
        self.model_20_path = self.get_final_checkpoint_path(20, seed)
        print(f"✓ 找到0%模型: {self.model_0_path}")
        print(f"✓ 找到20%模型: {self.model_20_path}")
        self.load_metadata_and_graph()
        self.token_S1 = list(range(2, 32))
        self.token_S2 = list(range(32, 62))
        self.token_S3 = list(range(62, 92))
        print(f"✓ Token映射: S1={self.token_S1[0]}-{self.token_S1[-1]}, " +
              f"S2={self.token_S2[0]}-{self.token_S2[-1]}, " +
              f"S3={self.token_S3[0]}-{self.token_S3[-1]}")

    def get_final_checkpoint_path(self, ratio, seed):
        pattern = f"{self.checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
        dirs = glob.glob(pattern)
        if not dirs:
            raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
        latest_dir = sorted(dirs)[-1]
        iteration = 50000
        expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
        path = os.path.join(latest_dir, expected_filename)
        if not os.path.exists(path):
            available_files = glob.glob(os.path.join(latest_dir, f"ckpt_mix{ratio}_seed{seed}_iter*.pt"))
            if available_files:
                path = sorted(available_files, key=os.path.getmtime)[-1]
                print(f"⚠️ 未找到iter50000，使用最新的checkpoint: {os.path.basename(path)}")
            else:
                raise FileNotFoundError(f"未找到checkpoint文件在: {latest_dir}")
        return path

    def load_metadata_and_graph(self):
        base_data_dir = self.data_dir_0
        meta_path = os.path.join(base_data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"未找到meta.pkl在: {base_data_dir}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        print(f"✓ 加载元数据: vocab_size={self.vocab_size}")
        graph_path = os.path.join(base_data_dir, 'composition_graph.graphml')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"未找到图文件: {graph_path}")
        self.G = nx.read_graphml(graph_path)
        self.G = nx.relabel_nodes(self.G, {node: int(node) for node in self.G.nodes()})
        print(f"✓ 加载图结构: {len(self.G.nodes)}个节点, {len(self.G.edges)}条边")
        print("\n   计算Ground Truth矩阵...")
        self.compute_ground_truth_matrices()

    def compute_ground_truth_matrices(self):
        num_nodes = 90
        nodelist = sorted(self.G.nodes())
        a_true_90 = nx.to_numpy_array(self.G, nodelist=nodelist)
        r_true_graph = nx.transitive_closure(self.G, reflexive=True)
        r_true_90 = nx.to_numpy_array(r_true_graph, nodelist=nodelist)
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        self.R_true = np.zeros((self.vocab_size, self.vocab_size))
        self.A_true[2:num_nodes+2, 2:num_nodes+2] = a_true_90
        self.R_true[2:num_nodes+2, 2:num_nodes+2] = r_true_90
        print(f"   ✓ A_true (真实邻接矩阵) 计算完成，形状: {self.A_true.shape}")
        print(f"   ✓ R_true (真实可达性矩阵) 计算完成，形状: {self.R_true.shape}")

# ==================== ALPINE矩阵提取器 (恢复为您的原始稳定版本) ====================
class ALPINEMatrixExtractor:
    def __init__(self, checkpoint_path: str, config: Config, model_type: str = "unknown"):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = config.device
        self.model_type = model_type
        self.model = self.load_model()
        self.model.eval()
        print(f"✓ 成功加载{model_type}模型: {os.path.basename(checkpoint_path)}")

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_args = checkpoint.get('model_args', {
            'n_layer': self.config.n_layer, 'n_head': self.config.n_head, 'n_embd': self.config.n_embd,
            'block_size': 32, 'bias': False, 'vocab_size': self.config.vocab_size, 'dropout': 0.0
        })
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(self.device)
        model.load_state_dict(checkpoint['model'])
        return model

    def extract_adjacency_matrix(self) -> np.ndarray:
        vocab_size = self.config.vocab_size
        W_M_prime = []
        print(f"  提取邻接矩阵 ({self.model_type})...")
        with torch.no_grad():
            for node_i in range(vocab_size):
                if node_i % 10 == 0:
                    print(f"    处理节点 {node_i}/{vocab_size}...", end='\r')
                token_emb = self.model.transformer.wte(torch.tensor([node_i], device=self.device)).squeeze(0)
                token_emb_expanded = token_emb.unsqueeze(0).unsqueeze(0)
                ffn_out = self.model.transformer.h[0].mlp(token_emb_expanded).squeeze()
                combined = ffn_out + token_emb
                combined = self.model.transformer.ln_f(combined.unsqueeze(0)).squeeze()
                output_weight = self.model.lm_head.weight
                logits = combined @ output_weight.T
                W_M_prime.append(logits.cpu().numpy())
        print(f"\n    完成! Shape: {len(W_M_prime)}x{len(W_M_prime[0])}")
        return np.array(W_M_prime)

    def extract_reachability_matrix(self) -> np.ndarray:
        vocab_size = self.config.vocab_size
        n_embd = self.config.n_embd
        W_V_prime = []
        print(f"  提取可达矩阵 ({self.model_type})...")
        with torch.no_grad():
            c_attn_weight = self.model.transformer.h[0].attn.c_attn.weight
            W_V = c_attn_weight[:, 2*n_embd:3*n_embd]
            for target_node in range(vocab_size):
                if target_node % 10 == 0:
                    print(f"    处理目标节点 {target_node}/{vocab_size}...", end='\r')
                target_emb = self.model.transformer.wte(torch.tensor([target_node], device=self.device)).squeeze(0)
                value_features = target_emb @ W_V
                value_features_expanded = value_features.unsqueeze(0).unsqueeze(0)
                ffn_out = self.model.transformer.h[0].mlp(value_features_expanded).squeeze()
                combined = value_features + ffn_out
                combined = self.model.transformer.ln_f(combined.unsqueeze(0)).squeeze()
                output_weight = self.model.lm_head.weight
                reachability_scores = combined @ output_weight.T
                W_V_prime.append(reachability_scores.cpu().numpy())
        print(f"\n    完成! Shape: {len(W_V_prime)}x{len(W_V_prime[0])}")
        return np.array(W_V_prime)

# ==================== 矩阵比较和分析器 ====================
class MatrixComparator:
    def __init__(self, config: Config):
        self.config = config
        print("\n" + "="*60)
        print("ALPINE矩阵提取和比较分析")
        print("="*60)
        self.extractor_0 = ALPINEMatrixExtractor(config.model_0_path, config, "0%")
        self.extractor_20 = ALPINEMatrixExtractor(config.model_20_path, config, "20%")
        print("\n2. 提取矩阵表示...")
        self.W_M_0 = self.extractor_0.extract_adjacency_matrix()
        self.W_V_0 = self.extractor_0.extract_reachability_matrix()
        self.W_M_20 = self.extractor_20.extract_adjacency_matrix()
        self.W_V_20 = self.extractor_20.extract_reachability_matrix()
        print("\n✓ 矩阵提取完成！")

    def compute_similarities_between_models(self) -> Dict:
        results = {}
        for name, m1, m2 in [("Adjacency", self.W_M_0, self.W_M_20), ("Reachability", self.W_V_0, self.W_V_20)]:
            flat1, flat2 = m1.flatten(), m2.flatten()
            results[name] = {
                'pearson': float(np.corrcoef(flat1, flat2)[0, 1]),
                'spearman': float(spearmanr(flat1, flat2)[0]),
                'cosine': float(cosine_similarity(flat1.reshape(1, -1), flat2.reshape(1, -1))[0, 0]),
                'rmse': float(np.sqrt(np.mean((m1 - m2)**2))),
            }
        return results

    def analyze_weight_gap_vs_ground_truth(self) -> Dict:
        results = {'adjacency': {}, 'reachability': {}}
        adj_edge_indices = np.where(self.config.A_true == 1)
        adj_non_edge_indices = np.where(self.config.A_true == 0)
        for model_name, W_M in [("0%", self.W_M_0), ("20%", self.W_M_20)]:
            edge_weights = W_M[adj_edge_indices]
            non_edge_weights = W_M[adj_non_edge_indices]
            gap = np.mean(edge_weights) - np.mean(non_edge_weights)
            results['adjacency'][model_name] = {
                'mean_edge_weight': float(np.mean(edge_weights)),
                'mean_non_edge_weight': float(np.mean(non_edge_weights)),
                'weight_gap': float(gap)
            }
        gap_0_adj = results['adjacency']['0%']['weight_gap']
        gap_20_adj = results['adjacency']['20%']['weight_gap']
        results['adjacency']['improvement'] = {'gap_increase': float(gap_20_adj - gap_0_adj)}
        
        reach_indices = np.where(self.config.R_true == 1)
        non_reach_indices = np.where(self.config.R_true == 0)
        for model_name, W_V in [("0%", self.W_V_0), ("20%", self.W_V_20)]:
            reach_weights = W_V[reach_indices]
            non_reach_weights = W_V[non_reach_indices]
            gap = np.mean(reach_weights) - np.mean(non_reach_weights)
            results['reachability'][model_name] = {
                'mean_reach_weight': float(np.mean(reach_weights)),
                'mean_non_reach_weight': float(np.mean(non_reach_weights)),
                'weight_gap': float(gap)
            }
        gap_0_reach = results['reachability']['0%']['weight_gap']
        gap_20_reach = results['reachability']['20%']['weight_gap']
        results['reachability']['improvement'] = {'gap_increase': float(gap_20_reach - gap_0_reach)}
        return results

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="ALPINE Matrix Analysis - Final Stable Version")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='out_d92', help='Checkpoints directory')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 ALPINE Matrix Analysis - Final Stable Version")
    print("="*80)
    
    try:
        config = Config(seed=args.seed, checkpoint_dir=args.checkpoint_dir)
        comparator = MatrixComparator(config)
        
        print("\n" + "="*60)
        print("3. Ground Truth Validation (Robust Weight Gap Method)")
        print("="*60)
        
        gap_results = comparator.analyze_weight_gap_vs_ground_truth()
        
        print("\n📊 Adjacency Matrix - Weight Gap Analysis:")
        adj_0 = gap_results['adjacency']['0%']
        adj_20 = gap_results['adjacency']['20%']
        adj_imp = gap_results['adjacency']['improvement']
        
        print(f"  0% Model:")
        print(f"    - Mean Edge Weight:     {adj_0['mean_edge_weight']:.4f}")
        print(f"    - Mean Non-Edge Weight: {adj_0['mean_non_edge_weight']:.4f}")
        print(f"    - WEIGHT GAP:           {adj_0['weight_gap']:.4f}")
        
        print(f"  20% Model:")
        print(f"    - Mean Edge Weight:     {adj_20['mean_edge_weight']:.4f}")
        print(f"    - Mean Non-Edge Weight: {adj_20['mean_non_edge_weight']:.4f}")
        print(f"    - WEIGHT GAP:           {adj_20['weight_gap']:.4f}")
        
        print(f"  ✨ Improvement in Gap: {adj_imp['gap_increase']:+.4f}")
        
        print("\n📊 Reachability Matrix - Weight Gap Analysis:")
        reach_0 = gap_results['reachability']['0%']
        reach_20 = gap_results['reachability']['20%']
        reach_imp = gap_results['reachability']['improvement']
        
        print(f"  0% Model:")
        print(f"    - Mean Reachable Weight:     {reach_0['mean_reach_weight']:.4f}")
        print(f"    - Mean Non-Reachable Weight: {reach_0['mean_non_reach_weight']:.4f}")
        print(f"    - WEIGHT GAP:                {reach_0['weight_gap']:.4f}")
        
        print(f"  20% Model:")
        print(f"    - Mean Reachable Weight:     {reach_20['mean_reach_weight']:.4f}")
        print(f"    - Mean Non-Reachable Weight: {reach_20['mean_non_reach_weight']:.4f}")
        print(f"    - WEIGHT GAP:                {reach_20['weight_gap']:.4f}")
        
        print(f"  ✨ Improvement in Gap: {reach_imp['gap_increase']:+.4f}")

        print("\n" + "="*60)
        print("4. Model Comparison (0% vs 20%) - For Context")
        print("="*60)
        similarities = comparator.compute_similarities_between_models()
        for name, metrics in similarities.items():
            print(f"\n{name} Matrix Similarity:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()