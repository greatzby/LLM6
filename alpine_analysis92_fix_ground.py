"""
ALPINE Matrix Extraction and Comparison for Compositional Learning Analysis
完整版 - 包含Ground Truth验证、自循环分析和全面的连接分析
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
from model import GPTConfig, GPT

# ==================== 配置部分 ====================
class Config:
    """配置类 - 增强版包含Ground Truth"""
    def __init__(self, seed=42, checkpoint_dir="out_d92"):
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 92
        
        # 数据目录
        self.data_dir_0 = "data/simple_graph/composition_90"
        self.data_dir_20 = "data/simple_graph/composition_90_mixed_20"
        
        # 验证数据目录
        for data_dir in [self.data_dir_0, self.data_dir_20]:
            if not os.path.exists(data_dir):
                print(f"⚠️ 警告: 数据目录不存在: {data_dir}")
        
        # 查找checkpoint
        self.model_0_path = self.get_final_checkpoint_path(0, seed)
        self.model_20_path = self.get_final_checkpoint_path(20, seed)
        
        print(f"✓ 找到0%模型: {self.model_0_path}")
        print(f"✓ 找到20%模型: {self.model_20_path}")
        
        # 加载元数据和图结构（包含Ground Truth计算）
        self.load_metadata_and_graph()
        
        # Token映射关系
        self.token_S1 = list(range(2, 32))
        self.token_S2 = list(range(32, 62))
        self.token_S3 = list(range(62, 92))
        
        print(f"✓ Token映射: S1={self.token_S1[0]}-{self.token_S1[-1]}, " +
              f"S2={self.token_S2[0]}-{self.token_S2[-1]}, " +
              f"S3={self.token_S3[0]}-{self.token_S3[-1]}")
        
    def get_final_checkpoint_path(self, ratio, seed):
        """查找模型路径"""
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
                path = sorted(available_files)[-1]
                print(f"使用checkpoint: {path}")
            else:
                raise FileNotFoundError(f"未找到checkpoint文件在: {latest_dir}")
        return path
    
    def load_metadata_and_graph(self):
        """加载元数据和图结构，并计算Ground Truth矩阵"""
        base_data_dir = self.data_dir_0
        
        # 加载meta.pkl
        meta_path = os.path.join(base_data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"未找到meta.pkl在: {base_data_dir}")
            
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.block_size = meta['block_size']
        self.vocab_size = meta['vocab_size']
        
        print(f"✓ 加载元数据: vocab_size={self.vocab_size}, block_size={self.block_size}")
        
        # 加载stage信息
        stage_path = os.path.join(base_data_dir, 'stage_info.pkl')
        if os.path.exists(stage_path):
            with open(stage_path, 'rb') as f:
                stage_info = pickle.load(f)
            self.stages = stage_info['stages']
            self.S1, self.S2, self.S3 = self.stages
            print(f"✓ 加载stage信息: S1={len(self.S1)}个节点, S2={len(self.S2)}个节点, S3={len(self.S3)}个节点")
        else:
            # 如果没有stage_info.pkl，使用默认值
            self.S1 = list(range(30))
            self.S2 = list(range(30, 60))
            self.S3 = list(range(60, 90))
            self.stages = [self.S1, self.S2, self.S3]
            print(f"⚠️ 使用默认stage划分")
        
        # 加载图结构
        graph_path = os.path.join(base_data_dir, 'composition_graph.graphml')
        if not os.path.exists(graph_path):
            print(f"⚠️ 图文件不存在: {graph_path}")
            print("创建默认的三阶段图结构...")
            self.G = self.create_default_graph()
        else:
            self.G = nx.read_graphml(graph_path)
            # 将节点标签从字符串转回整数
            self.G = nx.relabel_nodes(self.G, {node: int(node) for node in self.G.nodes()})
        
        print(f"✓ 加载图结构: {len(self.G.nodes)}个节点, {len(self.G.edges)}条边")
        
        # --- 计算Ground Truth矩阵 ---
        print("\n   计算Ground Truth矩阵...")
        self.compute_ground_truth_matrices()
        
    def create_default_graph(self):
        """创建默认的三阶段图结构"""
        G = nx.DiGraph()
        # 添加所有节点
        G.add_nodes_from(range(90))
        
        # 添加S1→S2边
        for s1 in self.S1:
            for s2 in self.S2:
                if np.random.rand() < 0.1:  # 10%概率
                    G.add_edge(s1, s2)
        
        # 添加S2→S3边
        for s2 in self.S2:
            for s3 in self.S3:
                if np.random.rand() < 0.1:  # 10%概率
                    G.add_edge(s2, s3)
        
        return G
    
    def compute_ground_truth_matrices(self):
        """计算Ground Truth矩阵"""
        num_nodes = len(self.G.nodes())
        nodelist = sorted(self.G.nodes())
        
        # 1. A_true (真实邻接矩阵) - 90x90
        a_true_90 = nx.to_numpy_array(self.G, nodelist=nodelist)
        
        # 2. R_true (真实可达性矩阵) - 90x90
        r_true_graph = nx.transitive_closure(self.G, reflexive=True)
        r_true_90 = nx.to_numpy_array(r_true_graph, nodelist=nodelist)
        
        # 填充到92x92以匹配模型输出
        # Token 0: [PAD], Token 1: \n, Token 2-91: 图节点0-89
        self.A_true = np.zeros((92, 92))
        self.R_true = np.zeros((92, 92))
        
        # 将图矩阵(90x90)映射到token空间(92x92)
        # 图节点i → Token i+2
        self.A_true[2:92, 2:92] = a_true_90
        self.R_true[2:92, 2:92] = r_true_90
        
        print(f"   ✓ A_true (真实邻接矩阵) 计算完成，形状: {self.A_true.shape}")
        print(f"   ✓ R_true (真实可达性矩阵) 计算完成，形状: {self.R_true.shape}")
        
        # 统计信息
        print(f"   ✓ 真实图统计: S1→S2边={np.sum(a_true_90[:30, 30:60]):.0f}, "
              f"S2→S3边={np.sum(a_true_90[30:60, 60:90]):.0f}, "
              f"总边数={np.sum(a_true_90):.0f}")
        
    def verify_data_consistency(self):
        """验证数据一致性"""
        try:
            with open(os.path.join(self.data_dir_20, 'meta.pkl'), 'rb') as f:
                meta_20 = pickle.load(f)
            
            if meta_20['vocab_size'] != self.vocab_size:
                print(f"⚠️ 警告: 20%数据的vocab_size不一致!")
            if meta_20['block_size'] != self.block_size:
                print(f"⚠️ 警告: 20%数据的block_size不一致!")
                
            print(f"✓ 数据一致性检查通过")
        except Exception as e:
            print(f"⚠️ 无法验证20%数据目录: {e}")

# ==================== ALPINE矩阵提取器 ====================
class ALPINEMatrixExtractor:
    """基于ALPINE论文的矩阵提取器"""
    
    def __init__(self, checkpoint_path: str, config: Config, model_type: str = "unknown"):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = config.device
        self.model_type = model_type
        
        # 加载模型
        self.model = self.load_model()
        self.model.eval()
        
        print(f"✓ 成功加载{model_type}模型: {os.path.basename(checkpoint_path)}")
        
    def load_model(self):
        """加载GPT模型"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'model_args' in checkpoint:
            model_args = checkpoint['model_args']
        else:
            model_args = dict(
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_embd=self.config.n_embd,
                block_size=self.config.block_size,
                bias=False,
                vocab_size=self.config.vocab_size,
                dropout=0.0
            )
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(self.device)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params/1e6:.2f}M")
        
        return model
    
    def extract_adjacency_matrix(self) -> np.ndarray:
        """提取邻接矩阵表示 W'_M (修复版本处理bias=None)"""
        vocab_size = self.config.vocab_size
        
        print(f"  提取邻接矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            # 提取所需权重
            wte = self.model.transformer.wte.weight  # (vocab_size, n_embd)
            ffn_w1 = self.model.transformer.h[0].mlp.c_fc.weight
            ffn_b1 = self.model.transformer.h[0].mlp.c_fc.bias
            ffn_w2 = self.model.transformer.h[0].mlp.c_proj.weight
            ffn_b2 = self.model.transformer.h[0].mlp.c_proj.bias
            ln_f_w = self.model.transformer.ln_f.weight
            ln_f_b = self.model.transformer.ln_f.bias
            lm_head_w = self.model.lm_head.weight if hasattr(self.model, 'lm_head') else wte

            # 批量计算FFN(e_i) - 处理bias可能为None的情况
            if ffn_b1 is not None:
                ffn_hidden = nn.functional.gelu(wte @ ffn_w1.T + ffn_b1)
            else:
                ffn_hidden = nn.functional.gelu(wte @ ffn_w1.T)
            
            if ffn_b2 is not None:
                ffn_out = ffn_hidden @ ffn_w2.T + ffn_b2
            else:
                ffn_out = ffn_hidden @ ffn_w2.T
            
            # 组合：FFN(e_i) + e_i
            combined = ffn_out + wte

            # LayerNorm
            mean = combined.mean(dim=1, keepdim=True)
            std = combined.std(dim=1, keepdim=True, unbiased=False)
            ln_out = ln_f_w * (combined - mean) / (std + 1e-5) + ln_f_b
            
            # 投影到词汇表
            W_M_prime = (ln_out @ lm_head_w.T).cpu().numpy()

        print(f"    完成! Shape: {W_M_prime.shape}")
        return W_M_prime
    
    def extract_reachability_matrix(self) -> np.ndarray:
        """提取可达矩阵表示 W'_V (修复版本处理bias=None)"""
        vocab_size = self.config.vocab_size
        n_embd = self.config.n_embd
        
        print(f"  提取可达矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            wte = self.model.transformer.wte.weight
            c_attn_weight = self.model.transformer.h[0].attn.c_attn.weight
            W_V = c_attn_weight[:, 2*n_embd:3*n_embd]  # Value矩阵
            
            ffn_w1 = self.model.transformer.h[0].mlp.c_fc.weight
            ffn_b1 = self.model.transformer.h[0].mlp.c_fc.bias
            ffn_w2 = self.model.transformer.h[0].mlp.c_proj.weight
            ffn_b2 = self.model.transformer.h[0].mlp.c_proj.bias
            
            ln_f_w = self.model.transformer.ln_f.weight
            ln_f_b = self.model.transformer.ln_f.bias
            lm_head_w = self.model.lm_head.weight if hasattr(self.model, 'lm_head') else wte

            # e_j @ W_V
            value_features = wte @ W_V

            # FFN(e_j @ W_V) - 处理bias可能为None的情况
            if ffn_b1 is not None:
                ffn_hidden = nn.functional.gelu(value_features @ ffn_w1.T + ffn_b1)
            else:
                ffn_hidden = nn.functional.gelu(value_features @ ffn_w1.T)
            
            if ffn_b2 is not None:
                ffn_out = ffn_hidden @ ffn_w2.T + ffn_b2
            else:
                ffn_out = ffn_hidden @ ffn_w2.T

            # 组合
            combined = value_features + ffn_out
            
            # LayerNorm
            mean = combined.mean(dim=1, keepdim=True)
            std = combined.std(dim=1, keepdim=True, unbiased=False)
            ln_out = ln_f_w * (combined - mean) / (std + 1e-5) + ln_f_b

            # 投影
            W_V_prime = (ln_out @ lm_head_w.T).cpu().numpy()
        
        print(f"    完成! Shape: {W_V_prime.shape}")
        return W_V_prime
    
    def extract_attention_pattern(self, num_samples: int = 50) -> np.ndarray:
        """提取注意力模式（保持原有实现）"""
        print(f"  提取注意力模式 ({self.model_type})...")
        
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        
        attention_maps = []
        
        try:
            with torch.no_grad():
                c_attn = self.model.transformer.h[0].attn.c_attn
                c_attn_weight = c_attn.weight
                
                W_Q = c_attn_weight[:n_embd, :]
                W_K = c_attn_weight[n_embd:2*n_embd, :]
                
                for i in range(min(num_samples, 50)):
                    source = np.random.choice(self.config.token_S1)
                    target = np.random.choice(self.config.token_S3)
                    middle = np.random.choice(self.config.token_S2)
                    
                    input_ids = torch.tensor([source, middle, target], dtype=torch.long, device=self.device)
                    
                    token_emb = self.model.transformer.wte(input_ids)
                    positions = torch.arange(0, 3, dtype=torch.long, device=self.device)
                    pos_emb = self.model.transformer.wpe(positions)
                    
                    hidden = token_emb + pos_emb
                    
                    Q = hidden @ W_Q.T
                    K = hidden @ W_K.T
                    
                    Q = Q.view(3, n_head, head_dim).squeeze(1)
                    K = K.view(3, n_head, head_dim).squeeze(1)
                    
                    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
                    attn_scores = attn_scores / math.sqrt(head_dim)
                    
                    seq_len = 3
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                    
                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    attention_maps.append(attn_weights.cpu().numpy())
            
            if attention_maps:
                avg_attention = np.mean(attention_maps, axis=0)
                print(f"    完成! Shape: {avg_attention.shape}")
                return avg_attention
            else:
                print(f"    ⚠️ 未能提取注意力模式，返回零矩阵")
                return np.zeros((3, 3))
                
        except Exception as e:
            print(f"    注意力提取出错: {e}")
            return np.zeros((3, 3))

# ==================== 矩阵比较和分析器（完整版） ====================
class MatrixComparator:
    """矩阵比较和可视化工具 - 完整版"""
    
    def __init__(self, config: Config):
        self.config = config
        
        print("\n" + "="*60)
        print("ALPINE矩阵提取和比较分析 - 完整版")
        print("="*60)
        print(f"Seed: {config.seed}")
        print(f"Model dimension: {config.n_embd}")
        print(f"Vocabulary size: {config.vocab_size}")
        
        # 创建提取器
        print("\n1. 加载模型...")
        self.extractor_0 = ALPINEMatrixExtractor(config.model_0_path, config, "0%")
        self.extractor_20 = ALPINEMatrixExtractor(config.model_20_path, config, "20%")
        
        # 提取矩阵
        print("\n2. 提取矩阵表示...")
        self.W_M_0 = self.extractor_0.extract_adjacency_matrix()
        self.W_V_0 = self.extractor_0.extract_reachability_matrix()
        self.attention_0 = self.extractor_0.extract_attention_pattern()
        
        self.W_M_20 = self.extractor_20.extract_adjacency_matrix()
        self.W_V_20 = self.extractor_20.extract_reachability_matrix()
        self.attention_20 = self.extractor_20.extract_attention_pattern()
        
        print("\n✓ 矩阵提取完成！")
        
    def compute_similarities(self) -> Dict:
        """计算0% vs 20%模型的矩阵相似度"""
        results = {}
        
        # 邻接矩阵相似度
        flat_M_0 = self.W_M_0.flatten()
        flat_M_20 = self.W_M_20.flatten()
        
        mask = ~(np.isnan(flat_M_0) | np.isnan(flat_M_20))
        if mask.sum() > 0:
            flat_M_0_clean = flat_M_0[mask]
            flat_M_20_clean = flat_M_20[mask]
            
            results['adjacency'] = {
                'pearson_correlation': float(np.corrcoef(flat_M_0_clean, flat_M_20_clean)[0, 1]),
                'spearman_correlation': float(spearmanr(flat_M_0_clean, flat_M_20_clean)[0]),
                'cosine_similarity': float(cosine_similarity(flat_M_0_clean.reshape(1, -1), 
                                                            flat_M_20_clean.reshape(1, -1))[0, 0]),
                'rmse': float(np.sqrt(np.mean((self.W_M_0 - self.W_M_20)**2))),
                'mean_abs_diff': float(np.mean(np.abs(self.W_M_0 - self.W_M_20)))
            }
        
        # 可达矩阵相似度
        flat_V_0 = self.W_V_0.flatten()
        flat_V_20 = self.W_V_20.flatten()
        
        mask = ~(np.isnan(flat_V_0) | np.isnan(flat_V_20))
        if mask.sum() > 0:
            flat_V_0_clean = flat_V_0[mask]
            flat_V_20_clean = flat_V_20[mask]
            
            results['reachability'] = {
                'pearson_correlation': float(np.corrcoef(flat_V_0_clean, flat_V_20_clean)[0, 1]),
                'spearman_correlation': float(spearmanr(flat_V_0_clean, flat_V_20_clean)[0]),
                'cosine_similarity': float(cosine_similarity(flat_V_0_clean.reshape(1, -1), 
                                                            flat_V_20_clean.reshape(1, -1))[0, 0]),
                'rmse': float(np.sqrt(np.mean((self.W_V_0 - self.W_V_20)**2))),
                'mean_abs_diff': float(np.mean(np.abs(self.W_V_0 - self.W_V_20)))
            }
        
        return results
    
    def analyze_vs_ground_truth(self) -> Dict:
        """将模型提取的矩阵与真实的图矩阵进行比较（教授要求的核心分析）"""
        results = {'adjacency': {}, 'reachability': {}}
        
        # 准备Ground Truth矩阵
        A_true = self.config.A_true.flatten()
        R_true = self.config.R_true.flatten()
        
        # 1. 邻接矩阵验证
        W_M_0_flat = self.W_M_0.flatten()
        W_M_20_flat = self.W_M_20.flatten()
        
        # Cosine相似度
        results['adjacency']['0%_vs_True'] = {
            'cosine': float(cosine_similarity(W_M_0_flat.reshape(1, -1), A_true.reshape(1, -1))[0, 0]),
            'pearson': float(np.corrcoef(W_M_0_flat, A_true)[0, 1]) if not np.isnan(np.corrcoef(W_M_0_flat, A_true)[0, 1]) else 0.0,
            'rmse': float(np.sqrt(np.mean((self.W_M_0 - self.config.A_true)**2)))
        }
        
        results['adjacency']['20%_vs_True'] = {
            'cosine': float(cosine_similarity(W_M_20_flat.reshape(1, -1), A_true.reshape(1, -1))[0, 0]),
            'pearson': float(np.corrcoef(W_M_20_flat, A_true)[0, 1]) if not np.isnan(np.corrcoef(W_M_20_flat, A_true)[0, 1]) else 0.0,
            'rmse': float(np.sqrt(np.mean((self.W_M_20 - self.config.A_true)**2)))
        }
        
        # 计算提升
        results['adjacency']['improvement'] = {
            'cosine': results['adjacency']['20%_vs_True']['cosine'] - results['adjacency']['0%_vs_True']['cosine'],
            'pearson': results['adjacency']['20%_vs_True']['pearson'] - results['adjacency']['0%_vs_True']['pearson'],
            'rmse': results['adjacency']['0%_vs_True']['rmse'] - results['adjacency']['20%_vs_True']['rmse']  # RMSE降低是好的
        }
        
        # 2. 可达性矩阵验证
        W_V_0_flat = self.W_V_0.flatten()
        W_V_20_flat = self.W_V_20.flatten()
        
        results['reachability']['0%_vs_True'] = {
            'cosine': float(cosine_similarity(W_V_0_flat.reshape(1, -1), R_true.reshape(1, -1))[0, 0]),
            'pearson': float(np.corrcoef(W_V_0_flat, R_true)[0, 1]) if not np.isnan(np.corrcoef(W_V_0_flat, R_true)[0, 1]) else 0.0,
            'rmse': float(np.sqrt(np.mean((self.W_V_0 - self.config.R_true)**2)))
        }
        
        results['reachability']['20%_vs_True'] = {
            'cosine': float(cosine_similarity(W_V_20_flat.reshape(1, -1), R_true.reshape(1, -1))[0, 0]),
            'pearson': float(np.corrcoef(W_V_20_flat, R_true)[0, 1]) if not np.isnan(np.corrcoef(W_V_20_flat, R_true)[0, 1]) else 0.0,
            'rmse': float(np.sqrt(np.mean((self.W_V_20 - self.config.R_true)**2)))
        }
        
        results['reachability']['improvement'] = {
            'cosine': results['reachability']['20%_vs_True']['cosine'] - results['reachability']['0%_vs_True']['cosine'],
            'pearson': results['reachability']['20%_vs_True']['pearson'] - results['reachability']['0%_vs_True']['pearson'],
            'rmse': results['reachability']['0%_vs_True']['rmse'] - results['reachability']['20%_vs_True']['rmse']
        }
        
        return results
    
    def analyze_self_loops(self) -> Dict:
        """专门分析自循环（对角线元素）的变化"""
        results = {
            'all_self_loops': [],
            'by_stage': {
                'Special': [],
                'S1': [],
                'S2': [],
                'S3': []
            },
            'summary': {}
        }
        
        # 分析所有对角线元素
        for i in range(self.config.vocab_size):
            weight_0 = self.W_M_0[i, i]
            weight_20 = self.W_M_20[i, i]
            change = weight_20 - weight_0
            
            # 确定阶段
            if i < 2:
                stage = 'Special'
            elif i < 32:
                stage = 'S1'
            elif i < 62:
                stage = 'S2'
            else:
                stage = 'S3'
            
            loop_info = {
                'token': i,
                'stage': stage,
                'node': i - 2 if i >= 2 else i,
                'weight_0%': float(weight_0),
                'weight_20%': float(weight_20),
                'change': float(change)
            }
            
            results['all_self_loops'].append(loop_info)
            results['by_stage'][stage].append(loop_info)
        
        # 计算每个阶段的统计
        for stage in ['Special', 'S1', 'S2', 'S3']:
            if results['by_stage'][stage]:
                changes = [item['change'] for item in results['by_stage'][stage]]
                results['summary'][stage] = {
                    'count': len(changes),
                    'mean_change': float(np.mean(changes)),
                    'std_change': float(np.std(changes)),
                    'max_change': float(np.max(changes)),
                    'min_change': float(np.min(changes)),
                    'significant_increases': sum(1 for c in changes if c > 1.0),
                    'significant_decreases': sum(1 for c in changes if c < -1.0)
                }
        
        results['top_increases'] = sorted(results['all_self_loops'], 
                                         key=lambda x: x['change'], 
                                         reverse=True)[:10]
        
        return results
    
    def analyze_all_s1_s3_connections(self) -> Dict:
        """分析所有S1→S3连接的变化"""
        results = {
            'all_connections': [],
            'summary': {},
            'distribution': {}
        }
        
        # 分析所有S1→S3连接
        for s1_token in self.config.token_S1:
            for s3_token in self.config.token_S3:
                s1_node = s1_token - 2
                s3_node = s3_token - 2
                
                adj_0 = self.W_M_0[s1_token, s3_token]
                adj_20 = self.W_M_20[s1_token, s3_token]
                adj_diff = adj_20 - adj_0
                
                reach_0 = self.W_V_0[s3_token, s1_token]
                reach_20 = self.W_V_20[s3_token, s1_token]
                reach_diff = reach_20 - reach_0
                
                connection_info = {
                    'path': f"Node{s1_node}→Node{s3_node}",
                    's1_token': s1_token,
                    's3_token': s3_token,
                    's1_node': s1_node,
                    's3_node': s3_node,
                    'adjacency': {
                        '0%': float(adj_0),
                        '20%': float(adj_20),
                        'diff': float(adj_diff)
                    },
                    'reachability': {
                        '0%': float(reach_0),
                        '20%': float(reach_20),
                        'diff': float(reach_diff)
                    }
                }
                
                results['all_connections'].append(connection_info)
        
        # 统计分析
        adj_diffs = [conn['adjacency']['diff'] for conn in results['all_connections']]
        reach_diffs = [conn['reachability']['diff'] for conn in results['all_connections']]
        
        results['summary'] = {
            'total_connections': len(results['all_connections']),
            'adjacency': {
                'mean_change': float(np.mean(adj_diffs)),
                'std_change': float(np.std(adj_diffs)),
                'median_change': float(np.median(adj_diffs)),
                'max_increase': float(np.max(adj_diffs)),
                'min_increase': float(np.min(adj_diffs)),
                'positive_changes': sum(1 for d in adj_diffs if d > 0),
                'negative_changes': sum(1 for d in adj_diffs if d < 0),
                'significant_increases': sum(1 for d in adj_diffs if d > 1.0)
            },
            'reachability': {
                'mean_change': float(np.mean(reach_diffs)),
                'std_change': float(np.std(reach_diffs)),
                'median_change': float(np.median(reach_diffs)),
                'max_increase': float(np.max(reach_diffs)),
                'min_increase': float(np.min(reach_diffs)),
                'positive_changes': sum(1 for d in reach_diffs if d > 0),
                'negative_changes': sum(1 for d in reach_diffs if d < 0),
                'significant_increases': sum(1 for d in reach_diffs if d > 1.0)
            }
        }
        
        results['top_adj_increases'] = sorted(results['all_connections'], 
                                             key=lambda x: x['adjacency']['diff'], 
                                             reverse=True)[:20]
        
        return results
    
    def analyze_all_stage_transitions(self) -> Dict:
        """分析所有阶段转换的变化"""
        results = {}
        
        stage_pairs = [
            ('Special', 'Special', list(range(2)), list(range(2))),
            ('Special', 'S1', list(range(2)), self.config.token_S1),
            ('Special', 'S2', list(range(2)), self.config.token_S2),
            ('Special', 'S3', list(range(2)), self.config.token_S3),
            ('S1', 'Special', self.config.token_S1, list(range(2))),
            ('S1', 'S1', self.config.token_S1, self.config.token_S1),
            ('S1', 'S2', self.config.token_S1, self.config.token_S2),
            ('S1', 'S3', self.config.token_S1, self.config.token_S3),
            ('S2', 'Special', self.config.token_S2, list(range(2))),
            ('S2', 'S1', self.config.token_S2, self.config.token_S1),
            ('S2', 'S2', self.config.token_S2, self.config.token_S2),
            ('S2', 'S3', self.config.token_S2, self.config.token_S3),
            ('S3', 'Special', self.config.token_S3, list(range(2))),
            ('S3', 'S1', self.config.token_S3, self.config.token_S1),
            ('S3', 'S2', self.config.token_S3, self.config.token_S2),
            ('S3', 'S3', self.config.token_S3, self.config.token_S3),
        ]
        
        for from_stage, to_stage, from_tokens, to_tokens in stage_pairs:
            changes = []
            
            for i in from_tokens:
                for j in to_tokens:
                    diff = self.W_M_20[i, j] - self.W_M_0[i, j]
                    changes.append(diff)
            
            if changes:
                results[f"{from_stage}→{to_stage}"] = {
                    'count': len(changes),
                    'mean_change': float(np.mean(changes)),
                    'std_change': float(np.std(changes)),
                    'max_change': float(np.max(changes)),
                    'min_change': float(np.min(changes)),
                    'median_change': float(np.median(changes)),
                    'significant_increases': sum(1 for c in changes if c > 1.0)
                }
        
        return results
    
    def visualize_comparison_complete(self, save_dir: str = "alpine_analysis"):
        """创建完整的可视化（包含Ground Truth）"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        
        fig = plt.figure(figsize=(20, 14))
        
        # 创建子图布局
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # === 第一行：邻接矩阵 ===
        # Ground Truth
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.config.A_true, cmap='binary', aspect='auto')
        ax1.set_title('Ground Truth\nAdjacency (A_true)', fontweight='bold', fontsize=10)
        ax1.set_xlabel('To Token')
        ax1.set_ylabel('From Token')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 0% Model
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.W_M_0, cmap='RdBu_r', aspect='auto')
        ax2.set_title('0% Model\nAdjacency (W_M_0)', fontweight='bold', fontsize=10)
        ax2.set_xlabel('To Token')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 20% Model
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(self.W_M_20, cmap='RdBu_r', aspect='auto')
        ax3.set_title('20% Model\nAdjacency (W_M_20)', fontweight='bold', fontsize=10)
        ax3.set_xlabel('To Token')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Difference
        ax4 = fig.add_subplot(gs[0, 3])
        diff_adj = self.W_M_20 - self.W_M_0
        vmax = np.abs(diff_adj).max()
        im4 = ax4.imshow(diff_adj, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax4.set_title('Difference\n(20% - 0%)', fontweight='bold', fontsize=10)
        ax4.set_xlabel('To Token')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # === 第二行：可达性矩阵 ===
        # Ground Truth
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(self.config.R_true, cmap='binary', aspect='auto')
        ax5.set_title('Ground Truth\nReachability (R_true)', fontweight='bold', fontsize=10)
        ax5.set_xlabel('To Token')
        ax5.set_ylabel('From Token')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 0% Model
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(self.W_V_0, cmap='RdBu_r', aspect='auto')
        ax6.set_title('0% Model\nReachability (W_V_0)', fontweight='bold', fontsize=10)
        ax6.set_xlabel('To Token')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 20% Model
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = ax7.imshow(self.W_V_20, cmap='RdBu_r', aspect='auto')
        ax7.set_title('20% Model\nReachability (W_V_20)', fontweight='bold', fontsize=10)
        ax7.set_xlabel('To Token')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # Difference
        ax8 = fig.add_subplot(gs[1, 3])
        diff_reach = self.W_V_20 - self.W_V_0
        vmax = np.abs(diff_reach).max()
        im8 = ax8.imshow(diff_reach, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax8.set_title('Difference\n(20% - 0%)', fontweight='bold', fontsize=10)
        ax8.set_xlabel('To Token')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        # === 第三行：注意力模式 ===
        if self.attention_0.shape == (3, 3):
            # 0% Model Attention
            ax9 = fig.add_subplot(gs[2, 0])
            im9 = ax9.imshow(self.attention_0, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax9.set_title('0% Model\nAttention Pattern', fontweight='bold', fontsize=10)
            ax9.set_xticks([0, 1, 2])
            ax9.set_yticks([0, 1, 2])
            ax9.set_xticklabels(['S1', 'S2', 'S3'])
            ax9.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im9, ax=ax9, fraction=0.046)
            
            # 20% Model Attention
            ax10 = fig.add_subplot(gs[2, 1])
            im10 = ax10.imshow(self.attention_20, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax10.set_title('20% Model\nAttention Pattern', fontweight='bold', fontsize=10)
            ax10.set_xticks([0, 1, 2])
            ax10.set_yticks([0, 1, 2])
            ax10.set_xticklabels(['S1', 'S2', 'S3'])
            ax10.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im10, ax=ax10, fraction=0.046)
            
            # Attention Difference
            ax11 = fig.add_subplot(gs[2, 2])
            attn_diff = self.attention_20 - self.attention_0
            vmax = np.abs(attn_diff).max()
            im11 = ax11.imshow(attn_diff, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
            ax11.set_title('Attention Change\n(20% - 0%)', fontweight='bold', fontsize=10)
            ax11.set_xticks([0, 1, 2])
            ax11.set_yticks([0, 1, 2])
            ax11.set_xticklabels(['S1', 'S2', 'S3'])
            ax11.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im11, ax=ax11, fraction=0.046)
        
        # 总标题
        fig.suptitle(f'ALPINE Complete Analysis - Seed {self.config.seed}', 
                     fontsize=16, fontweight='bold')
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, 
                                f'alpine_complete_analysis_seed{self.config.seed}_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 完整可视化已保存: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, save_dir: str = "alpine_analysis"):
        """保存所有结果"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 完整结果
        results = {
            'metadata': {
                'seed': self.config.seed,
                'timestamp': timestamp,
                'model_dimension': self.config.n_embd,
                'vocab_size': self.config.vocab_size
            },
            'ground_truth_comparison': self.analyze_vs_ground_truth(),
            'model_comparison': self.compute_similarities(),
            'self_loops': self.analyze_self_loops(),
            's1_s3_connections': self.analyze_all_s1_s3_connections(),
            'stage_transitions': self.analyze_all_stage_transitions()
        }
        
        json_path = os.path.join(save_dir, f'complete_results_seed{self.config.seed}_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ 完整结果已保存: {json_path}")
        
        return json_path

# ==================== 主函数 ====================
def main():
    """主执行函数"""
    import argparse
    parser = argparse.ArgumentParser(description="ALPINE Matrix Analysis - Complete Version")
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in training')
    parser.add_argument('--checkpoint_dir', type=str, default='out_d92', 
                       help='Directory containing checkpoints')
    parser.add_argument('--save_dir', type=str, default='alpine_analysis', 
                       help='Directory to save results')
    parser.add_argument('--skip_viz', action='store_true', 
                       help='Skip visualization to save time')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 ALPINE Matrix Analysis - Complete Version")
    print("="*80)
    
    try:
        # 创建配置
        config = Config(seed=args.seed, checkpoint_dir=args.checkpoint_dir)
        
        # 运行分析
        comparator = MatrixComparator(config)
        
        # ==================== 核心分析1: Ground Truth验证（教授要求） ====================
        print("\n" + "="*60)
        print("3. Ground Truth Validation (Responding to Professor's Feedback)")
        print("="*60)
        
        gt_results = comparator.analyze_vs_ground_truth()
        
        print("\n📊 Adjacency Matrix vs. Ground Truth:")
        print(f"  0% Model (W_M_0 vs A_true):")
        print(f"    - Cosine similarity: {gt_results['adjacency']['0%_vs_True']['cosine']:.4f}")
        print(f"    - Pearson correlation: {gt_results['adjacency']['0%_vs_True']['pearson']:.4f}")
        print(f"  20% Model (W_M_20 vs A_true):")
        print(f"    - Cosine similarity: {gt_results['adjacency']['20%_vs_True']['cosine']:.4f}")
        print(f"    - Pearson correlation: {gt_results['adjacency']['20%_vs_True']['pearson']:.4f}")
        print(f"  ✨ Improvement:")
        print(f"    - Cosine: {gt_results['adjacency']['improvement']['cosine']:+.4f}")
        print(f"    - Pearson: {gt_results['adjacency']['improvement']['pearson']:+.4f}")
        
        print("\n📊 Reachability Matrix vs. Ground Truth:")
        print(f"  0% Model (W_V_0 vs R_true):")
        print(f"    - Cosine similarity: {gt_results['reachability']['0%_vs_True']['cosine']:.4f}")
        print(f"    - Pearson correlation: {gt_results['reachability']['0%_vs_True']['pearson']:.4f}")
        print(f"  20% Model (W_V_20 vs R_true):")
        print(f"    - Cosine similarity: {gt_results['reachability']['20%_vs_True']['cosine']:.4f}")
        print(f"    - Pearson correlation: {gt_results['reachability']['20%_vs_True']['pearson']:.4f}")
        print(f"  ✨ Improvement:")
        print(f"    - Cosine: {gt_results['reachability']['improvement']['cosine']:+.4f}")
        print(f"    - Pearson: {gt_results['reachability']['improvement']['pearson']:+.4f}")
        
        # ==================== 核心分析2: 0% vs 20%比较 ====================
        print("\n" + "="*60)
        print("4. Model Comparison (0% vs 20%)")
        print("="*60)
        
        similarities = comparator.compute_similarities()
        print("\n📊 Matrix Similarities between 0% and 20%:")
        for matrix_type, metrics in similarities.items():
            print(f"\n{matrix_type.capitalize()} Matrix:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # ==================== 核心分析3: 自循环分析 ====================
        print("\n" + "="*60)
        print("5. Self-Loop Analysis")
        print("="*60)
        
        self_loops = comparator.analyze_self_loops()
        for stage in ['Special', 'S1', 'S2', 'S3']:
            if stage in self_loops['summary']:
                stats = self_loops['summary'][stage]
                print(f"\n{stage} Stage:")
                print(f"  Mean change: {stats['mean_change']:+.3f}")
                print(f"  Max change: {stats['max_change']:+.3f}")
                print(f"  Significant increases (>1.0): {stats['significant_increases']}/{stats['count']}")
        
        # ==================== 核心分析4: S1→S3连接 ====================
        print("\n" + "="*60)
        print("6. S1→S3 Connection Analysis")
        print("="*60)
        
        s1_s3 = comparator.analyze_all_s1_s3_connections()
        print(f"\n📊 Complete S1→S3 Analysis:")
        print(f"  Total connections: {s1_s3['summary']['total_connections']}")
        print(f"  Adjacency mean change: {s1_s3['summary']['adjacency']['mean_change']:+.3f}")
        print(f"  Positive changes: {s1_s3['summary']['adjacency']['positive_changes']}/{s1_s3['summary']['total_connections']}")
        print(f"  Significant increases (>1.0): {s1_s3['summary']['adjacency']['significant_increases']}")
        
        # ==================== 核心分析5: 阶段转换 ====================
        print("\n" + "="*60)
        print("7. Stage Transition Analysis")
        print("="*60)
        
        transitions = comparator.analyze_all_stage_transitions()
        sorted_transitions = sorted(transitions.items(), 
                                   key=lambda x: abs(x[1]['mean_change']), 
                                   reverse=True)
        
        print("\n📊 Top 5 Stage Transitions by Change:")
        for trans_name, stats in sorted_transitions[:5]:
            print(f"  {trans_name}: mean {stats['mean_change']:+.3f}, "
                  f"significant increases {stats['significant_increases']}/{stats['count']}")
        
        # ==================== 可视化 ====================
        if not args.skip_viz:
            print("\n8. Generating Visualizations...")
            comparator.visualize_comparison_complete(save_dir=args.save_dir)
        
        # ==================== 保存结果 ====================
        print("\n9. Saving Results...")
        json_path = comparator.save_results(save_dir=args.save_dir)
        
        print("\n" + "="*80)
        print("✅ Analysis Complete!")
        print("="*80)
        print(f"\n📁 Results saved in: {args.save_dir}/")
        print(f"📊 JSON results: {json_path}")
        
        return comparator
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    comparator = main()