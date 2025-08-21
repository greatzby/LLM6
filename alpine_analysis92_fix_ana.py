"""
ALPINE Matrix Extraction and Comparison for Compositional Learning Analysis
增强版 - 包含全面的自循环分析和所有连接分析
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
    """配置类"""
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
        
        # 加载元数据
        self.load_metadata()
        
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
    
    def load_metadata(self):
        """加载元数据"""
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
        if not os.path.exists(stage_path):
            raise FileNotFoundError(f"未找到stage_info.pkl在: {base_data_dir}")
            
        with open(stage_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.stages = stage_info['stages']
        self.S1, self.S2, self.S3 = self.stages
        
        print(f"✓ 加载stage信息: S1={len(self.S1)}个节点, S2={len(self.S2)}个节点, S3={len(self.S3)}个节点")
        
        # 加载图结构
        graph_path = os.path.join(base_data_dir, 'composition_graph.graphml')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"未找到composition_graph.graphml在: {base_data_dir}")
            
        self.G = nx.read_graphml(graph_path)
        print(f"✓ 加载图结构: {len(self.G.nodes)}个节点, {len(self.G.edges)}条边")
        
        # 验证数据一致性
        self.verify_data_consistency()
    
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
        """提取邻接矩阵表示 W'_M"""
        vocab_size = self.config.vocab_size
        W_M_prime = []
        
        print(f"  提取邻接矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            for node_i in range(vocab_size):
                if node_i % 10 == 0:
                    print(f"    处理节点 {node_i}/{vocab_size}...", end='\r')
                
                # 获取token embedding
                token_emb = self.model.transformer.wte(torch.tensor([node_i], device=self.device))
                token_emb = token_emb.squeeze(0)
                
                # 只通过FFN
                token_emb_expanded = token_emb.unsqueeze(0).unsqueeze(0)
                ffn_out = self.model.transformer.h[0].mlp(token_emb_expanded)
                ffn_out = ffn_out.squeeze()
                
                # 组合：FFN(emb) + emb
                combined = ffn_out + token_emb
                
                # Layer norm
                combined = self.model.transformer.ln_f(combined.unsqueeze(0)).squeeze()
                
                # 投影到词汇表
                if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
                    if hasattr(self.model.lm_head, 'weight') and self.model.lm_head.weight is not None:
                        output_weight = self.model.lm_head.weight
                    else:
                        output_weight = self.model.transformer.wte.weight
                else:
                    output_weight = self.model.transformer.wte.weight
                
                logits = combined @ output_weight.T
                
                row_i = logits.cpu().numpy()
                W_M_prime.append(row_i)
        
        print(f"\n    完成! Shape: {len(W_M_prime)}x{len(W_M_prime[0])}")
        return np.array(W_M_prime)
    
    def extract_reachability_matrix(self) -> np.ndarray:
        """提取可达矩阵表示 W'_V"""
        vocab_size = self.config.vocab_size
        n_embd = self.config.n_embd
        W_V_prime = []
        
        print(f"  提取可达矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            # 获取attention层的c_attn权重
            c_attn = self.model.transformer.h[0].attn.c_attn
            c_attn_weight = c_attn.weight
            
            # 提取Value矩阵
            W_V = c_attn_weight[2*n_embd:3*n_embd, :]
            print(f"    Value矩阵形状: {W_V.shape}")
            
            for target_node in range(vocab_size):
                if target_node % 10 == 0:
                    print(f"    处理目标节点 {target_node}/{vocab_size}...", end='\r')
                
                # 获取目标节点的embedding
                target_emb = self.model.transformer.wte(torch.tensor([target_node], device=self.device))
                target_emb = target_emb.squeeze(0)
                
                # 通过Value矩阵变换
                value_features = target_emb @ W_V.T
                
                # 通过FFN
                value_features_expanded = value_features.unsqueeze(0).unsqueeze(0)
                ffn_out = self.model.transformer.h[0].mlp(value_features_expanded)
                ffn_out = ffn_out.squeeze()
                
                # 组合
                combined = value_features + ffn_out
                
                # Layer norm
                combined = self.model.transformer.ln_f(combined.unsqueeze(0)).squeeze()
                
                # 投影到词汇表空间
                if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
                    if hasattr(self.model.lm_head, 'weight') and self.model.lm_head.weight is not None:
                        output_weight = self.model.lm_head.weight
                    else:
                        output_weight = self.model.transformer.wte.weight
                else:
                    output_weight = self.model.transformer.wte.weight
                
                reachability_scores = combined @ output_weight.T
                
                row_i = reachability_scores.cpu().numpy()
                W_V_prime.append(row_i)
        
        print(f"\n    完成! Shape: {len(W_V_prime)}x{len(W_V_prime[0])}")
        return np.array(W_V_prime)
    
    def extract_attention_pattern(self, num_samples: int = 50) -> np.ndarray:
        """手动提取注意力模式"""
        print(f"  提取注意力模式 ({self.model_type})...")
        
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        
        attention_maps = []
        
        try:
            with torch.no_grad():
                # 获取c_attn权重
                c_attn = self.model.transformer.h[0].attn.c_attn
                c_attn_weight = c_attn.weight
                
                # 分离Q、K、V投影矩阵
                W_Q = c_attn_weight[:n_embd, :]
                W_K = c_attn_weight[n_embd:2*n_embd, :]
                
                for i in range(min(num_samples, 50)):
                    # 使用正确的token索引
                    source = np.random.choice(self.config.token_S1)
                    target = np.random.choice(self.config.token_S3)
                    middle = np.random.choice(self.config.token_S2)
                    
                    # 构建输入序列
                    input_ids = torch.tensor([source, middle, target], dtype=torch.long, device=self.device)
                    
                    # 获取embeddings
                    token_emb = self.model.transformer.wte(input_ids)
                    positions = torch.arange(0, 3, dtype=torch.long, device=self.device)
                    pos_emb = self.model.transformer.wpe(positions)
                    
                    hidden = token_emb + pos_emb
                    
                    # 计算Q和K
                    Q = hidden @ W_Q.T
                    K = hidden @ W_K.T
                    
                    # Reshape for attention
                    Q = Q.view(3, n_head, head_dim)
                    K = K.view(3, n_head, head_dim)
                    
                    # 计算注意力分数
                    Q = Q.squeeze(1)
                    K = K.squeeze(1)
                    
                    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
                    attn_scores = attn_scores / math.sqrt(head_dim)
                    
                    # 应用因果掩码
                    seq_len = 3
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                    
                    # Softmax
                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    
                    attention_maps.append(attn_weights.cpu().numpy())
            
            if attention_maps:
                avg_attention = np.mean(attention_maps, axis=0)
                print(f"    完成! Shape: {avg_attention.shape}")
                print(f"    成功提取了{len(attention_maps)}个样本的注意力模式")
                return avg_attention
            else:
                print(f"    ⚠️ 未能提取任何注意力模式，返回零矩阵")
                return np.zeros((3, 3))
                
        except Exception as e:
            print(f"    注意力提取出错: {e}")
            import traceback
            traceback.print_exc()
            print(f"    返回零矩阵")
            return np.zeros((3, 3))

# ==================== 矩阵比较和分析器（增强版） ====================
class MatrixComparator:
    """矩阵比较和可视化工具 - 增强版"""
    
    def __init__(self, config: Config):
        self.config = config
        
        print("\n" + "="*60)
        print("ALPINE矩阵提取和比较分析")
        print("="*60)
        print(f"Seed: {config.seed}")
        print(f"Model dimension: {config.n_embd}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Stages: S1={len(config.S1)}, S2={len(config.S2)}, S3={len(config.S3)}")
        
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
        """计算矩阵相似度"""
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
    
    def analyze_attention_patterns(self) -> Dict:
        """分析注意力模式的详细信息"""
        results = {}
        
        if self.attention_0.shape == (3, 3) and self.attention_20.shape == (3, 3):
            results['0%_model'] = {
                'position_0_attention': {
                    'to_pos_0': float(self.attention_0[0, 0]),
                },
                'position_1_attention': {
                    'to_pos_0': float(self.attention_0[1, 0]),
                    'to_pos_1': float(self.attention_0[1, 1]),
                },
                'position_2_attention': {
                    'to_pos_0': float(self.attention_0[2, 0]),
                    'to_pos_1': float(self.attention_0[2, 1]),
                    'to_pos_2': float(self.attention_0[2, 2]),
                }
            }
            
            results['20%_model'] = {
                'position_0_attention': {
                    'to_pos_0': float(self.attention_20[0, 0]),
                },
                'position_1_attention': {
                    'to_pos_0': float(self.attention_20[1, 0]),
                    'to_pos_1': float(self.attention_20[1, 1]),
                },
                'position_2_attention': {
                    'to_pos_0': float(self.attention_20[2, 0]),
                    'to_pos_1': float(self.attention_20[2, 1]),
                    'to_pos_2': float(self.attention_20[2, 2]),
                }
            }
            
            diff = self.attention_20 - self.attention_0
            results['changes'] = {
                'max_increase': float(np.max(diff)),
                'max_decrease': float(np.min(diff)),
                'mean_change': float(np.mean(diff)),
                'attention_to_first_token_change': float(diff[:, 0].mean()),
                'self_attention_change': float(np.mean([diff[i, i] for i in range(3)])),
            }
            
        return results
    
    def analyze_compositional_paths(self) -> Dict:
        """分析S1→S3的组合路径"""
        results = {'paths': {}, 'summary': {}}
        
        sample_s1_tokens = self.config.token_S1[:min(5, len(self.config.token_S1))]
        sample_s3_tokens = self.config.token_S3[:min(5, len(self.config.token_S3))]
        
        adj_diffs = []
        reach_diffs = []
        
        for s1_token in sample_s1_tokens:
            for s3_token in sample_s3_tokens:
                s1_node = s1_token - 2
                s3_node = s3_token - 2
                
                adj_0 = self.W_M_0[s1_token, s3_token]
                adj_20 = self.W_M_20[s1_token, s3_token]
                adj_diff = adj_20 - adj_0
                
                reach_0 = self.W_V_0[s3_token, s1_token]
                reach_20 = self.W_V_20[s3_token, s1_token]
                reach_diff = reach_20 - reach_0
                
                path_key = f"Node{s1_node}→Node{s3_node}"
                results['paths'][path_key] = {
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
                
                adj_diffs.append(adj_diff)
                reach_diffs.append(reach_diff)
        
        results['summary'] = {
            'num_paths_analyzed': len(adj_diffs),
            'adjacency': {
                'mean_increase': float(np.mean(adj_diffs)),
                'std_increase': float(np.std(adj_diffs)),
                'max_increase': float(np.max(adj_diffs)),
                'min_increase': float(np.min(adj_diffs)),
                'significant_increases': int(sum(1 for d in adj_diffs if d > 1.0))
            },
            'reachability': {
                'mean_increase': float(np.mean(reach_diffs)),
                'std_increase': float(np.std(reach_diffs)),
                'max_increase': float(np.max(reach_diffs)),
                'min_increase': float(np.min(reach_diffs)),
                'significant_increases': int(sum(1 for d in reach_diffs if d > 1.0))
            }
        }
        
        return results
    
    def find_key_differences(self, top_k: int = 20) -> Dict:
        """找出最大的差异"""
        differences = {}
        
        adj_diff = self.W_M_20 - self.W_M_0
        adj_abs_diff = np.abs(adj_diff)
        
        flat_indices = np.argsort(adj_abs_diff.ravel())[-top_k:][::-1]
        top_indices = np.unravel_index(flat_indices, adj_abs_diff.shape)
        
        differences['adjacency_top_changes'] = []
        for i, j in zip(top_indices[0], top_indices[1]):
            if i < 2:
                from_stage = 'Special'
            elif i < 32:
                from_stage = 'S1'
            elif i < 62:
                from_stage = 'S2'
            else:
                from_stage = 'S3'
            
            if j < 2:
                to_stage = 'Special'
            elif j < 32:
                to_stage = 'S1'
            elif j < 62:
                to_stage = 'S2'
            else:
                to_stage = 'S3'
            
            differences['adjacency_top_changes'].append({
                'from': int(i),
                'to': int(j),
                'from_stage': from_stage,
                'to_stage': to_stage,
                'path_type': f"{from_stage}→{to_stage}",
                'weight_0%': float(self.W_M_0[i, j]),
                'weight_20%': float(self.W_M_20[i, j]),
                'change': float(adj_diff[i, j])
            })
        
        path_type_stats = {}
        for change in differences['adjacency_top_changes']:
            path_type = change['path_type']
            if path_type not in path_type_stats:
                path_type_stats[path_type] = []
            path_type_stats[path_type].append(change['change'])
        
        differences['path_type_statistics'] = {}
        for path_type, changes in path_type_stats.items():
            differences['path_type_statistics'][path_type] = {
                'count': len(changes),
                'mean_change': float(np.mean(changes)),
                'max_change': float(np.max(np.abs(changes)))
            }
        
        return differences
    
    # ==================== 新增的分析方法 ====================
    
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
                'node': i - 2 if i >= 2 else i,  # 转换为图节点编号
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
        
        # 找出变化最大的自循环
        results['top_increases'] = sorted(results['all_self_loops'], 
                                         key=lambda x: x['change'], 
                                         reverse=True)[:10]
        results['top_decreases'] = sorted(results['all_self_loops'], 
                                         key=lambda x: x['change'])[:10]
        
        return results
    
    def analyze_all_s1_s3_connections(self) -> Dict:
        """分析所有S1→S3连接的变化"""
        results = {
            'all_connections': [],
            'summary': {},
            'distribution': {}
        }
        
        # 分析所有S1→S3连接
        for s1_token in self.config.token_S1:  # tokens 2-31
            for s3_token in self.config.token_S3:  # tokens 62-91
                s1_node = s1_token - 2
                s3_node = s3_token - 2
                
                # 邻接矩阵权重
                adj_0 = self.W_M_0[s1_token, s3_token]
                adj_20 = self.W_M_20[s1_token, s3_token]
                adj_diff = adj_20 - adj_0
                
                # 可达矩阵权重
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
                'significant_increases': sum(1 for d in adj_diffs if d > 1.0),
                'significant_decreases': sum(1 for d in adj_diffs if d < -1.0)
            },
            'reachability': {
                'mean_change': float(np.mean(reach_diffs)),
                'std_change': float(np.std(reach_diffs)),
                'median_change': float(np.median(reach_diffs)),
                'max_increase': float(np.max(reach_diffs)),
                'min_increase': float(np.min(reach_diffs)),
                'positive_changes': sum(1 for d in reach_diffs if d > 0),
                'negative_changes': sum(1 for d in reach_diffs if d < 0),
                'significant_increases': sum(1 for d in reach_diffs if d > 1.0),
                'significant_decreases': sum(1 for d in reach_diffs if d < -1.0)
            }
        }
        
        # 分布分析
        results['distribution'] = {
            'adjacency_bins': np.histogram(adj_diffs, bins=20),
            'reachability_bins': np.histogram(reach_diffs, bins=20)
        }
        
        # Top变化
        results['top_adj_increases'] = sorted(results['all_connections'], 
                                             key=lambda x: x['adjacency']['diff'], 
                                             reverse=True)[:20]
        results['top_adj_decreases'] = sorted(results['all_connections'], 
                                             key=lambda x: x['adjacency']['diff'])[:20]
        
        return results
    
    def analyze_all_stage_transitions(self) -> Dict:
        """分析所有阶段转换的变化"""
        results = {}
        
        # 定义所有可能的阶段转换
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
                    'significant_increases': sum(1 for c in changes if c > 1.0),
                    'significant_decreases': sum(1 for c in changes if c < -1.0)
                }
        
        return results
    
    def find_key_differences_extended(self, top_k: int = 100) -> Dict:
        """扩展版：找出更多的关键差异"""
        differences = {}
        
        # 邻接矩阵差异
        adj_diff = self.W_M_20 - self.W_M_0
        adj_abs_diff = np.abs(adj_diff)
        
        # 找出变化最大的位置（增加到top_k个）
        flat_indices = np.argsort(adj_abs_diff.ravel())[-top_k:][::-1]
        top_indices = np.unravel_index(flat_indices, adj_abs_diff.shape)
        
        differences['adjacency_top_changes'] = []
        for i, j in zip(top_indices[0], top_indices[1]):
            # 正确分类token
            if i < 2:
                from_stage = 'Special'
            elif i < 32:
                from_stage = 'S1'
            elif i < 62:
                from_stage = 'S2'
            else:
                from_stage = 'S3'
            
            if j < 2:
                to_stage = 'Special'
            elif j < 32:
                to_stage = 'S1'
            elif j < 62:
                to_stage = 'S2'
            else:
                to_stage = 'S3'
            
            # 检查是否是自循环
            is_self_loop = (i == j)
            
            differences['adjacency_top_changes'].append({
                'from': int(i),
                'to': int(j),
                'from_stage': from_stage,
                'to_stage': to_stage,
                'path_type': f"{from_stage}→{to_stage}",
                'is_self_loop': is_self_loop,
                'weight_0%': float(self.W_M_0[i, j]),
                'weight_20%': float(self.W_M_20[i, j]),
                'change': float(adj_diff[i, j])
            })
        
        # 分别统计自循环和非自循环
        self_loops = [c for c in differences['adjacency_top_changes'] if c['is_self_loop']]
        non_self_loops = [c for c in differences['adjacency_top_changes'] if not c['is_self_loop']]
        
        differences['self_loop_changes'] = self_loops
        differences['non_self_loop_changes'] = non_self_loops
        
        # 路径类型统计（更详细）
        path_type_stats = {}
        for change in differences['adjacency_top_changes']:
            path_type = change['path_type']
            if path_type not in path_type_stats:
                path_type_stats[path_type] = {
                    'all_changes': [],
                    'self_loops': [],
                    'non_self_loops': []
                }
            
            path_type_stats[path_type]['all_changes'].append(change['change'])
            if change['is_self_loop']:
                path_type_stats[path_type]['self_loops'].append(change['change'])
            else:
                path_type_stats[path_type]['non_self_loops'].append(change['change'])
        
        differences['path_type_statistics'] = {}
        for path_type, data in path_type_stats.items():
            stats = {}
            
            # 总体统计
            if data['all_changes']:
                stats['total'] = {
                    'count': len(data['all_changes']),
                    'mean_change': float(np.mean(data['all_changes'])),
                    'std_change': float(np.std(data['all_changes'])),
                    'max_change': float(np.max(np.abs(data['all_changes'])))
                }
            
            # 自循环统计
            if data['self_loops']:
                stats['self_loops'] = {
                    'count': len(data['self_loops']),
                    'mean_change': float(np.mean(data['self_loops'])),
                    'max_change': float(np.max(np.abs(data['self_loops'])))
                }
            
            # 非自循环统计
            if data['non_self_loops']:
                stats['non_self_loops'] = {
                    'count': len(data['non_self_loops']),
                    'mean_change': float(np.mean(data['non_self_loops'])),
                    'max_change': float(np.max(np.abs(data['non_self_loops'])))
                }
            
            differences['path_type_statistics'][path_type] = stats
        
        return differences
    
    def generate_detailed_report(self) -> str:
        """生成详细的分析报告"""
        report = []
        report.append("="*80)
        report.append("DETAILED ALPINE ANALYSIS REPORT")
        report.append("="*80)
        
        # 1. 自循环分析
        report.append("\n" + "="*60)
        report.append("1. SELF-LOOP ANALYSIS (对角线元素)")
        report.append("="*60)
        
        self_loops = self.analyze_self_loops()
        
        for stage in ['Special', 'S1', 'S2', 'S3']:
            if stage in self_loops['summary']:
                stats = self_loops['summary'][stage]
                report.append(f"\n{stage} Stage Self-Loops:")
                report.append(f"  Nodes analyzed: {stats['count']}")
                report.append(f"  Mean change: {stats['mean_change']:+.3f}")
                report.append(f"  Std change: {stats['std_change']:.3f}")
                report.append(f"  Max change: {stats['max_change']:+.3f}")
                report.append(f"  Min change: {stats['min_change']:+.3f}")
                report.append(f"  Significant increases (>1.0): {stats['significant_increases']}")
                report.append(f"  Significant decreases (<-1.0): {stats['significant_decreases']}")
        
        report.append("\nTop 10 Self-Loop Increases:")
        for item in self_loops['top_increases']:
            report.append(f"  {item['stage']} Node{item['node']}: {item['change']:+.3f} "
                         f"({item['weight_0%']:.2f} → {item['weight_20%']:.2f})")
        
        # 2. S1→S3完整分析
        report.append("\n" + "="*60)
        report.append("2. COMPLETE S1→S3 CONNECTION ANALYSIS")
        report.append("="*60)
        
        s1_s3 = self.analyze_all_s1_s3_connections()
        
        report.append(f"\nTotal S1→S3 connections: {s1_s3['summary']['total_connections']}")
        
        report.append("\nAdjacency Matrix Changes:")
        adj_stats = s1_s3['summary']['adjacency']
        report.append(f"  Mean change: {adj_stats['mean_change']:+.3f}")
        report.append(f"  Std change: {adj_stats['std_change']:.3f}")
        report.append(f"  Median change: {adj_stats['median_change']:+.3f}")
        report.append(f"  Max increase: {adj_stats['max_increase']:+.3f}")
        report.append(f"  Min increase: {adj_stats['min_increase']:+.3f}")
        report.append(f"  Positive changes: {adj_stats['positive_changes']}/{s1_s3['summary']['total_connections']}")
        report.append(f"  Negative changes: {adj_stats['negative_changes']}/{s1_s3['summary']['total_connections']}")
        report.append(f"  Significant increases (>1.0): {adj_stats['significant_increases']}")
        
        report.append("\nTop 10 S1→S3 Adjacency Increases:")
        for item in s1_s3['top_adj_increases'][:10]:
            report.append(f"  {item['path']}: {item['adjacency']['diff']:+.3f} "
                         f"({item['adjacency']['0%']:.2f} → {item['adjacency']['20%']:.2f})")
        
        # 3. 所有阶段转换分析
        report.append("\n" + "="*60)
        report.append("3. ALL STAGE TRANSITIONS ANALYSIS")
        report.append("="*60)
        
        transitions = self.analyze_all_stage_transitions()
        
        # 按平均变化排序
        sorted_transitions = sorted(transitions.items(), 
                                   key=lambda x: abs(x[1]['mean_change']), 
                                   reverse=True)
        
        report.append("\nTop Stage Transitions by Mean Change:")
        for trans_name, stats in sorted_transitions[:10]:
            report.append(f"\n{trans_name}:")
            report.append(f"  Connections: {stats['count']}")
            report.append(f"  Mean change: {stats['mean_change']:+.3f}")
            report.append(f"  Std change: {stats['std_change']:.3f}")
            report.append(f"  Max change: {stats['max_change']:+.3f}")
            report.append(f"  Significant increases: {stats['significant_increases']}")
        
        # 4. 扩展的关键差异分析
        report.append("\n" + "="*60)
        report.append("4. EXTENDED KEY DIFFERENCES (Top 100)")
        report.append("="*60)
        
        extended_diff = self.find_key_differences_extended(top_k=100)
        
        report.append(f"\nTotal changes analyzed: {len(extended_diff['adjacency_top_changes'])}")
        report.append(f"Self-loops in top 100: {len(extended_diff['self_loop_changes'])}")
        report.append(f"Non-self-loops in top 100: {len(extended_diff['non_self_loop_changes'])}")
        
        report.append("\nPath Type Distribution in Top 100:")
        for path_type, stats in extended_diff['path_type_statistics'].items():
            if 'total' in stats:
                report.append(f"\n{path_type}:")
                report.append(f"  Total: {stats['total']['count']}")
                if 'self_loops' in stats:
                    report.append(f"  Self-loops: {stats['self_loops']['count']} "
                                 f"(mean: {stats['self_loops']['mean_change']:+.3f})")
                if 'non_self_loops' in stats:
                    report.append(f"  Non-self-loops: {stats['non_self_loops']['count']} "
                                 f"(mean: {stats['non_self_loops']['mean_change']:+.3f})")
        
        return "\n".join(report)
    
    def visualize_comparison(self, save_dir: str = "alpine_analysis"):
        """创建完整92×92矩阵可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.3)
        
        display_size = self.W_M_0.shape[0]  # 92
        boundaries = [0, 2, 32, 62, 92]
        
        # 第一行：邻接矩阵
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.W_M_0, cmap='RdBu_r', aspect='auto')
        ax1.set_title('0% Model - Adjacency (92×92)', fontweight='bold', fontsize=10)
        ax1.set_xlabel('To Token', fontsize=9)
        ax1.set_ylabel('From Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax1.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax1.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.W_M_20, cmap='RdBu_r', aspect='auto')
        ax2.set_title('20% Model - Adjacency (92×92)', fontweight='bold', fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff_adj = self.W_M_20 - self.W_M_0
        vmax = np.abs(diff_adj).max()
        im3 = ax3.imshow(diff_adj, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax3.set_title('Difference (20% - 0%)', fontweight='bold', fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'alpine_full_matrix_seed{self.config.seed}_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 可视化已保存: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, save_dir: str = "alpine_analysis"):
        """保存所有结果"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON结果
        results = {
            'metadata': {
                'seed': self.config.seed,
                'timestamp': timestamp,
            },
            'similarities': self.compute_similarities(),
            'compositional_analysis': self.analyze_compositional_paths(),
            'key_differences': self.find_key_differences(),
            'attention_analysis': self.analyze_attention_patterns()
        }
        
        json_path = os.path.join(save_dir, f'results_seed{self.config.seed}_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ JSON结果已保存: {json_path}")
        
        return json_path

# ==================== 主函数 ====================
def main():
    """主执行函数"""
    import argparse
    parser = argparse.ArgumentParser(description="ALPINE Matrix Extraction and Comparison")
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in training')
    parser.add_argument('--checkpoint_dir', type=str, default='out_d92', help='Directory containing checkpoints')
    parser.add_argument('--save_dir', type=str, default='alpine_analysis', help='Directory to save results')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualization to save time')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 ALPINE Matrix Analysis Starting...")
    print("="*60)
    
    try:
        # 创建配置
        config = Config(seed=args.seed, checkpoint_dir=args.checkpoint_dir)
        
        # 运行分析
        comparator = MatrixComparator(config)
        
        # 显示结果
        print("\n" + "="*60)
        print("3. 基础分析结果")
        print("="*60)
        
        # 相似度
        similarities = comparator.compute_similarities()
        print("\n📊 相似度指标:")
        for matrix_type, metrics in similarities.items():
            print(f"\n{matrix_type.capitalize()} Matrix:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # ==================== 增强分析部分 ====================
        print("\n" + "="*60)
        print("4. 增强分析结果")
        print("="*60)
        
        # 1. 自循环分析
        print("\n🔄 自循环分析:")
        self_loops = comparator.analyze_self_loops()
        for stage in ['Special', 'S1', 'S2', 'S3']:
            if stage in self_loops['summary']:
                stats = self_loops['summary'][stage]
                print(f"\n{stage} 阶段:")
                print(f"  平均变化: {stats['mean_change']:+.3f}")
                print(f"  最大变化: {stats['max_change']:+.3f}")
                print(f"  显著增加(>1.0): {stats['significant_increases']}/{stats['count']}")
        
        # 2. 完整S1→S3分析
        print("\n🔗 完整S1→S3分析:")
        s1_s3 = comparator.analyze_all_s1_s3_connections()
        print(f"  总连接数: {s1_s3['summary']['total_connections']}")
        print(f"  邻接矩阵平均变化: {s1_s3['summary']['adjacency']['mean_change']:+.3f}")
        print(f"  正向变化: {s1_s3['summary']['adjacency']['positive_changes']}/{s1_s3['summary']['total_connections']}")
        print(f"  显著增加(>1.0): {s1_s3['summary']['adjacency']['significant_increases']}")
        
        # 3. 阶段转换分析
        print("\n🔀 阶段转换分析:")
        transitions = comparator.analyze_all_stage_transitions()
        sorted_transitions = sorted(transitions.items(), 
                                   key=lambda x: abs(x[1]['mean_change']), 
                                   reverse=True)
        
        print("  Top 5 变化最大的转换:")
        for trans_name, stats in sorted_transitions[:5]:
            print(f"    {trans_name}: 平均{stats['mean_change']:+.3f}, 显著增加{stats['significant_increases']}")
        
        # 4. 扩展差异分析
        print("\n🎯 扩展差异分析 (Top 100):")
        extended_diff = comparator.find_key_differences_extended(top_k=100)
        print(f"  自循环: {len(extended_diff['self_loop_changes'])}")
        print(f"  非自循环: {len(extended_diff['non_self_loop_changes'])}")
        
        # 生成详细报告
        detailed_report = comparator.generate_detailed_report()
        
        # 保存详细报告
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_path = os.path.join(args.save_dir, f'detailed_report_seed{comparator.config.seed}_{timestamp}.txt')
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print(f"\n✓ 详细报告已保存: {detailed_path}")
        
        # 可视化（可选）
        if not args.skip_viz:
            print("\n5. 生成可视化...")
            comparator.visualize_comparison(save_dir=args.save_dir)
        
        # 保存结果
        print("\n6. 保存结果...")
        json_path = comparator.save_results(save_dir=args.save_dir)
        
        print("\n" + "="*60)
        print("✅ 分析完成！")
        print("="*60)
        
        return comparator
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    comparator = main()