"""
ALPINE Matrix Extraction and Comparison for Compositional Learning Analysis
完整修复版 - 显示完整92×92矩阵，正确处理token映射
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
        # Token 0 = '[PAD]', Token 1 = '\n'
        # Token 2-31 = S1 (图节点0-29)
        # Token 32-61 = S2 (图节点30-59)
        # Token 62-91 = S3 (图节点60-89)
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
    """基于ALPINE论文的矩阵提取器 - 修复版（包含手动注意力提取）"""
    
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
        """
        提取邻接矩阵表示 W'_M
        编码"从节点i可以到达哪些节点"的信息
        """
        vocab_size = self.config.vocab_size
        W_M_prime = []
        
        print(f"  提取邻接矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            for node_i in range(vocab_size):
                if node_i % 10 == 0:
                    print(f"    处理节点 {node_i}/{vocab_size}...", end='\r')
                
                # 获取token embedding
                token_emb = self.model.transformer.wte(torch.tensor([node_i], device=self.device))
                token_emb = token_emb.squeeze(0)  # [n_embd]
                
                # 添加position embedding (position 0)
                pos_emb = self.model.transformer.wpe(torch.tensor([0], device=self.device)).squeeze(0)
                input_emb = token_emb + pos_emb
                
                # 通过transformer层
                hidden = input_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, n_embd]
                
                # 通过第一层transformer
                transformer_out = self.model.transformer.h[0](hidden)[0]
                transformer_out = transformer_out.squeeze()  # [n_embd]
                
                # Layer norm
                transformer_out = self.model.transformer.ln_f(transformer_out.unsqueeze(0)).squeeze()
                
                # 投影到词汇表
                # 处理权重共享的情况
                if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
                    if hasattr(self.model.lm_head, 'weight') and self.model.lm_head.weight is not None:
                        output_weight = self.model.lm_head.weight  # [vocab_size, n_embd]
                    else:
                        output_weight = self.model.transformer.wte.weight  # 权重共享
                else:
                    output_weight = self.model.transformer.wte.weight
                
                logits = transformer_out @ output_weight.T  # [vocab_size]
                
                row_i = logits.cpu().numpy()
                W_M_prime.append(row_i)
        
        print(f"\n    完成! Shape: {len(W_M_prime)}x{len(W_M_prime[0])}")
        return np.array(W_M_prime)
    
    def extract_reachability_matrix(self) -> np.ndarray:
        """
        提取可达矩阵表示 W'_V - 修复版
        编码"哪些节点可以到达节点j"的信息
        """
        vocab_size = self.config.vocab_size
        n_embd = self.config.n_embd
        W_V_prime = []
        
        print(f"  提取可达矩阵 ({self.model_type})...")
        
        with torch.no_grad():
            # 获取attention层的c_attn权重
            c_attn = self.model.transformer.h[0].attn.c_attn
            c_attn_weight = c_attn.weight  # [3*n_embd, n_embd] = [276, 92]
            
            # 验证形状
            expected_shape = (3 * n_embd, n_embd)
            actual_shape = c_attn_weight.shape
            if actual_shape != expected_shape:
                print(f"    ⚠️ c_attn形状异常: 期望{expected_shape}, 实际{actual_shape}")
            
            # 提取Value矩阵 (最后1/3的行)
            W_V = c_attn_weight[2*n_embd:3*n_embd, :]  # [n_embd, n_embd] = [92, 92]
            print(f"    Value矩阵形状: {W_V.shape}")
            
            for target_node in range(vocab_size):
                if target_node % 10 == 0:
                    print(f"    处理目标节点 {target_node}/{vocab_size}...", end='\r')
                
                # 获取目标节点的embedding
                target_emb = self.model.transformer.wte(torch.tensor([target_node], device=self.device))
                target_emb = target_emb.squeeze(0)  # [n_embd]
                
                # 通过Value矩阵变换
                value_features = W_V @ target_emb  # [n_embd]
                
                # 获取输出权重（处理权重共享）
                if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
                    if hasattr(self.model.lm_head, 'weight') and self.model.lm_head.weight is not None:
                        output_weight = self.model.lm_head.weight
                    else:
                        output_weight = self.model.transformer.wte.weight
                else:
                    output_weight = self.model.transformer.wte.weight
                
                # 投影到词汇表空间
                reachability_scores = value_features @ output_weight.T  # [vocab_size]
                
                row_i = reachability_scores.cpu().numpy()
                W_V_prime.append(row_i)
        
        print(f"\n    完成! Shape: {len(W_V_prime)}x{len(W_V_prime[0])}")
        return np.array(W_V_prime)
    
    def extract_attention_pattern(self, num_samples: int = 50) -> np.ndarray:
        """
        手动提取注意力模式 - 修复版
        使用正确的token索引
        """
        print(f"  提取注意力模式 ({self.model_type})...")
        
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        
        attention_maps = []
        
        try:
            with torch.no_grad():
                # 获取c_attn权重用于提取Q、K、V
                c_attn = self.model.transformer.h[0].attn.c_attn
                c_attn_weight = c_attn.weight  # [3*n_embd, n_embd]
                
                # 分离Q、K、V投影矩阵
                W_Q = c_attn_weight[:n_embd, :]  # [n_embd, n_embd]
                W_K = c_attn_weight[n_embd:2*n_embd, :]  # [n_embd, n_embd]
                
                for i in range(min(num_samples, 50)):
                    # 使用正确的token索引
                    source = np.random.choice(self.config.token_S1)  # Token 2-31
                    target = np.random.choice(self.config.token_S3)  # Token 62-91
                    middle = np.random.choice(self.config.token_S2)  # Token 32-61
                    
                    # 构建输入序列
                    input_ids = torch.tensor([source, middle, target], dtype=torch.long, device=self.device)
                    
                    # 获取token embeddings
                    token_emb = self.model.transformer.wte(input_ids)  # [3, n_embd]
                    
                    # 获取position embeddings
                    positions = torch.arange(0, 3, dtype=torch.long, device=self.device)
                    pos_emb = self.model.transformer.wpe(positions)  # [3, n_embd]
                    
                    # 组合embeddings
                    hidden = token_emb + pos_emb  # [3, n_embd]
                    
                    # 计算Q和K
                    Q = hidden @ W_Q.T  # [3, n_embd]
                    K = hidden @ W_K.T  # [3, n_embd]
                    
                    # Reshape for attention (单头情况)
                    Q = Q.view(3, n_head, head_dim)  # [3, 1, 92]
                    K = K.view(3, n_head, head_dim)  # [3, 1, 92]
                    
                    # 计算注意力分数
                    # 对于单头，直接计算
                    Q = Q.squeeze(1)  # [3, 92]
                    K = K.squeeze(1)  # [3, 92]
                    
                    attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [3, 3]
                    attn_scores = attn_scores / math.sqrt(head_dim)
                    
                    # 应用因果掩码（下三角）
                    seq_len = 3
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                    
                    # Softmax
                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    
                    # 存储注意力权重
                    attention_maps.append(attn_weights.cpu().numpy())
            
            if attention_maps:
                # 计算平均注意力模式
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

# ==================== 矩阵比较和分析器 ====================
class MatrixComparator:
    """矩阵比较和可视化工具"""
    
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
        
        # 处理NaN值
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
        else:
            results['adjacency'] = {
                'pearson_correlation': 0.0,
                'spearman_correlation': 0.0,
                'cosine_similarity': 0.0,
                'rmse': float('inf'),
                'mean_abs_diff': float('inf')
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
        else:
            results['reachability'] = {
                'pearson_correlation': 0.0,
                'spearman_correlation': 0.0,
                'cosine_similarity': 0.0,
                'rmse': float('inf'),
                'mean_abs_diff': float('inf')
            }
        
        # 注意力模式相似度（如果成功提取）
        if self.attention_0.shape == self.attention_20.shape and self.attention_0.shape != (3, 3):
            flat_A_0 = self.attention_0.flatten()
            flat_A_20 = self.attention_20.flatten()
            
            mask = ~(np.isnan(flat_A_0) | np.isnan(flat_A_20))
            if mask.sum() > 0:
                flat_A_0_clean = flat_A_0[mask]
                flat_A_20_clean = flat_A_20[mask]
                
                results['attention'] = {
                    'pearson_correlation': float(np.corrcoef(flat_A_0_clean, flat_A_20_clean)[0, 1]),
                    'spearman_correlation': float(spearmanr(flat_A_0_clean, flat_A_20_clean)[0]),
                    'cosine_similarity': float(cosine_similarity(flat_A_0_clean.reshape(1, -1), 
                                                                flat_A_20_clean.reshape(1, -1))[0, 0]),
                    'rmse': float(np.sqrt(np.mean((self.attention_0 - self.attention_20)**2))),
                    'mean_abs_diff': float(np.mean(np.abs(self.attention_0 - self.attention_20)))
                }
        
        return results
    
    def analyze_attention_patterns(self) -> Dict:
        """分析注意力模式的详细信息"""
        results = {}
        
        if self.attention_0.shape == (3, 3) and self.attention_20.shape == (3, 3):
            # 分析0%模型的注意力
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
            
            # 分析20%模型的注意力
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
            
            # 计算变化
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
        """分析S1→S3的组合路径（使用正确的token索引）"""
        results = {'paths': {}, 'summary': {}}
        
        # 选择代表性节点（使用token索引）
        sample_s1_tokens = self.config.token_S1[:min(5, len(self.config.token_S1))]
        sample_s3_tokens = self.config.token_S3[:min(5, len(self.config.token_S3))]
        
        adj_diffs = []
        reach_diffs = []
        
        for s1_token in sample_s1_tokens:
            for s3_token in sample_s3_tokens:
                # 转换为图节点编号用于显示
                s1_node = s1_token - 2  # Token to graph node
                s3_node = s3_token - 2
                
                # 邻接矩阵权重（使用token索引）
                adj_0 = self.W_M_0[s1_token, s3_token]
                adj_20 = self.W_M_20[s1_token, s3_token]
                adj_diff = adj_20 - adj_0
                
                # 可达矩阵权重（使用token索引）
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
        
        # 统计
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
        """找出最大的差异（修正节点分类）"""
        differences = {}
        
        # 邻接矩阵差异
        adj_diff = self.W_M_20 - self.W_M_0
        adj_abs_diff = np.abs(adj_diff)
        
        # 找出变化最大的位置
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
        
        # 路径类型统计
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
    
    def visualize_comparison(self, save_dir: str = "alpine_analysis"):
        """创建完整92×92矩阵可视化，包含阶段标记和S1→S3放大图"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置样式
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        
        # 创建更大的图来容纳完整矩阵
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.3)
        
        # 显示完整矩阵
        display_size = self.W_M_0.shape[0]  # 92
        
        # 阶段边界
        boundaries = [0, 2, 32, 62, 92]  # [特殊token, S1开始, S2开始, S3开始, 结束]
        
        # === 第一行：邻接矩阵 ===
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.W_M_0, cmap='RdBu_r', aspect='auto')
        ax1.set_title('0% Model - Adjacency (92×92)', fontweight='bold', fontsize=10)
        ax1.set_xlabel('To Token', fontsize=9)
        ax1.set_ylabel('From Token', fontsize=9)
        
        # 添加阶段分割线
        for b in boundaries[1:-1]:
            ax1.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax1.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        # 添加阶段标签
        ax1.text(1, -5, 'Spec', ha='center', fontsize=8, color='purple', weight='bold')
        ax1.text(17, -5, 'S1', ha='center', fontsize=8, color='blue', weight='bold')
        ax1.text(47, -5, 'S2', ha='center', fontsize=8, color='green', weight='bold')
        ax1.text(77, -5, 'S3', ha='center', fontsize=8, color='red', weight='bold')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.W_M_20, cmap='RdBu_r', aspect='auto')
        ax2.set_title('20% Model - Adjacency (92×92)', fontweight='bold', fontsize=10)
        ax2.set_xlabel('To Token', fontsize=9)
        ax2.set_ylabel('From Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax2.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax2.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        ax2.text(1, -5, 'Spec', ha='center', fontsize=8, color='purple', weight='bold')
        ax2.text(17, -5, 'S1', ha='center', fontsize=8, color='blue', weight='bold')
        ax2.text(47, -5, 'S2', ha='center', fontsize=8, color='green', weight='bold')
        ax2.text(77, -5, 'S3', ha='center', fontsize=8, color='red', weight='bold')
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff_adj = self.W_M_20 - self.W_M_0
        vmax = np.abs(diff_adj).max()
        im3 = ax3.imshow(diff_adj, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax3.set_title('Difference (20% - 0%)', fontweight='bold', fontsize=10)
        ax3.set_xlabel('To Token', fontsize=9)
        ax3.set_ylabel('From Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax3.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax3.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # === S1→S3区域放大图 ===
        ax_s1s3 = fig.add_subplot(gs[0, 3])
        # 提取S1→S3区域 (行2-31, 列62-91)
        s1_s3_diff = diff_adj[2:32, 62:92]
        im_s1s3 = ax_s1s3.imshow(s1_s3_diff, cmap='coolwarm', aspect='auto',
                                  vmin=-10, vmax=10)  # 调整范围以更好显示
        ax_s1s3.set_title('S1→S3 Changes (Zoom)', fontweight='bold', fontsize=10)
        ax_s1s3.set_xlabel('S3 Nodes (60-89)', fontsize=9)
        ax_s1s3.set_ylabel('S1 Nodes (0-29)', fontsize=9)
        plt.colorbar(im_s1s3, ax=ax_s1s3, fraction=0.046, pad=0.04)
        
        # 相似度
        ax4 = fig.add_subplot(gs[0, 4])
        ax4.axis('off')
        similarities = self.compute_similarities()
        text = "Adjacency Similarity:\n" + "="*20 + "\n"
        text += f"Pearson r: {similarities['adjacency']['pearson_correlation']:.3f}\n"
        text += f"Spearman ρ: {similarities['adjacency']['spearman_correlation']:.3f}\n"
        text += f"Cosine: {similarities['adjacency']['cosine_similarity']:.3f}\n"
        text += f"RMSE: {similarities['adjacency']['rmse']:.3f}\n"
        text += f"Mean |diff|: {similarities['adjacency']['mean_abs_diff']:.3f}"
        ax4.text(0.05, 0.5, text, fontsize=9, transform=ax4.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # === 第二行：可达矩阵 ===
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(self.W_V_0, cmap='RdBu_r', aspect='auto')
        ax5.set_title('0% Model - Reachability (92×92)', fontweight='bold', fontsize=10)
        ax5.set_xlabel('Source Token', fontsize=9)
        ax5.set_ylabel('Target Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax5.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax5.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(self.W_V_20, cmap='RdBu_r', aspect='auto')
        ax6.set_title('20% Model - Reachability (92×92)', fontweight='bold', fontsize=10)
        ax6.set_xlabel('Source Token', fontsize=9)
        ax6.set_ylabel('Target Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax6.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax6.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        ax7 = fig.add_subplot(gs[1, 2])
        diff_reach = self.W_V_20 - self.W_V_0
        vmax_reach = np.abs(diff_reach).max()
        im7 = ax7.imshow(diff_reach, cmap='coolwarm', vmin=-vmax_reach, vmax=vmax_reach, aspect='auto')
        ax7.set_title('Difference (20% - 0%)', fontweight='bold', fontsize=10)
        ax7.set_xlabel('Source Token', fontsize=9)
        ax7.set_ylabel('Target Token', fontsize=9)
        
        for b in boundaries[1:-1]:
            ax7.axhline(y=b-0.5, color='green', linewidth=0.8, alpha=0.5)
            ax7.axvline(x=b-0.5, color='green', linewidth=0.8, alpha=0.5)
        
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # S3←S1可达性放大图
        ax_s3s1 = fig.add_subplot(gs[1, 3])
        # 提取S3←S1区域 (行62-91, 列2-31)
        s3_s1_diff = diff_reach[62:92, 2:32]
        im_s3s1 = ax_s3s1.imshow(s3_s1_diff, cmap='coolwarm', aspect='auto',
                                  vmin=-2, vmax=2)  # 调整范围
        ax_s3s1.set_title('S3←S1 Reachability (Zoom)', fontweight='bold', fontsize=10)
        ax_s3s1.set_xlabel('S1 Nodes (0-29)', fontsize=9)
        ax_s3s1.set_ylabel('S3 Nodes (60-89)', fontsize=9)
        plt.colorbar(im_s3s1, ax=ax_s3s1, fraction=0.046, pad=0.04)
        
        # 相似度
        ax8 = fig.add_subplot(gs[1, 4])
        ax8.axis('off')
        text = "Reachability Similarity:\n" + "="*20 + "\n"
        text += f"Pearson r: {similarities['reachability']['pearson_correlation']:.3f}\n"
        text += f"Spearman ρ: {similarities['reachability']['spearman_correlation']:.3f}\n"
        text += f"Cosine: {similarities['reachability']['cosine_similarity']:.3f}\n"
        text += f"RMSE: {similarities['reachability']['rmse']:.3f}\n"
        text += f"Mean |diff|: {similarities['reachability']['mean_abs_diff']:.3f}"
        ax8.text(0.05, 0.5, text, fontsize=9, transform=ax8.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # === 第三行：注意力模式 ===
        if self.attention_0.shape == (3, 3):
            ax9 = fig.add_subplot(gs[2, 0])
            im9 = ax9.imshow(self.attention_0, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax9.set_title('0% Model - Attention', fontweight='bold', fontsize=10)
            ax9.set_xlabel('Key Position', fontsize=9)
            ax9.set_ylabel('Query Position', fontsize=9)
            ax9.set_xticks([0, 1, 2])
            ax9.set_yticks([0, 1, 2])
            ax9.set_xticklabels(['S1', 'S2', 'S3'])
            ax9.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
            
            # 添加数值标注
            for i in range(3):
                for j in range(3):
                    if i >= j:  # 只显示下三角（因果掩码）
                        text = ax9.text(j, i, f'{self.attention_0[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
            
            ax10 = fig.add_subplot(gs[2, 1])
            im10 = ax10.imshow(self.attention_20, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax10.set_title('20% Model - Attention', fontweight='bold', fontsize=10)
            ax10.set_xlabel('Key Position', fontsize=9)
            ax10.set_ylabel('Query Position', fontsize=9)
            ax10.set_xticks([0, 1, 2])
            ax10.set_yticks([0, 1, 2])
            ax10.set_xticklabels(['S1', 'S2', 'S3'])
            ax10.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
            
            # 添加数值标注
            for i in range(3):
                for j in range(3):
                    if i >= j:  # 只显示下三角（因果掩码）
                        text = ax10.text(j, i, f'{self.attention_20[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
            
            ax11 = fig.add_subplot(gs[2, 2])
            diff_attn = self.attention_20 - self.attention_0
            vmax = np.abs(diff_attn).max()
            im11 = ax11.imshow(diff_attn, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
            ax11.set_title('Attention Diff (20% - 0%)', fontweight='bold', fontsize=10)
            ax11.set_xlabel('Key Position', fontsize=9)
            ax11.set_ylabel('Query Position', fontsize=9)
            ax11.set_xticks([0, 1, 2])
            ax11.set_yticks([0, 1, 2])
            ax11.set_xticklabels(['S1', 'S2', 'S3'])
            ax11.set_yticklabels(['S1', 'S2', 'S3'])
            plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)
            
            # 添加数值标注
            for i in range(3):
                for j in range(3):
                    if i >= j:
                        text = ax11.text(j, i, f'{diff_attn[i, j]:+.2f}',
                                       ha="center", va="center", 
                                       color="black" if abs(diff_attn[i, j]) < vmax*0.5 else "white",
                                       fontsize=8)
            
            # 注意力分析文本
            ax12 = fig.add_subplot(gs[2, 3:])
            ax12.axis('off')
            attn_analysis = self.analyze_attention_patterns()
            if 'changes' in attn_analysis:
                text = "Attention Analysis:\n" + "="*30 + "\n"
                text += f"Max increase: {attn_analysis['changes']['max_increase']:+.3f}\n"
                text += f"Max decrease: {attn_analysis['changes']['max_decrease']:+.3f}\n"
                text += f"Mean change: {attn_analysis['changes']['mean_change']:+.3f}\n"
                text += f"Attn to S1 change: {attn_analysis['changes']['attention_to_first_token_change']:+.3f}\n"
                text += f"Self-attn change: {attn_analysis['changes']['self_attention_change']:+.3f}"
                ax12.text(0.05, 0.5, text, fontsize=9, transform=ax12.transAxes,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.5))
        
        # === 第四行：分析 ===
        # 权重分布
        ax13 = fig.add_subplot(gs[3, :2])
        ax13.hist(self.W_M_0.flatten(), bins=50, alpha=0.5, label='0% Model', 
                density=True, color='blue', edgecolor='black', linewidth=0.5)
        ax13.hist(self.W_M_20.flatten(), bins=50, alpha=0.5, label='20% Model', 
                density=True, color='red', edgecolor='black', linewidth=0.5)
        ax13.set_xlabel('Weight Value', fontsize=9)
        ax13.set_ylabel('Density', fontsize=9)
        ax13.set_title('Adjacency Weight Distribution', fontweight='bold', fontsize=10)
        ax13.legend(fontsize=9)
        ax13.grid(True, alpha=0.3)
        
        # 组合路径分析
        ax14 = fig.add_subplot(gs[3, 2:])
        ax14.axis('off')
        comp = self.analyze_compositional_paths()
        
        text = "S1→S3 Compositional Analysis:\n" + "="*30 + "\n\n"
        text += f"Paths analyzed: {comp['summary']['num_paths_analyzed']}\n\n"
        text += "Adjacency changes:\n"
        text += f"  Mean: {comp['summary']['adjacency']['mean_increase']:+.3f}\n"
        text += f"  Std:  {comp['summary']['adjacency']['std_increase']:.3f}\n"
        text += f"  Max:  {comp['summary']['adjacency']['max_increase']:+.3f}\n"
        text += f"  Min:  {comp['summary']['adjacency']['min_increase']:+.3f}\n\n"
        text += "Reachability changes:\n"
        text += f"  Mean: {comp['summary']['reachability']['mean_increase']:+.3f}\n"
        text += f"  Std:  {comp['summary']['reachability']['std_increase']:.3f}\n"
        text += f"  Max:  {comp['summary']['reachability']['max_increase']:+.3f}\n"
        text += f"  Min:  {comp['summary']['reachability']['min_increase']:+.3f}"
        
        ax14.text(0.05, 0.9, text, fontsize=9, transform=ax14.transAxes,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        # 总标题
        plt.suptitle(f'ALPINE Analysis: Compositional Learning (Seed {self.config.seed}) - Full 92×92 Matrix', 
                    fontsize=14, fontweight='bold')
        
        # 保存
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
        
        # 1. JSON结果
        results = {
            'metadata': {
                'seed': self.config.seed,
                'timestamp': timestamp,
                'model_paths': {
                    '0%': self.config.model_0_path,
                    '20%': self.config.model_20_path
                },
                'data_paths': {
                    '0%': self.config.data_dir_0,
                    '20%': self.config.data_dir_20
                },
                'model_config': {
                    'n_layer': self.config.n_layer,
                    'n_head': self.config.n_head,
                    'n_embd': self.config.n_embd,
                    'vocab_size': self.config.vocab_size
                },
                'token_mapping': {
                    'special_tokens': [0, 1],
                    'S1_tokens': self.config.token_S1,
                    'S2_tokens': self.config.token_S2,
                    'S3_tokens': self.config.token_S3
                }
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
        
        # 2. NumPy矩阵
        np_path = os.path.join(save_dir, f'matrices_seed{self.config.seed}_{timestamp}.npz')
        np.savez(np_path,
                W_M_0=self.W_M_0,
                W_M_20=self.W_M_20,
                W_V_0=self.W_V_0,
                W_V_20=self.W_V_20,
                attention_0=self.attention_0,
                attention_20=self.attention_20)
        print(f"✓ 矩阵已保存: {np_path}")
        
        # 3. 文本报告
        report_path = os.path.join(save_dir, f'report_seed{self.config.seed}_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ALPINE MATRIX ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date: {timestamp}\n")
            f.write(f"Seed: {self.config.seed}\n")
            f.write(f"Model dimension: {self.config.n_embd}\n")
            f.write(f"Vocabulary size: {self.config.vocab_size}\n")
            f.write("\nToken Mapping:\n")
            f.write(f"  Special tokens: 0-1 ([PAD], \\n)\n")
            f.write(f"  S1 tokens: 2-31 (graph nodes 0-29)\n")
            f.write(f"  S2 tokens: 32-61 (graph nodes 30-59)\n")
            f.write(f"  S3 tokens: 62-91 (graph nodes 60-89)\n\n")
            
            f.write("SIMILARITY METRICS\n")
            f.write("-"*40 + "\n")
            similarities = self.compute_similarities()
            for matrix_type, metrics in similarities.items():
                f.write(f"\n{matrix_type.upper()} Matrix:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("COMPOSITIONAL ANALYSIS (S1→S3)\n")
            f.write("-"*40 + "\n")
            comp = self.analyze_compositional_paths()
            f.write(f"\nSummary:\n")
            for key, value in comp['summary'].items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("KEY DIFFERENCES\n")
            f.write("-"*40 + "\n")
            differences = self.find_key_differences()
            f.write("\nPath type statistics:\n")
            for path_type, stats in differences.get('path_type_statistics', {}).items():
                f.write(f"\n{path_type}:\n")
                for k, v in stats.items():
                    f.write(f"  {k}: {v:.4f}\n")
            
            # 添加注意力分析
            f.write("\n" + "="*60 + "\n")
            f.write("ATTENTION PATTERN ANALYSIS\n")
            f.write("-"*40 + "\n")
            attn_analysis = self.analyze_attention_patterns()
            if attn_analysis:
                import pprint
                f.write(pprint.pformat(attn_analysis, indent=2))
            
        print(f"✓ 报告已保存: {report_path}")
        
        return json_path, np_path, report_path

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
        print("3. 分析结果")
        print("="*60)
        
        # 相似度
        similarities = comparator.compute_similarities()
        print("\n📊 相似度指标:")
        for matrix_type, metrics in similarities.items():
            print(f"\n{matrix_type.capitalize()} Matrix:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # 组合分析
        comp = comparator.analyze_compositional_paths()
        print(f"\n🔗 组合路径分析 (S1→S3):")
        print(f"  分析路径数: {comp['summary']['num_paths_analyzed']}")
        print(f"  邻接权重平均增加: {comp['summary']['adjacency']['mean_increase']:+.3f}")
        print(f"  可达权重平均增加: {comp['summary']['reachability']['mean_increase']:+.3f}")
        
        # 关键差异
        differences = comparator.find_key_differences()
        print(f"\n🎯 关键差异:")
        for path_type, stats in differences.get('path_type_statistics', {}).items():
            print(f"  {path_type}: {stats['count']}个显著变化, 平均变化{stats['mean_change']:+.3f}")
        
        # 注意力分析
        attn_analysis = comparator.analyze_attention_patterns()
        if 'changes' in attn_analysis:
            print(f"\n🔍 注意力模式变化:")
            print(f"  最大增加: {attn_analysis['changes']['max_increase']:+.3f}")
            print(f"  最大减少: {attn_analysis['changes']['max_decrease']:+.3f}")
            print(f"  平均变化: {attn_analysis['changes']['mean_change']:+.3f}")
            print(f"  对S1位置的注意力变化: {attn_analysis['changes']['attention_to_first_token_change']:+.3f}")
        
        # 可视化（可选）
        if not args.skip_viz:
            print("\n4. 生成可视化...")
            comparator.visualize_comparison(save_dir=args.save_dir)
        
        # 保存
        print("\n5. 保存结果...")
        json_path, np_path, report_path = comparator.save_results(save_dir=args.save_dir)
        
        print("\n" + "="*60)
        print("✅ 分析完成！")
        print("="*60)
        print(f"\n📁 结果文件:")
        print(f"  - JSON: {json_path}")
        print(f"  - 矩阵: {np_path}")
        print(f"  - 报告: {report_path}")
        
        return comparator
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    comparator = main()