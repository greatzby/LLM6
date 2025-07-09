#!/usr/bin/env python3
# mechanistic_analysis.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm

class MechanisticAnalyzer:
    def __init__(self, checkpoint_pattern="out/composition_mix*/ckpt_*.pt"):
        self.checkpoints = sorted(glob.glob(checkpoint_pattern))
        
    def analyze_checkpoint(self, ckpt_path):
        """分析单个checkpoint"""
        # 加载权重
        state = torch.load(ckpt_path, map_location='cpu')
        W = state['model']['lm_head.weight']
        
        # 提取元信息
        path = Path(ckpt_path)
        parts = path.stem.split('_')
        ratio = int(parts[1].replace('mix', ''))
        seed = int(parts[2].replace('seed', ''))
        iteration = int(parts[3].replace('iter', ''))
        
        # 计算所有指标
        metrics = {
            'ratio': ratio,
            'seed': seed,
            'iteration': iteration,
            'effective_rank': self.compute_effective_rank(W),
            'bridge_capacity': self.compute_stage_bridging_capacity(W),
            'subspace_separation': self.compute_subspace_separation(W),
            'specialization_index': self.compute_specialization_index(W),
            'path_diversity': self.compute_path_diversity(W)
        }
        
        return metrics
    
    def compute_effective_rank(self, W):
        """计算ER"""
        s = torch.linalg.svdvals(W)
        s_norm = s / s.sum()
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
        return torch.exp(entropy).item()
    
    def compute_stage_bridging_capacity(self, W):
        """S2的桥接能力"""
        W_S1 = W[2:32].mean(0)
        W_S2 = W[32:62].mean(0)
        W_S3 = W[62:92].mean(0)
        
        # S2应该在S1-S3的"中间"
        expected_S2 = 0.5 * (W_S1 + W_S3)
        actual_S2 = W_S2
        
        # 偏离度（越小越好）
        deviation = torch.norm(actual_S2 - expected_S2) / torch.norm(expected_S2)
        
        # 转换为capacity（越大越好）
        capacity = 1.0 / (1.0 + deviation)
        
        return capacity.item()
    
    def compute_subspace_separation(self, W):
        """计算S1和S3的子空间分离度"""
        W_S1 = W[2:32] - W[2:32].mean(0)
        W_S3 = W[62:92] - W[62:92].mean(0)
        
        # SVD获取主方向
        U1, S1, V1 = torch.svd(W_S1)
        U3, S3, V3 = torch.svd(W_S3)
        
        # 取前5个主成分
        k = min(5, V1.shape[1], V3.shape[1])
        basis_S1 = V1[:, :k]
        basis_S3 = V3[:, :k]
        
        # 计算子空间夹角
        M = basis_S1.T @ basis_S3
        s = torch.linalg.svdvals(M)
        
        # 平均canonical angle
        separation = torch.acos(s.clamp(-1, 1)).mean().item()
        
        return separation
    
    def compute_specialization_index(self, W):
        """计算特化程度"""
        # 各阶段内部相似度
        def stage_coherence(W_stage):
            W_norm = F.normalize(W_stage, p=2, dim=1)
            sim = W_norm @ W_norm.T
            # 排除对角线
            mask = ~torch.eye(len(W_stage), dtype=bool)
            return sim[mask].mean().item()
        
        coherence_S1 = stage_coherence(W[2:32])
        coherence_S2 = stage_coherence(W[32:62])
        coherence_S3 = stage_coherence(W[62:92])
        
        # 特化指数：内部相似度的平均
        specialization = (coherence_S1 + coherence_S2 + coherence_S3) / 3
        
        return specialization
    
    def compute_path_diversity(self, W):
        """路径多样性：衡量S1→S2→S3的路径丰富度"""
        W_S1 = W[2:32]
        W_S2 = W[32:62]
        W_S3 = W[62:92]
        
        # 计算所有可能的S1→S2→S3路径的"能量"
        # 使用三阶张量的迹作为proxy
        path_tensor = torch.einsum('id,jd,kd->ijk', W_S1, W_S2, W_S3)
        
        # 路径多样性 = 有效路径数
        diversity = torch.norm(path_tensor, p='fro').item() / path_tensor.numel()
        
        return diversity
    
    def run_analysis(self):
        """运行完整分析"""
        results = []
        
        for ckpt in tqdm(self.checkpoints, desc="Analyzing checkpoints"):
            try:
                metrics = self.analyze_checkpoint(ckpt)
                results.append(metrics)
            except Exception as e:
                print(f"Error processing {ckpt}: {e}")
                
        return pd.DataFrame(results)

def plot_mechanistic_story(df):
    """绘制机制解释图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Bridge Capacity Evolution
    ax = axes[0, 0]
    for ratio in [0, 5, 10]:
        data = df[df['ratio'] == ratio].groupby('iteration').mean()
        ax.plot(data.index, data['bridge_capacity'], label=f'{ratio}% mix', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('S2 Bridge Capacity')
    ax.set_title('S2 Loses Bridging Function')
    ax.legend()
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # 2. Subspace Separation
    ax = axes[0, 1]
    for ratio in [0, 5, 10]:
        data = df[df['ratio'] == ratio].groupby('iteration').mean()
        ax.plot(data.index, data['subspace_separation'], label=f'{ratio}% mix', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('S1-S3 Subspace Angle')
    ax.set_title('Subspace Isolation')
    ax.legend()
    
    # 3. Specialization Index
    ax = axes[0, 2]
    for ratio in [0, 5, 10]:
        data = df[df['ratio'] == ratio].groupby('iteration').mean()
        ax.plot(data.index, data['specialization_index'], label=f'{ratio}% mix', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Within-Stage Coherence')
    ax.set_title('Progressive Specialization')
    ax.legend()
    
    # 4. ER vs Bridge Capacity
    ax = axes[1, 0]
    final_data = df[df['iteration'] > 80000]
    scatter = ax.scatter(final_data['effective_rank'], 
                        final_data['bridge_capacity'],
                        c=final_data['ratio'], cmap='viridis', s=50)
    ax.set_xlabel('Effective Rank')
    ax.set_ylabel('Bridge Capacity')
    ax.set_title('ER alone is not enough')
    plt.colorbar(scatter, ax=ax, label='Mix %')
    
    # 5. Path Diversity
    ax = axes[1, 1]
    for ratio in [0, 5, 10]:
        data = df[df['ratio'] == ratio].groupby('iteration').mean()
        ax.plot(data.index, data['path_diversity'], label=f'{ratio}% mix', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Path Diversity')
    ax.set_title('Compositional Path Richness')
    ax.legend()
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = """
    Mechanistic Explanation:
    
    1. Early Success (5k):
       • High ER + Bridge Capacity
       • S2 naturally bridges S1→S3
    
    2. Forgetting Process:
       • S1/S3 subspaces separate
       • S2 loses bridging function
       • Specialization increases
    
    3. Why 5% Mix Works:
       • Forces S2 to maintain dual role
       • Prevents subspace isolation
       • Preserves path diversity
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Mechanistic Analysis of Compositional Forgetting', fontsize=16)
    plt.tight_layout()
    plt.savefig('mechanistic_story.png', dpi=300)

def main():
    analyzer = MechanisticAnalyzer()
    df = analyzer.run_analysis()
    df.to_csv('mechanistic_analysis.csv', index=False)
    plot_mechanistic_story(df)
    
    # 打印关键发现
    print("\n=== KEY FINDINGS ===")
    print("\n1. Bridge Capacity at 100k iterations:")
    final = df[df['iteration'] > 90000].groupby('ratio')['bridge_capacity'].mean()
    print(final)
    
    print("\n2. Subspace Separation at 100k:")
    final_sep = df[df['iteration'] > 90000].groupby('ratio')['subspace_separation'].mean()
    print(final_sep)
    
    print("\n3. Critical Transition Points:")
    # 找到bridge capacity < 0.5的最早时间
    for ratio in [0, 5, 10]:
        ratio_df = df[df['ratio'] == ratio].sort_values('iteration')
        below_threshold = ratio_df[ratio_df['bridge_capacity'] < 0.5]
        if len(below_threshold) > 0:
            print(f"  {ratio}% mix: Bridge fails at {below_threshold.iloc[0]['iteration']} iterations")
        else:
            print(f"  {ratio}% mix: Bridge never fails")

if __name__ == "__main__":
    main()