#!/usr/bin/env python3
"""
MLP层演化分析（92维健康模型对照组）
对比20k vs 50k checkpoint
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd, orthogonal_procrustes
from datetime import datetime
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class MLPEvolutionAnalyzer:
    """MLP层演化分析器"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_mlp_weights(self, checkpoint_path: str) -> Dict[str, np.ndarray]:
        """加载MLP层权重"""
        print(f"\n加载: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        mlp_weights = {}
        
        # 提取MLP层权重
        fc_key = 'transformer.h.0.mlp.c_fc.weight'
        proj_key = 'transformer.h.0.mlp.c_proj.weight'
        
        if fc_key in state_dict:
            mlp_weights['c_fc'] = state_dict[fc_key].cpu().numpy()
            print(f"  c_fc shape: {mlp_weights['c_fc'].shape}")
            
        if proj_key in state_dict:
            mlp_weights['c_proj'] = state_dict[proj_key].cpu().numpy()
            print(f"  c_proj shape: {mlp_weights['c_proj'].shape}")
            
        return mlp_weights
    
    def procrustes_align(self, W_source: np.ndarray, W_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Procrustes对齐"""
        R, scale = orthogonal_procrustes(W_target.T, W_source.T)
        W_target_aligned = (W_target.T @ R).T
        return W_target_aligned, R
    
    def compute_similarity(self, W1: np.ndarray, W2: np.ndarray) -> Dict[str, float]:
        """计算两个矩阵的相似度指标"""
        w1_flat = W1.flatten()
        w2_flat = W2.flatten()
        cosine_sim = np.dot(w1_flat, w2_flat) / (np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat))
        pearson_corr = np.corrcoef(w1_flat, w2_flat)[0, 1]
        frob_diff = np.linalg.norm(W1 - W2, 'fro')
        return {'cosine': cosine_sim, 'pearson': pearson_corr, 'frobenius_diff': frob_diff}
    
    def analyze_layer(self, W_20k: np.ndarray, W_50k: np.ndarray, layer_name: str) -> Dict:
        """分析单个层的演化"""
        print(f"\n{'='*60}\n分析 {layer_name} 层\n{'='*60}")
        print(f"形状: {W_20k.shape}")
        
        sim_before = self.compute_similarity(W_20k, W_50k)
        print(f"\n对齐前:\n  Cosine: {sim_before['cosine']:.4f}, Pearson: {sim_before['pearson']:.4f}")
        
        W_50k_aligned, rotation_matrix = self.procrustes_align(W_20k, W_50k)
        
        sim_after = self.compute_similarity(W_20k, W_50k_aligned)
        print(f"\n对齐后:\n  Cosine: {sim_after['cosine']:.4f}, Pearson: {sim_after['pearson']:.4f}")
        print(f"  改善: {sim_after['cosine'] - sim_before['cosine']:.4f}")
        
        U_20k, s_20k, Vt_20k = svd(W_20k, full_matrices=False)
        U_50k, s_50k, Vt_50k = svd(W_50k, full_matrices=False)
        U_50k_aligned, s_50k_aligned, Vt_50k_aligned = svd(W_50k_aligned, full_matrices=False)
        
        print(f"\nSVD分解:\n  奇异值数量: {len(s_20k)}")
        
        n_dims = len(s_20k)
        per_dim_analysis = []
        for i in range(n_dims):
            sv_20k_val = s_20k[i]
            sv_50k_aligned_val = s_50k_aligned[i]
            rel_change = ((sv_50k_aligned_val - sv_20k_val) / sv_20k_val * 100) if sv_20k_val > 1e-10 else 0
            
            if abs(rel_change) < 5: status = 'stable'
            elif rel_change < -50: status = 'collapsed'
            elif rel_change > 100: status = 'exploded'
            else: status = 'changed'
            
            per_dim_analysis.append({
                'dimension': i + 1, 'sv_20k': sv_20k_val, 'sv_50k_aligned': sv_50k_aligned_val,
                'relative_change_%': rel_change, 'status': status
            })
        df = pd.DataFrame(per_dim_analysis)
        
        print("\n维度状态统计:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items(): print(f"  {status}: {count}")
        
        cumsum_20k = np.cumsum(s_20k ** 2) / np.sum(s_20k ** 2)
        cumsum_50k = np.cumsum(s_50k_aligned ** 2) / np.sum(s_50k_aligned ** 2)
        rank_90_20k = np.argmax(cumsum_20k >= 0.9) + 1
        rank_90_50k = np.argmax(cumsum_50k >= 0.9) + 1
        
        print(f"\n有效秩分析 (90%能量):\n  20k: {rank_90_20k}, 50k: {rank_90_50k}, 变化: {rank_90_50k - rank_90_20k}")
        
        return {
            'layer_name': layer_name, 'shape': W_20k.shape,
            'similarity_before': sim_before, 'similarity_after': sim_after,
            'rotation_matrix': rotation_matrix,
            'singular_values': {'sv_20k': s_20k, 'sv_50k_aligned': s_50k_aligned},
            'per_dimension_analysis': df, 'status_counts': status_counts.to_dict(),
            'effective_ranks': {'90%': {'20k': rank_90_20k, '50k': rank_90_50k}},
            'U_matrices': {'U_20k': U_20k, 'U_50k_aligned': U_50k_aligned}
        }
    
    def create_visualization(self, results: Dict, performance_str: str) -> str:
        """创建综合可视化"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        df = results['per_dimension_analysis']
        sv_20k = results['singular_values']['sv_20k']
        sv_50k_aligned = results['singular_values']['sv_50k_aligned']
        
        # 1. 奇异值对比
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(df['dimension'], df['sv_20k'], 'b-', label='20k', linewidth=2)
        ax1.semilogy(df['dimension'], df['sv_50k_aligned'], 'r-', label='50k aligned', linewidth=2, alpha=0.7)
        ax1.set_title(f"{results['layer_name']}: Singular Values"); ax1.set_xlabel('Dimension'); ax1.set_ylabel('Singular Value (log)'); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # 2. 相对变化条形图
        ax2 = fig.add_subplot(gs[0, 1:3])
        colors = df['status'].map({'collapsed': 'red', 'exploded': 'orange', 'changed': 'blue', 'stable': 'green'})
        ax2.bar(df['dimension'], df['relative_change_%'], color=colors, alpha=0.7)
        ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Collapse thr.'); ax2.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Explosion thr.')
        ax2.set_title('Per-Dimension Change (20k → 50k)'); ax2.set_xlabel('Dimension'); ax2.set_ylabel('Relative Change (%)'); ax2.legend(); ax2.grid(True, alpha=0.3)
        
        # 3. 能量集中度
        ax3 = fig.add_subplot(gs[0, 3])
        cumsum_20k = np.cumsum(sv_20k ** 2) / np.sum(sv_20k ** 2) * 100
        cumsum_50k = np.cumsum(sv_50k_aligned ** 2) / np.sum(sv_50k_aligned ** 2) * 100
        ax3.plot(df['dimension'], cumsum_20k, 'b-', label='20k', linewidth=2)
        ax3.plot(df['dimension'], cumsum_50k, 'r-', label='50k', linewidth=2)
        ax3.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('Energy Concentration'); ax3.set_xlabel('Number of Dimensions'); ax3.set_ylabel('Cumulative Energy (%)'); ax3.legend(); ax3.grid(True, alpha=0.3)
        
        # 4. 奇异值散点相关图
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(sv_20k, sv_50k_aligned, alpha=0.6, s=30)
        ax4.plot([0, max(sv_20k)], [0, max(sv_20k)], 'r--', alpha=0.5, label='y=x')
        ax4.set_title('Singular Value Correlation'); ax4.set_xlabel('SV at 20k'); ax4.set_ylabel('SV at 50k (aligned)'); ax4.legend(); ax4.grid(True, alpha=0.3)
        
        # 5. 维度状态分布
        ax5 = fig.add_subplot(gs[1, 1])
        status_counts = df['status'].value_counts()
        pie_colors = status_counts.index.map({'stable': 'green', 'changed': 'blue', 'collapsed': 'red', 'exploded': 'orange'}).fillna('gray')
        ax5.pie(status_counts.values, labels=status_counts.index, colors=pie_colors, autopct='%1.0f%%', startangle=90)
        ax5.set_title('Dimension Status Distribution')
        
        # 6. 对齐质量
        ax6 = fig.add_subplot(gs[1, 2])
        metrics = ['Cosine', 'Pearson']; before = [results['similarity_before'][m.lower()] for m in metrics]; after = [results['similarity_after'][m.lower()] for m in metrics]
        x = np.arange(len(metrics)); width = 0.35
        ax6.bar(x - width/2, before, width, label='Before', color='coral'); ax6.bar(x + width/2, after, width, label='After Alignment', color='skyblue')
        ax6.set_title('Alignment Quality'); ax6.set_ylabel('Similarity'); ax6.set_xticks(x); ax6.set_xticklabels(metrics); ax6.legend(); ax6.set_ylim([0, 1])
        for i, (b, a) in enumerate(zip(before, after)): ax6.text(i, max(b, a) + 0.02, f'+{a-b:.3f}', ha='center', fontsize=9)
        
        # 7. 前30个维度的详细变化
        ax7 = fig.add_subplot(gs[1, 3])
        n_show = min(30, len(df))
        ax7.plot(df['dimension'][:n_show], df['sv_20k'][:n_show], 'b-o', label='20k', markersize=4)
        ax7.plot(df['dimension'][:n_show], df['sv_50k_aligned'][:n_show], 'r-s', label='50k aligned', markersize=4, alpha=0.7)
        ax7.set_title('Top 30 Dimensions Detail'); ax7.set_xlabel('Dimension'); ax7.set_ylabel('Singular Value'); ax7.legend(); ax7.grid(True, alpha=0.3)
        
        # 8. 变化率分布直方图
        ax8 = fig.add_subplot(gs[2, 0])
        ax8.hist(df['relative_change_%'], bins=30, edgecolor='black', alpha=0.7)
        ax8.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax8.set_title('Distribution of Changes'); ax8.set_xlabel('Relative Change (%)'); ax8.set_ylabel('Count'); ax8.grid(True, alpha=0.3)
        
        # 9. 权重差异模式 (左奇异向量)
        ax9 = fig.add_subplot(gs[2, 1:3])
        n_components = min(20, results['U_matrices']['U_20k'].shape[1])
        diff_matrix = results['U_matrices']['U_50k_aligned'][:, :n_components] - results['U_matrices']['U_20k'][:, :n_components]
        im = ax9.imshow(diff_matrix, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
        ax9.set_title('Difference in Left Singular Vectors (Top 20)'); ax9.set_xlabel('Component'); ax9.set_ylabel('Dimension'); plt.colorbar(im, ax=ax9)
        
        # 10. 摘要文本
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis('off')
        summary_text = f"""ANALYSIS SUMMARY
{'='*25}
Layer: {results['layer_name']}
Shape: {results['shape']}
Alignment Quality:
• Cosine: {results['similarity_before']['cosine']:.4f} → {results['similarity_after']['cosine']:.4f}
• Improv: +{results['similarity_after']['cosine'] - results['similarity_before']['cosine']:.4f}
Dimension Status:
• Stable: {results['status_counts'].get('stable', 0)}
• Changed: {results['status_counts'].get('changed', 0)}
• Collapsed: {results['status_counts'].get('collapsed', 0)}
• Exploded: {results['status_counts'].get('exploded', 0)}
Effective Rank (90%):
• 20k: {results['effective_ranks']['90%']['20k']}
• 50k: {results['effective_ranks']['90%']['50k']}
• Change: {results['effective_ranks']['90%']['50k'] - results['effective_ranks']['90%']['20k']}"""
        ax10.text(0.0, 0.5, summary_text, fontsize=10, family='monospace', va='center')
        
        fig.suptitle(f'MLP {results["layer_name"]} Evolution: 20k vs 50k (Procrustes Aligned)\n' + 
                     f'{performance_str} | Similarity: {results["similarity_before"]["cosine"]:.3f} → {results["similarity_after"]["cosine"]:.3f}',
                     fontsize=14, fontweight='bold')
        
        save_dir = "checkpoint_evolution"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'mlp_{results["layer_name"]}_analysis_92d_{self.timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ 图像已保存: {save_path}")
        return save_path

def main():
    """主函数"""
    print("="*80)
    print("🔬 MLP Layer Evolution Analysis (92-dim HEALTHY CONTROL)")
    print("="*80)
    
    # ### 修改点 1: 更新为92维模型的路径 ###
    checkpoint_dir = "out_d92/composition_mix0_seed42_20250801_054758"
    ckpt_20k_path = os.path.join(checkpoint_dir, "ckpt_mix0_seed42_iter20000.pt")
    ckpt_50k_path = os.path.join(checkpoint_dir, "ckpt_mix0_seed42_iter50000.pt")
    
    analyzer = MLPEvolutionAnalyzer(checkpoint_dir)
    
    weights_20k = analyzer.load_mlp_weights(ckpt_20k_path)
    weights_50k = analyzer.load_mlp_weights(ckpt_50k_path)
    
    all_results = {}
    
    if 'c_proj' in weights_20k:
        print("\n" + "="*80)
        print("📊 分析 c_proj 层 (Hidden → Output)")
        results_proj = analyzer.analyze_layer(
            weights_20k['c_proj'],
            weights_50k['c_proj'],
            'c_proj'
        )
        all_results['c_proj'] = results_proj
    
    print("\n" + "="*80)
    print("📊 创建可视化...")
    for layer_name, results in all_results.items():
        # ### 修改点 2: 传入能反映92维模型情况的标题字符串 ###
        performance_title = "Performance: Stable (Control Group)"
        analyzer.create_visualization(results, performance_title)
        # analyzer.save_results(results, layer_name) # 如果需要保存详细数据，取消此行注释
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()