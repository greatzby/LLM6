#!/usr/bin/env python3
"""
MLP层演化分析：20k vs 50k checkpoint对比
特别关注c_proj层（问题最严重的层）
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
        """
        Procrustes对齐
        返回：对齐后的target矩阵，旋转矩阵
        """
        # 计算最优旋转矩阵
        R, scale = orthogonal_procrustes(W_target.T, W_source.T)
        W_target_aligned = (W_target.T @ R).T
        
        return W_target_aligned, R
    
    def compute_similarity(self, W1: np.ndarray, W2: np.ndarray) -> Dict[str, float]:
        """计算两个矩阵的相似度指标"""
        # Flatten
        w1_flat = W1.flatten()
        w2_flat = W2.flatten()
        
        # Cosine similarity
        cosine_sim = np.dot(w1_flat, w2_flat) / (np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat))
        
        # Pearson correlation
        pearson_corr = np.corrcoef(w1_flat, w2_flat)[0, 1]
        
        # Frobenius norm of difference
        frob_diff = np.linalg.norm(W1 - W2, 'fro')
        
        return {
            'cosine': cosine_sim,
            'pearson': pearson_corr,
            'frobenius_diff': frob_diff
        }
    
    def analyze_layer(self, W_20k: np.ndarray, W_50k: np.ndarray, 
                     layer_name: str) -> Dict:
        """分析单个层的演化"""
        print(f"\n{'='*60}")
        print(f"分析 {layer_name} 层")
        print(f"{'='*60}")
        print(f"形状: {W_20k.shape}")
        
        # 1. 对齐前的相似度
        sim_before = self.compute_similarity(W_20k, W_50k)
        print(f"\n对齐前:")
        print(f"  Cosine: {sim_before['cosine']:.4f}")
        print(f"  Pearson: {sim_before['pearson']:.4f}")
        
        # 2. Procrustes对齐
        W_50k_aligned, rotation_matrix = self.procrustes_align(W_20k, W_50k)
        
        # 3. 对齐后的相似度
        sim_after = self.compute_similarity(W_20k, W_50k_aligned)
        print(f"\n对齐后:")
        print(f"  Cosine: {sim_after['cosine']:.4f}")
        print(f"  Pearson: {sim_after['pearson']:.4f}")
        print(f"  改善: {sim_after['cosine'] - sim_before['cosine']:.4f}")
        
        # 4. SVD分解
        U_20k, s_20k, Vt_20k = svd(W_20k, full_matrices=False)
        U_50k, s_50k, Vt_50k = svd(W_50k, full_matrices=False)
        U_50k_aligned, s_50k_aligned, Vt_50k_aligned = svd(W_50k_aligned, full_matrices=False)
        
        print(f"\nSVD分解:")
        print(f"  奇异值数量: {len(s_20k)}")
        
        # 5. 创建逐维度分析DataFrame
        n_dims = len(s_20k)
        per_dim_analysis = []
        
        for i in range(n_dims):
            sv_20k = s_20k[i]
            sv_50k_orig = s_50k[i]
            sv_50k_aligned = s_50k_aligned[i]
            
            abs_change = sv_50k_aligned - sv_20k
            rel_change = (abs_change / sv_20k * 100) if sv_20k > 1e-10 else 0
            
            # 能量百分比
            energy_20k = (sv_20k ** 2) / np.sum(s_20k ** 2) * 100
            energy_50k_aligned = (sv_50k_aligned ** 2) / np.sum(s_50k_aligned ** 2) * 100
            
            # 状态分类
            if abs(rel_change) < 5:
                status = 'stable'
            elif rel_change < -50:
                status = 'collapsed'
            elif rel_change > 100:
                status = 'exploded'
            else:
                status = 'changed'
            
            per_dim_analysis.append({
                'dimension': i + 1,
                'sv_20k': sv_20k,
                'sv_50k_original': sv_50k_orig,
                'sv_50k_aligned': sv_50k_aligned,
                'absolute_change': abs_change,
                'relative_change_%': rel_change,
                'energy_20k_%': energy_20k,
                'energy_50k_aligned_%': energy_50k_aligned,
                'status': status
            })
        
        df = pd.DataFrame(per_dim_analysis)
        
        # 6. 统计分析
        print(f"\n维度状态统计:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # 7. 有效秩分析
        cumsum_20k = np.cumsum(s_20k ** 2) / np.sum(s_20k ** 2)
        cumsum_50k = np.cumsum(s_50k_aligned ** 2) / np.sum(s_50k_aligned ** 2)
        
        rank_90_20k = np.argmax(cumsum_20k >= 0.9) + 1
        rank_90_50k = np.argmax(cumsum_50k >= 0.9) + 1
        rank_99_20k = np.argmax(cumsum_20k >= 0.99) + 1
        rank_99_50k = np.argmax(cumsum_50k >= 0.99) + 1
        
        print(f"\n有效秩分析:")
        print(f"  90%能量 - 20k: {rank_90_20k}, 50k: {rank_90_50k}, 变化: {rank_90_50k - rank_90_20k}")
        print(f"  99%能量 - 20k: {rank_99_20k}, 50k: {rank_99_50k}, 变化: {rank_99_50k - rank_99_20k}")
        
        return {
            'layer_name': layer_name,
            'shape': W_20k.shape,
            'similarity_before': sim_before,
            'similarity_after': sim_after,
            'rotation_matrix': rotation_matrix,
            'singular_values': {
                'sv_20k': s_20k,
                'sv_50k': s_50k,
                'sv_50k_aligned': s_50k_aligned
            },
            'per_dimension_analysis': df,
            'status_counts': status_counts.to_dict(),
            'effective_ranks': {
                '90%': {'20k': rank_90_20k, '50k': rank_90_50k},
                '99%': {'20k': rank_99_20k, '50k': rank_99_50k}
            }
        }
    
    def create_visualization(self, results: Dict) -> str:
        """创建综合可视化"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        df = results['per_dimension_analysis']
        sv_20k = results['singular_values']['sv_20k']
        sv_50k = results['singular_values']['sv_50k']
        sv_50k_aligned = results['singular_values']['sv_50k_aligned']
        
        # 1. 奇异值对比（对数尺度）
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(range(1, len(sv_20k)+1), sv_20k, 'b-', label='20k', linewidth=2)
        ax1.semilogy(range(1, len(sv_50k_aligned)+1), sv_50k_aligned, 'r-', 
                    label='50k aligned', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Singular Value (log)')
        ax1.set_title(f"{results['layer_name']}: Singular Values")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 相对变化条形图
        ax2 = fig.add_subplot(gs[0, 1:3])
        colors = ['red' if x < -50 else 'orange' if x > 100 else 'blue' 
                 for x in df['relative_change_%']]
        ax2.bar(df['dimension'], df['relative_change_%'], color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
        ax2.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Explosion threshold')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Relative Change (%)')
        ax2.set_title('Per-Dimension Change (20k → 50k)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 能量集中度
        ax3 = fig.add_subplot(gs[0, 3])
        cumsum_20k = np.cumsum(sv_20k ** 2) / np.sum(sv_20k ** 2) * 100
        cumsum_50k = np.cumsum(sv_50k_aligned ** 2) / np.sum(sv_50k_aligned ** 2) * 100
        ax3.plot(range(1, len(cumsum_20k)+1), cumsum_20k, 'b-', label='20k', linewidth=2)
        ax3.plot(range(1, len(cumsum_50k)+1), cumsum_50k, 'r-', label='50k', linewidth=2)
        ax3.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=99, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Number of Dimensions')
        ax3.set_ylabel('Cumulative Energy (%)')
        ax3.set_title('Energy Concentration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 奇异值散点相关图
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(sv_20k, sv_50k_aligned, alpha=0.6, s=30)
        ax4.plot([0, max(sv_20k)], [0, max(sv_20k)], 'r--', alpha=0.5, label='y=x')
        ax4.set_xlabel('SV at 20k')
        ax4.set_ylabel('SV at 50k (aligned)')
        ax4.set_title('Singular Value Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 维度状态分布
        ax5 = fig.add_subplot(gs[1, 1])
        status_counts = df['status'].value_counts()
        colors_pie = {'stable': 'green', 'changed': 'blue', 
                     'collapsed': 'red', 'exploded': 'orange'}
        pie_colors = [colors_pie.get(s, 'gray') for s in status_counts.index]
        ax5.pie(status_counts.values, labels=status_counts.index, colors=pie_colors,
                autopct='%1.0f%%', startangle=90)
        ax5.set_title('Dimension Status Distribution')
        
        # 6. 对齐质量
        ax6 = fig.add_subplot(gs[1, 2])
        metrics = ['Cosine', 'Pearson']
        before = [results['similarity_before']['cosine'], 
                 results['similarity_before']['pearson']]
        after = [results['similarity_after']['cosine'], 
                results['similarity_after']['pearson']]
        x = np.arange(len(metrics))
        width = 0.35
        ax6.bar(x - width/2, before, width, label='Before', color='coral')
        ax6.bar(x + width/2, after, width, label='After Alignment', color='skyblue')
        ax6.set_ylabel('Similarity')
        ax6.set_title('Alignment Quality')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics)
        ax6.legend()
        ax6.set_ylim([0, 1])
        for i, (b, a) in enumerate(zip(before, after)):
            ax6.text(i, max(b, a) + 0.02, f'+{a-b:.3f}', ha='center', fontsize=9)
        
        # 7. 前30个维度的详细变化
        ax7 = fig.add_subplot(gs[1, 3])
        n_show = min(30, len(df))
        ax7.plot(df['dimension'][:n_show], df['sv_20k'][:n_show], 'b-o', 
                label='20k', markersize=4)
        ax7.plot(df['dimension'][:n_show], df['sv_50k_aligned'][:n_show], 'r-s', 
                label='50k aligned', markersize=4, alpha=0.7)
        ax7.set_xlabel('Dimension')
        ax7.set_ylabel('Singular Value')
        ax7.set_title('Top 30 Dimensions Detail')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. 变化率分布直方图
        ax8 = fig.add_subplot(gs[2, 0])
        ax8.hist(df['relative_change_%'], bins=30, edgecolor='black', alpha=0.7)
        ax8.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Relative Change (%)')
        ax8.set_ylabel('Count')
        ax8.set_title('Distribution of Changes')
        ax8.grid(True, alpha=0.3)
        
        # 9. 权重矩阵热图差异（采样）
        ax9 = fig.add_subplot(gs[2, 1:3])
        # 这里需要原始权重矩阵，暂时用奇异值重构的近似
        n_components = 20
        U_20k = results.get('U_20k', np.eye(n_components))[:, :n_components]
        U_50k = results.get('U_50k', np.eye(n_components))[:, :n_components]
        diff_matrix = np.abs(U_50k - U_20k)
        im = ax9.imshow(diff_matrix, aspect='auto', cmap='hot')
        ax9.set_title('Weight Difference Pattern (Top 20 components)')
        ax9.set_xlabel('Component')
        ax9.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax9)
        
        # 10. 摘要文本
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis('off')
        summary_text = f"""
ANALYSIS SUMMARY
{'='*25}
Layer: {results['layer_name']}
Shape: {results['shape']}

Alignment Quality:
• Cosine: {results['similarity_before']['cosine']:.4f} → {results['similarity_after']['cosine']:.4f}
• Improvement: +{results['similarity_after']['cosine'] - results['similarity_before']['cosine']:.4f}

Dimension Status:
• Stable: {results['status_counts'].get('stable', 0)}
• Changed: {results['status_counts'].get('changed', 0)}
• Collapsed: {results['status_counts'].get('collapsed', 0)}
• Exploded: {results['status_counts'].get('exploded', 0)}

Effective Rank (90%):
• 20k: {results['effective_ranks']['90%']['20k']}
• 50k: {results['effective_ranks']['90%']['50k']}
• Change: {results['effective_ranks']['90%']['50k'] - results['effective_ranks']['90%']['20k']}
        """
        ax10.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        # 总标题
        fig.suptitle(f'MLP {results["layer_name"]} Evolution: 20k vs 50k (Procrustes Aligned)\n' + 
                    f'Performance Drop: 72% → 32% | Similarity: {results["similarity_before"]["cosine"]:.3f} → {results["similarity_after"]["cosine"]:.3f}',
                    fontsize=14, fontweight='bold')
        
        # 保存
        save_dir = "checkpoint_evolution"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'mlp_{results["layer_name"]}_analysis_{self.timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ 图像已保存: {save_path}")
        
        return save_path
    
    def save_results(self, results: Dict, layer_name: str):
        """保存分析结果（修复JSON序列化问题）"""
        import numpy as np
        
        save_dir = "checkpoint_evolution"
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存CSV
        csv_path = os.path.join(save_dir, f'mlp_{layer_name}_dimensions_{self.timestamp}.csv')
        results['per_dimension_analysis'].to_csv(csv_path, index=False)
        print(f"✅ CSV已保存: {csv_path}")
        
        # 2. 保存JSON摘要（转换numpy类型）
        def convert_to_native(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, pd.Series):
                return convert_to_native(obj.to_dict())
            else:
                return obj
        
        json_results = {
            'layer_name': results['layer_name'],
            'shape': list(results['shape']),
            'similarity_before': convert_to_native(results['similarity_before']),
            'similarity_after': convert_to_native(results['similarity_after']),
            'status_counts': convert_to_native(results['status_counts']),
            'effective_ranks': convert_to_native(results['effective_ranks'])
        }
        
        json_path = os.path.join(save_dir, f'mlp_{layer_name}_summary_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ JSON已保存: {json_path}")
        
        # 3. 保存NPZ
        npz_path = os.path.join(save_dir, f'mlp_{layer_name}_svd_{self.timestamp}.npz')
        np.savez(npz_path,
                sv_20k=results['singular_values']['sv_20k'],
                sv_50k=results['singular_values']['sv_50k'],
                sv_50k_aligned=results['singular_values']['sv_50k_aligned'],
                rotation_matrix=results['rotation_matrix'])
        print(f"✅ NPZ已保存: {npz_path}")
        
        return csv_path, json_path, npz_path

def main():
    """主函数"""
    print("="*80)
    print("🔬 MLP Layer Evolution Analysis")
    print("="*80)
    
    # 设置路径
    checkpoint_dir = "out/composition_mix0_seed42_20250705_065227"
    ckpt_20k_path = os.path.join(checkpoint_dir, "ckpt_mix0_seed42_iter20000.pt")
    ckpt_50k_path = os.path.join(checkpoint_dir, "ckpt_mix0_seed42_iter50000.pt")
    
    # 创建分析器
    analyzer = MLPEvolutionAnalyzer(checkpoint_dir)
    
    # 加载权重
    weights_20k = analyzer.load_mlp_weights(ckpt_20k_path)
    weights_50k = analyzer.load_mlp_weights(ckpt_50k_path)
    
    # 分析两个MLP层
    all_results = {}
    
    # 1. 分析c_fc层 (480, 120)
    print("\n" + "="*80)
    print("📊 分析 c_fc 层 (Input → Hidden)")
    results_fc = analyzer.analyze_layer(
        weights_20k['c_fc'], 
        weights_50k['c_fc'],
        'c_fc'
    )
    all_results['c_fc'] = results_fc
    
    # 2. 分析c_proj层 (120, 480) - 问题最严重的层
    print("\n" + "="*80)
    print("📊 分析 c_proj 层 (Hidden → Output) - PROBLEMATIC!")
    results_proj = analyzer.analyze_layer(
        weights_20k['c_proj'],
        weights_50k['c_proj'],
        'c_proj'
    )
    all_results['c_proj'] = results_proj
    
    # 创建可视化
    print("\n" + "="*80)
    print("📊 创建可视化...")
    for layer_name, results in all_results.items():
        analyzer.create_visualization(results)
        analyzer.save_results(results, layer_name)
    
    # 对比分析
    print("\n" + "="*80)
    print("📊 层间对比分析")
    print("="*80)
    
    print(f"\n{'Layer':<10} {'Shape':<15} {'Cosine Before':<15} {'Cosine After':<15} {'Improvement':<12} {'Collapsed':<10} {'Exploded':<10}")
    print("-"*90)
    
    for layer_name, results in all_results.items():
        cosine_before = results['similarity_before']['cosine']
        cosine_after = results['similarity_after']['cosine']
        improvement = cosine_after - cosine_before
        collapsed = results['status_counts'].get('collapsed', 0)
        exploded = results['status_counts'].get('exploded', 0)
        shape_str = f"{results['shape']}"
        
        # 标记问题层
        marker = " ⚠️" if cosine_after < 0.8 else ""
        print(f"{layer_name:<10} {shape_str:<15} {cosine_before:<15.4f} {cosine_after:<15.4f} "
              f"{improvement:<12.4f} {collapsed:<10} {exploded:<10}{marker}")
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
    
    # 关键发现总结
    proj_results = all_results['c_proj']
    print("\n📊 关键发现 (c_proj层):")
    print(f"  1. 对齐后相似度: {proj_results['similarity_after']['cosine']:.4f}")
    print(f"  2. 崩溃维度数: {proj_results['status_counts'].get('collapsed', 0)}")
    print(f"  3. 爆炸维度数: {proj_results['status_counts'].get('exploded', 0)}")
    print(f"  4. 有效秩变化: {proj_results['effective_ranks']['90%']['20k']} → {proj_results['effective_ranks']['90%']['50k']}")
    
    if proj_results['similarity_after']['cosine'] < 0.7:
        print("\n  ⚠️ c_proj层严重退化，这解释了性能下降！")
    
    return all_results

if __name__ == "__main__":
    results = main()