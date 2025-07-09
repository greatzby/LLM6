#!/usr/bin/env python3
# coding: utf-8
"""
生成关键的可视化图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_spectral_evolution():
    """绘制奇异值谱的演化"""
    print("绘制谱演化图...")
    
    # 加载谱数据
    with open('spectral_evolution.csv', 'r') as f:
        spectral_data = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 选择要展示的iteration
    iterations = [5000, 10000, 25000, 50000, 75000, 100000]
    
    for idx, target_iter in enumerate(iterations):
        ax = axes[idx]
        
        # 为每个ratio绘制
        for ratio in [0, 5, 10]:
            # 获取该ratio在target_iter附近的数据
            ratio_data = [d for d in spectral_data 
                         if d['ratio'] == ratio and 
                         abs(d['iteration'] - target_iter) < 2500]
            
            if ratio_data:
                # 平均多个seed的结果
                all_sv = np.array([d['singular_values'] for d in ratio_data])
                mean_sv = np.mean(all_sv, axis=0)
                
                # 绘制
                indices = np.arange(1, len(mean_sv) + 1)
                ax.loglog(indices, mean_sv, 'o-', 
                         label=f'{ratio}% mix', alpha=0.8, markersize=4)
        
        ax.set_title(f'Iteration {target_iter}')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Spectral Evolution: How Tail Decays', fontsize=16)
    plt.tight_layout()
    plt.savefig('spectral_evolution.png', dpi=300, bbox_inches='tight')
    print("✓ 保存到 spectral_evolution.png")

def plot_similarity_heatmap():
    """绘制token相似度热图"""
    print("绘制相似度热图...")
    
    # 加载数据
    df = pd.read_csv('forgetting_metrics_complete.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, ratio in enumerate([0, 5, 10]):
        ax = axes[idx]
        
        # 获取最后的checkpoint数据
        ratio_df = df[(df['ratio'] == ratio) & (df['iteration'] > 90000)]
        
        if len(ratio_df) > 0:
            # 构建3x3相似度矩阵
            sim_matrix = np.array([
                [ratio_df['within_S1_sim'].mean(), 
                 ratio_df['S1_S2_sim'].mean(), 
                 ratio_df['S1_S3_sim'].mean()],
                [ratio_df['S1_S2_sim'].mean(), 
                 ratio_df['within_S2_sim'].mean(), 
                 ratio_df['S2_S3_sim'].mean()],
                [ratio_df['S1_S3_sim'].mean(), 
                 ratio_df['S2_S3_sim'].mean(), 
                 ratio_df['within_S3_sim'].mean()]
            ])
            
            # 绘制热图
            sns.heatmap(sim_matrix, annot=True, fmt='.3f', 
                       cmap='RdYlBu_r', center=0.5,
                       xticklabels=['S1', 'S2', 'S3'],
                       yticklabels=['S1', 'S2', 'S3'],
                       ax=ax, vmin=0, vmax=1,
                       cbar_kws={'label': 'Cosine Similarity'})
            
            ax.set_title(f'{ratio}% mix (iter>90k)')
    
    plt.suptitle('Token Similarity Structure at Convergence', fontsize=16)
    plt.tight_layout()
    plt.savefig('similarity_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 保存到 similarity_heatmap.png")

def plot_risk_surface():
    """绘制多指标风险表面"""
    print("绘制风险表面...")
    
    df = pd.read_csv('forgetting_metrics_complete.csv')
    
    # 假设你有success rate数据
    # 如果没有，这里用effective_rank作为代理
    if 'success_rate' not in df.columns:
        # 用ER作为success的代理（ER越高越好）
        df['success_proxy'] = (df['effective_rank'] - 60) / 10
        df['success_proxy'] = df['success_proxy'].clip(0, 1)
    else:
        df['success_proxy'] = df['success_rate']
    
    # 创建2D网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ER vs Ratio
    ax = axes[0, 0]
    scatter = ax.scatter(df['ratio'], df['effective_rank'], 
                        c=df['success_proxy'], cmap='RdYlGn',
                        s=50, alpha=0.6)
    ax.axhline(y=63, color='red', linestyle='--', label='Critical ER')
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('Effective Rank')
    ax.set_title('ER vs Ratio colored by Success')
    plt.colorbar(scatter, ax=ax, label='Success Rate')
    
    # 2. Clustering vs Bridge Score
    ax = axes[0, 1]
    scatter = ax.scatter(df['clustering_coef'], df['bridge_score'],
                        c=df['success_proxy'], cmap='RdYlGn',
                        s=50, alpha=0.6)
    ax.set_xlabel('Clustering Coefficient')
    ax.set_ylabel('S2 Bridge Score')
    ax.set_title('Token Organization')
    plt.colorbar(scatter, ax=ax, label='Success Rate')
    
    # 3. Composite Risk Score
    ax = axes[1, 0]
    for ratio in sorted(df['ratio'].unique()):
        ratio_df = df[df['ratio'] == ratio]
        ratio_df = ratio_df.sort_values('iteration')
        ax.plot(ratio_df['iteration'], ratio_df['composite_risk'],
               label=f'{ratio}% mix', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composite Risk Score')
    ax.set_title('Risk Evolution')
    ax.legend()
    
    # 4. Correlation heatmap
    ax = axes[1, 1]
    key_metrics = ['effective_rank', 'direction_diversity', 'clustering_coef',
                   'bridge_score', 'row_norm_cv', 'S1_S3_sim', 'success_proxy']
    corr_df = df[key_metrics].corr()
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True)
    ax.set_title('Metric Correlations')
    
    plt.suptitle('Multi-Metric Risk Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('risk_surface.png', dpi=300, bbox_inches='tight')
    print("✓ 保存到 risk_surface.png")

def plot_forgetting_timeline():
    """绘制forgetting的时间线"""
    print("绘制时间线...")
    
    df = pd.read_csv('forgetting_metrics_complete.csv')
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # 选择关键指标
    metrics = [
        ('effective_rank', 'Effective Rank', True),
        ('direction_diversity', 'Direction Diversity', True),
        ('S1_S3_sim', 'S1-S3 Similarity', False),
        ('bridge_score', 'S2 Bridge Score', True),
        ('row_norm_cv', 'Row Norm CV', False),
        ('condition_number', 'Condition Number', False)
    ]
    
    for idx, (metric, title, higher_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 为每个ratio计算均值和置信区间
        for ratio in sorted(df['ratio'].unique()):
            ratio_df = df[df['ratio'] == ratio]
            
            # 按iteration分组
            grouped = ratio_df.groupby('iteration')[metric].agg(['mean', 'std', 'count'])
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # 绘制均值线和置信区间
            ax.plot(grouped.index, grouped['mean'], label=f'{ratio}% mix', linewidth=2)
            ax.fill_between(grouped.index, 
                          grouped['mean'] - 1.96 * grouped['se'],
                          grouped['mean'] + 1.96 * grouped['se'],
                          alpha=0.2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加"好/坏"区域
        if metric == 'effective_rank':
            ax.axhspan(0, 63, alpha=0.1, color='red', label='Risk Zone')
        elif metric == 'S1_S3_sim' and not higher_better:
            ax.axhspan(0.8, 1.0, alpha=0.1, color='red', label='Too Similar')
    
    plt.suptitle('Temporal Evolution of Key Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('forgetting_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ 保存到 forgetting_timeline.png")

def generate_summary_table():
    """生成总结表格"""
    print("生成总结表格...")
    
    df = pd.read_csv('forgetting_metrics_complete.csv')
    
    # 只看最后阶段的数据
    df_final = df[df['iteration'] > 90000]
    
    # 按ratio分组统计
    summary = df_final.groupby('ratio').agg({
        'effective_rank': ['mean', 'std'],
        'direction_diversity': ['mean', 'std'],
        'S1_S3_sim': ['mean', 'std'],
        'bridge_score': ['mean', 'std'],
        'composite_risk': ['mean', 'std']
    }).round(3)
    
    # 保存
    summary.to_csv('final_metrics_summary.csv')
    print("✓ 保存到 final_metrics_summary.csv")
    print("\n最终指标总结：")
    print(summary)

def main():
    """运行所有可视化"""
    plot_spectral_evolution()
    plot_similarity_heatmap()
    plot_risk_surface()
    plot_forgetting_timeline()
    generate_summary_table()
    print("\n✓ 所有图表生成完成！")

if __name__ == "__main__":
    main()