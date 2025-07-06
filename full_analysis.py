#!/usr/bin/env python3
"""
完整分析脚本：混合比例 -> 权重统计 -> 组合性能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_merge_data():
    """加载并合并数据"""
    print("加载数据...")
    success_df = pd.read_csv('success_log.csv')
    collapse_df = pd.read_csv('collapse_metrics.csv')
    
    # 合并数据
    df = pd.merge(success_df, collapse_df, on=['ratio', 'seed', 'iter'])
    
    # 计算WRC* (如果还没有)
    if 'WRC_star' not in df.columns:
        df['WRC_star'] = df['sigma_weight'] * (1 - df['direction_diversity'])
    
    # 计算row_norm_std (如果需要)
    if 'row_norm_std' not in df.columns:
        df['row_norm_std'] = df['sigma_weight']  # 如果sigma_weight就是row_norm_std
    
    return df

def analyze_by_ratio(df):
    """按混合比例分析各指标"""
    print("\n分析各混合比例下的指标...")
    
    # 只看收敛后的数据
    stable_df = df[df['iter'] >= 10000]
    
    # 按ratio分组统计
    ratio_stats = stable_df.groupby('ratio').agg({
        'success': ['mean', 'std', 'count'],
        'direction_diversity': ['mean', 'std'],
        'row_norm_std': ['mean', 'std'],
        'WRC_star': ['mean', 'std'],
        'effective_rank': ['mean', 'std']
    }).round(4)
    
    print("\n=== 各混合比例的指标统计 (iter >= 10000) ===")
    print(ratio_stats)
    
    return ratio_stats

def plot_ratio_trends(df):
    """绘制混合比例与各指标的关系"""
    print("\n生成趋势图...")
    
    # 使用最后10个checkpoint的平均值
    last_10_iters = df.groupby(['ratio', 'seed'])['iter'].nlargest(10).index.get_level_values(2)
    last_10_df = df.loc[last_10_iters]
    
    # 计算平均值和标准误
    ratio_trends = last_10_df.groupby('ratio').agg({
        'success': ['mean', 'std', 'sem'],
        'direction_diversity': ['mean', 'std', 'sem'],
        'row_norm_std': ['mean', 'std', 'sem'],
        'WRC_star': ['mean', 'std', 'sem']
    })
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Success Rate
    ax = axes[0, 0]
    ratios = ratio_trends.index
    means = ratio_trends[('success', 'mean')]
    sems = ratio_trends[('success', 'sem')]
    
    ax.errorbar(ratios, means, yerr=sems, marker='o', linewidth=2, markersize=8, capsize=5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target = 0.8')
    ax.axvline(x=6, color='green', linestyle='--', alpha=0.5, label='Critical ratio = 6%')
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs Mixture Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Direction Diversity
    ax = axes[0, 1]
    means = ratio_trends[('direction_diversity', 'mean')]
    sems = ratio_trends[('direction_diversity', 'sem')]
    
    ax.errorbar(ratios, means, yerr=sems, marker='s', linewidth=2, markersize=8, capsize=5, color='orange')
    ax.axvline(x=6, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('Direction Diversity')
    ax.set_title('Direction Diversity vs Mixture Ratio')
    ax.grid(True, alpha=0.3)
    
    # 3. Row Norm Std
    ax = axes[1, 0]
    means = ratio_trends[('row_norm_std', 'mean')]
    sems = ratio_trends[('row_norm_std', 'sem')]
    
    ax.errorbar(ratios, means, yerr=sems, marker='^', linewidth=2, markersize=8, capsize=5, color='green')
    ax.axvline(x=6, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('Row Norm Std')
    ax.set_title('Row Norm Std vs Mixture Ratio')
    ax.grid(True, alpha=0.3)
    
    # 4. WRC*
    ax = axes[1, 1]
    means = ratio_trends[('WRC_star', 'mean')]
    sems = ratio_trends[('WRC_star', 'sem')]
    
    ax.errorbar(ratios, means, yerr=sems, marker='d', linewidth=2, markersize=8, capsize=5, color='red')
    ax.axhline(y=0.06, color='red', linestyle='--', alpha=0.5, label='Threshold = 0.06')
    ax.axvline(x=6, color='green', linestyle='--', alpha=0.5, label='Critical ratio = 6%')
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('WRC*')
    ax.set_title('WRC* vs Mixture Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ratio_trends.png', dpi=300, bbox_inches='tight')
    print("已保存: ratio_trends.png")
    
    return ratio_trends

def analyze_low_ratio_region(df):
    """分析低混合比例区间 (<=6%)"""
    print("\n分析低混合比例区间...")
    
    # 筛选数据
    low_ratio_df = df[(df['ratio'] <= 6) & (df['iter'] >= 10000)]
    high_ratio_df = df[(df['ratio'] > 6) & (df['iter'] >= 10000)]
    
    # 回归分析
    features = ['row_norm_std', 'direction_diversity', 'WRC_star']
    
    print("\n=== 回归分析：预测Success Rate ===")
    print("特征：", features)
    
    # 低混合比例区间
    X_low = low_ratio_df[features]
    y_low = low_ratio_df['success']
    
    lr_low = LinearRegression()
    lr_low.fit(X_low, y_low)
    r2_low = lr_low.score(X_low, y_low)
    
    print(f"\n低混合比例 (ratio <= 6%):")
    print(f"  R² = {r2_low:.3f}")
    print("  系数:")
    for feat, coef in zip(features, lr_low.coef_):
        print(f"    {feat}: {coef:.3f}")
    
    # 高混合比例区间
    X_high = high_ratio_df[features]
    y_high = high_ratio_df['success']
    
    lr_high = LinearRegression()
    lr_high.fit(X_high, y_high)
    r2_high = lr_high.score(X_high, y_high)
    
    print(f"\n高混合比例 (ratio > 6%):")
    print(f"  R² = {r2_high:.3f}")
    print("  系数:")
    for feat, coef in zip(features, lr_high.coef_):
        print(f"    {feat}: {coef:.3f}")
    
    # WRC*单独分析
    print("\n=== WRC*与Success的相关性 ===")
    
    # 低混合区间
    corr_low = low_ratio_df['WRC_star'].corr(low_ratio_df['success'])
    print(f"低混合比例区间: Pearson r = {corr_low:.3f}")
    
    # 计算AUC（如果想要）
    if low_ratio_df['success'].std() > 0:
        # 创建二元标签
        threshold = low_ratio_df['success'].median()
        binary_label = (low_ratio_df['success'] > threshold).astype(int)
        if binary_label.nunique() > 1:
            auc_low = roc_auc_score(binary_label, -low_ratio_df['WRC_star'])
            print(f"  AUC (median split) = {auc_low:.3f}")
    
    # 高混合区间
    corr_high = high_ratio_df['WRC_star'].corr(high_ratio_df['success'])
    print(f"高混合比例区间: Pearson r = {corr_high:.3f}")
    
    return lr_low, lr_high

def plot_wrc_analysis(df):
    """WRC*的详细分析图"""
    print("\n生成WRC*分析图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 只看收敛后数据
    stable_df = df[df['iter'] >= 10000]
    
    # 1. WRC* vs Success散点图（按ratio着色）
    ax = axes[0, 0]
    scatter = ax.scatter(stable_df['WRC_star'], stable_df['success'], 
                        c=stable_df['ratio'], cmap='viridis', alpha=0.6, s=20)
    ax.axvline(x=0.06, color='red', linestyle='--', alpha=0.5, label='WRC* = 0.06')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Success = 0.8')
    ax.set_xlabel('WRC*')
    ax.set_ylabel('Success Rate')
    ax.set_title('WRC* vs Success Rate')
    plt.colorbar(scatter, ax=ax, label='Ratio (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. WRC*箱线图（低混合比例）
    ax = axes[0, 1]
    low_ratio_df = stable_df[stable_df['ratio'] <= 6]
    if len(low_ratio_df) > 0:
        # 创建WRC*分箱
        try:
            low_ratio_df['WRC_bin'] = pd.qcut(low_ratio_df['WRC_star'], q=5, duplicates='drop')
            sns.boxplot(data=low_ratio_df, x='WRC_bin', y='success', ax=ax)
            ax.set_xlabel('WRC* Bins')
            ax.set_ylabel('Success Rate')
            ax.set_title('Success by WRC* Bins (ratio <= 6%)')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        except:
            ax.text(0.5, 0.5, 'Not enough data for binning', 
                   transform=ax.transAxes, ha='center')
    
    # 3. 混合比例分组的WRC*分布
    ax = axes[1, 0]
    ratio_groups = [0, 1, 2, 3, 4, 5, 6, 10, 15, 20]
    wrc_by_ratio = []
    for r in ratio_groups:
        if r in stable_df['ratio'].values:
            wrc_values = stable_df[stable_df['ratio'] == r]['WRC_star'].values
            wrc_by_ratio.append(wrc_values)
        else:
            wrc_by_ratio.append([])
    
    ax.boxplot(wrc_by_ratio, labels=ratio_groups)
    ax.axhline(y=0.06, color='red', linestyle='--', alpha=0.5, label='Threshold = 0.06')
    ax.set_xlabel('Mixture Ratio (%)')
    ax.set_ylabel('WRC*')
    ax.set_title('WRC* Distribution by Mixture Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 相关性热力图
    ax = axes[1, 1]
    metrics = ['success', 'direction_diversity', 'row_norm_std', 'WRC_star', 'effective_rank']
    corr_matrix = stable_df[metrics].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('wrc_analysis.png', dpi=300, bbox_inches='tight')
    print("已保存: wrc_analysis.png")

def generate_summary_table(df):
    """生成用于论文的汇总表"""
    print("\n生成汇总表...")
    
    # 使用最后10个checkpoint
    last_10_iters = df.groupby(['ratio', 'seed'])['iter'].nlargest(10).index.get_level_values(2)
    last_10_df = df.loc[last_10_iters]
    
    # 创建汇总
    summary = []
    for ratio in sorted(last_10_df['ratio'].unique()):
        ratio_data = last_10_df[last_10_df['ratio'] == ratio]
        summary.append({
            'Ratio (%)': ratio,
            'Success Rate': f"{ratio_data['success'].mean():.3f} ± {ratio_data['success'].std():.3f}",
            'Direction Diversity': f"{ratio_data['direction_diversity'].mean():.3f}",
            'Row Norm Std': f"{ratio_data['row_norm_std'].mean():.3f}",
            'WRC*': f"{ratio_data['WRC_star'].mean():.3f}",
            'Above 0.8 (%)': f"{(ratio_data['success'] >= 0.8).mean() * 100:.1f}"
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n=== 论文用汇总表 ===")
    print(summary_df.to_string(index=False))
    
    # 保存为CSV
    summary_df.to_csv('summary_table.csv', index=False)
    print("\n已保存: summary_table.csv")
    
    return summary_df

def main():
    """主函数"""
    # 1. 加载数据
    df = load_and_merge_data()
    print(f"总数据量: {len(df)} 条记录")
    
    # 2. 按比例分析
    ratio_stats = analyze_by_ratio(df)
    
    # 3. 绘制趋势图
    ratio_trends = plot_ratio_trends(df)
    
    # 4. 低混合比例区间分析
    lr_low, lr_high = analyze_low_ratio_region(df)
    
    # 5. WRC*详细分析
    plot_wrc_analysis(df)
    
    # 6. 生成汇总表
    summary_df = generate_summary_table(df)
    
    # 7. 关键发现总结
    print("\n" + "="*50)
    print("关键发现总结")
    print("="*50)
    
    stable_df = df[df['iter'] >= 10000]
    
    # 找出转折点
    success_by_ratio = stable_df.groupby('ratio')['success'].mean()
    above_80_ratio = success_by_ratio[success_by_ratio >= 0.8].index.min()
    
    print(f"1. 临界混合比例: {above_80_ratio}%")
    print(f"   - 当ratio >= {above_80_ratio}%时，平均成功率 >= 0.8")
    
    # WRC*阈值
    high_success = stable_df[stable_df['success'] >= 0.8]
    wrc_threshold = high_success['WRC_star'].quantile(0.95)
    print(f"\n2. WRC*阈值: {wrc_threshold:.3f}")
    print(f"   - 95%的高成功率样本满足 WRC* < {wrc_threshold:.3f}")
    
    # 低混合区间的预测能力
    low_ratio_df = stable_df[stable_df['ratio'] <= 6]
    if len(low_ratio_df) > 0:
        corr = low_ratio_df['WRC_star'].corr(low_ratio_df['success'])
        print(f"\n3. 低混合区间(ratio<=6%)的WRC*预测能力:")
        print(f"   - 与成功率的相关性: r = {corr:.3f}")
    
    print("\n分析完成！")
    print("\n生成的文件:")
    print("- ratio_trends.png: 混合比例与各指标的关系")
    print("- wrc_analysis.png: WRC*的详细分析")
    print("- summary_table.csv: 论文用汇总表")

if __name__ == "__main__":
    main()