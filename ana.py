import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr

# 1. 加载数据
collapse_df = pd.read_csv('collapse_metrics.csv')
success_df = pd.read_csv('success_log.csv')

# 2. 合并数据
merged_df = pd.merge(collapse_df, success_df, on=['ratio', 'seed', 'iter'])

# 检查success列的值
print("=== Success列统计信息 ===")
print(merged_df['success'].describe())
print(f"\nUnique values count: {merged_df['success'].nunique()}")

# 3. 如果success是连续值，定义成功的阈值
# 通常HellaSwag的随机猜测准确率是25%，所以可以用更高的阈值
SUCCESS_THRESHOLD = 0.30  # 可以根据需要调整

# 创建二进制成功标签
merged_df['success_binary'] = (merged_df['success'] >= SUCCESS_THRESHOLD).astype(int)
print(f"\n使用阈值 {SUCCESS_THRESHOLD} 将success转换为二进制")
print(f"成功样本比例: {merged_df['success_binary'].mean():.1%}")

# 4. 计算各指标与成功率的相关性（使用连续值）
print("\n=== 各指标与成功率的相关性 ===")
metrics = ['direction_diversity', 'sigma_weight', 'BRC_star_adapted', 
           'token_spread', 'collapse_score', 'norm_variance', 'effective_rank']

correlations = {}
for metric in metrics:
    # Pearson相关系数
    pearson_corr, p_val = pearsonr(merged_df[metric], merged_df['success'])
    # Spearman相关系数（对非线性关系更鲁棒）
    spearman_corr, _ = spearmanr(merged_df[metric], merged_df['success'])
    correlations[metric] = pearson_corr
    print(f"{metric:20s}: Pearson={pearson_corr:6.3f}, Spearman={spearman_corr:6.3f}")

# 5. 过滤稳定数据
stable_df = merged_df[merged_df['iter'] >= 10000].copy()
stable_df['success_binary'] = (stable_df['success'] >= SUCCESS_THRESHOLD).astype(int)

# 6. 计算BRC_star_adapted的AUC（使用二进制标签）
if stable_df['success_binary'].nunique() > 1:  # 确保有两个类别
    auc = roc_auc_score(stable_df['success_binary'], 1 - stable_df['BRC_star_adapted'])
    print(f"\nBRC_star_adapted AUC (threshold={SUCCESS_THRESHOLD}): {auc:.3f}")
    
    # 7. 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(stable_df['success_binary'], 1 - stable_df['BRC_star_adapted'])
    plt.plot(fpr, tpr, label=f'BRC* (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for BRC_star_adapted\n(Success threshold = {SUCCESS_THRESHOLD})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('brc_roc_curve.png', dpi=300)
    plt.close()
else:
    print("\n警告：所有样本都属于同一类别，无法计算AUC")

# 8. 散点图：BRC vs Success（连续值）
plt.figure(figsize=(12, 10))

# 子图1: BRC vs Success散点图
plt.subplot(2, 2, 1)
colors = plt.cm.viridis(stable_df['ratio'] / 20)
scatter = plt.scatter(stable_df['BRC_star_adapted'], stable_df['success'], 
                     c=colors, alpha=0.5, s=20)
plt.xlabel('BRC_star_adapted')
plt.ylabel('Success Rate')
plt.title('BRC* vs Success Rate')
plt.axhline(y=SUCCESS_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Threshold={SUCCESS_THRESHOLD}')
plt.colorbar(scatter, label='Ratio (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 按比例分组的箱线图
plt.subplot(2, 2, 2)
ratio_groups = []
for ratio in sorted(stable_df['ratio'].unique()):
    ratio_data = stable_df[stable_df['ratio'] == ratio]
    ratio_groups.append({
        'ratio': ratio,
        'BRC_mean': ratio_data['BRC_star_adapted'].mean(),
        'BRC_std': ratio_data['BRC_star_adapted'].std(),
        'success_mean': ratio_data['success'].mean(),
        'success_std': ratio_data['success'].std()
    })
ratio_df = pd.DataFrame(ratio_groups)

# 绘制BRC均值和误差条
plt.errorbar(ratio_df['ratio'], ratio_df['BRC_mean'], 
             yerr=ratio_df['BRC_std'], fmt='bo-', label='BRC*', capsize=5)
plt.xlabel('Mixture Ratio (%)')
plt.ylabel('BRC_star_adapted', color='b')
plt.tick_params(axis='y', labelcolor='b')

# 添加成功率在右轴
ax2 = plt.twinx()
ax2.errorbar(ratio_df['ratio'], ratio_df['success_mean'], 
             yerr=ratio_df['success_std'], fmt='ro-', label='Success', capsize=5)
ax2.set_ylabel('Success Rate', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.axhline(y=SUCCESS_THRESHOLD, color='r', linestyle='--', alpha=0.5)

plt.title('BRC* and Success Rate by Mixture Ratio')
plt.grid(True, alpha=0.3)

# 子图3: 相关性热力图
plt.subplot(2, 2, 3)
# 计算所有指标之间的相关性
corr_matrix = stable_df[metrics + ['success']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Collapse Metrics')

# 子图4: 成功率分布
plt.subplot(2, 2, 4)
for ratio in [0, 5, 10, 15, 20]:
    data = stable_df[stable_df['ratio'] == ratio]['success']
    plt.hist(data, bins=20, alpha=0.5, label=f'{ratio}%', density=True)
plt.axvline(x=SUCCESS_THRESHOLD, color='r', linestyle='--', alpha=0.5, 
            label=f'Threshold={SUCCESS_THRESHOLD}')
plt.xlabel('Success Rate')
plt.ylabel('Density')
plt.title('Success Rate Distribution by Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('collapse_metrics_analysis.png', dpi=300)
plt.close()

# 9. 详细统计表
print("\n=== 按混合比例的统计 (iter≥10000) ===")
print("Ratio | Avg BRC* | Avg Success | Success>30% | Collapse Risk")
print("-" * 60)
for ratio in sorted(stable_df['ratio'].unique()):
    ratio_data = stable_df[stable_df['ratio'] == ratio]
    avg_brc = ratio_data['BRC_star_adapted'].mean()
    avg_success = ratio_data['success'].mean()
    success_rate = (ratio_data['success'] >= SUCCESS_THRESHOLD).mean()
    
    # 定义崩溃风险级别
    if avg_brc > 0.4:
        risk = "High"
    elif avg_brc > 0.3:
        risk = "Medium"
    else:
        risk = "Low"
    
    print(f"{ratio:5.0f} | {avg_brc:8.3f} | {avg_success:11.3f} | {success_rate:11.1%} | {risk}")

# 10. BRC阈值分析
print("\n=== BRC*阈值分析 ===")
brc_thresholds = np.percentile(stable_df['BRC_star_adapted'], [10, 25, 50, 75, 90])
print("BRC* Percentiles:")
for p, v in zip([10, 25, 50, 75, 90], brc_thresholds):
    success_above = stable_df[stable_df['BRC_star_adapted'] <= v]['success'].mean()
    print(f"  {p}th percentile: BRC*={v:.3f}, Avg Success={success_above:.3f}")

# 11. 创建性能摘要图
plt.figure(figsize=(10, 6))

# 计算移动平均以平滑曲线
window_size = 100
stable_sorted = stable_df.sort_values('BRC_star_adapted')
brc_values = stable_sorted['BRC_star_adapted'].rolling(window=window_size).mean()
success_values = stable_sorted['success'].rolling(window=window_size).mean()

plt.plot(brc_values, success_values, 'b-', alpha=0.7, linewidth=2, 
         label='Moving Average')
plt.scatter(stable_df['BRC_star_adapted'], stable_df['success'], 
            c=stable_df['ratio'], cmap='viridis', alpha=0.3, s=10)
plt.xlabel('BRC_star_adapted')
plt.ylabel('Success Rate')
plt.title('BRC* vs Success Rate (with moving average)')
plt.axhline(y=SUCCESS_THRESHOLD, color='r', linestyle='--', alpha=0.5, 
            label=f'Success Threshold={SUCCESS_THRESHOLD}')
plt.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, 
            label='Random Baseline (25%)')
plt.colorbar(label='Mixture Ratio (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('brc_success_summary.png', dpi=300)
plt.close()

print("\n分析完成！已生成图表：")
print("- brc_roc_curve.png: ROC曲线")
print("- collapse_metrics_analysis.png: 综合分析")
print("- brc_success_summary.png: BRC与成功率关系")