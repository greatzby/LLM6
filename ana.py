import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# 1. 加载数据
collapse_df = pd.read_csv('collapse_metrics.csv')
success_df = pd.read_csv('success_log_detailed.csv')

# 2. 合并数据
merged_df = pd.merge(collapse_df, success_df, on=['ratio', 'seed', 'iter'])

# 3. 计算每个指标与成功的相关性
print("=== 各指标与成功率的相关性 ===")
metrics = ['direction_diversity', 'sigma_weight', 'BRC_star_adapted', 
           'token_spread', 'collapse_score', 'norm_variance', 'effective_rank']

correlations = {}
for metric in metrics:
    corr = merged_df[metric].corr(merged_df['success'])
    correlations[metric] = corr
    print(f"{metric}: {corr:.3f}")

# 4. 计算BRC_star_adapted的AUC
# 过滤掉早期迭代
stable_df = merged_df[merged_df['iter'] >= 10000]

# 计算AUC (注意：低BRC值应该预测高成功率，所以要反转)
auc = roc_auc_score(stable_df['success'], 1 - stable_df['BRC_star_adapted'])
print(f"\nBRC_star_adapted AUC: {auc:.3f}")

# 5. 绘制ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(stable_df['success'], 1 - stable_df['BRC_star_adapted'])
plt.plot(fpr, tpr, label=f'BRC* (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for BRC_star_adapted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('brc_roc_curve.png', dpi=300)
plt.show()

# 6. 找出最佳阈值
# 使用Youden's J statistic
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = 1 - thresholds[best_idx]  # 转换回BRC值
print(f"\n最佳BRC阈值: {best_threshold:.3f}")
print(f"对应的敏感度: {tpr[best_idx]:.3f}")
print(f"对应的特异度: {1-fpr[best_idx]:.3f}")

# 7. 按混合比例分析
plt.figure(figsize=(12, 8))

# 子图1: BRC vs Success by Ratio
plt.subplot(2, 2, 1)
ratio_groups = stable_df.groupby('ratio').agg({
    'BRC_star_adapted': 'mean',
    'success': 'mean'
}).reset_index()

plt.scatter(ratio_groups['ratio'], ratio_groups['BRC_star_adapted'], 
            s=100, alpha=0.7, label='Avg BRC*')
plt.xlabel('Mixture Ratio (%)')
plt.ylabel('Average BRC_star_adapted')
plt.title('BRC* vs Mixture Ratio')
plt.grid(True, alpha=0.3)

# 子图2: 成功率覆盖
ax2 = plt.twinx()
ax2.plot(ratio_groups['ratio'], ratio_groups['success'], 
         'r-', marker='o', label='Success Rate')
ax2.set_ylabel('Success Rate', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 子图3: 崩溃指标热力图
plt.subplot(2, 2, 3)
metric_by_ratio = stable_df.groupby('ratio')[metrics].mean()
sns.heatmap(metric_by_ratio.T, cmap='RdYlBu_r', cbar_kws={'label': 'Value'})
plt.title('Collapse Metrics by Ratio')
plt.xlabel('Mixture Ratio (%)')

# 子图4: BRC分布
plt.subplot(2, 2, 4)
for ratio in [0, 5, 10, 15, 20]:
    data = stable_df[stable_df['ratio'] == ratio]['BRC_star_adapted']
    plt.hist(data, bins=30, alpha=0.5, label=f'{ratio}%', density=True)
plt.xlabel('BRC_star_adapted')
plt.ylabel('Density')
plt.title('BRC* Distribution by Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('collapse_metrics_analysis.png', dpi=300)
plt.show()

# 8. 输出关键统计
print("\n=== 按混合比例的平均BRC* (iter≥10000) ===")
print("Ratio | Avg BRC* | Std BRC* | Success Rate")
print("-" * 45)
for ratio in sorted(stable_df['ratio'].unique()):
    ratio_data = stable_df[stable_df['ratio'] == ratio]
    avg_brc = ratio_data['BRC_star_adapted'].mean()
    std_brc = ratio_data['BRC_star_adapted'].std()
    success_rate = ratio_data['success'].mean()
    print(f"{ratio:5.0f} | {avg_brc:8.3f} | {std_brc:8.3f} | {success_rate:12.1%}")

# 9. 混合比例的预测能力
print("\n=== 混合比例作为预测变量 ===")
from sklearn.linear_model import LogisticRegression

# 使用混合比例预测成功
X_ratio = stable_df[['ratio']]
y = stable_df['success']
lr_ratio = LogisticRegression()
lr_ratio.fit(X_ratio, y)
ratio_auc = roc_auc_score(y, lr_ratio.predict_proba(X_ratio)[:, 1])

# 使用BRC预测成功
X_brc = stable_df[['BRC_star_adapted']]
lr_brc = LogisticRegression()
lr_brc.fit(X_brc, y)
brc_auc = roc_auc_score(y, lr_brc.predict_proba(X_brc)[:, 1])

print(f"Ratio-based AUC: {ratio_auc:.3f}")
print(f"BRC-based AUC: {brc_auc:.3f}")