# verify_rank_hypothesis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])

# 只看稳定数据
stable = merged[merged['iter'] >= 10000]

# 1. 验证effective_rank与success的相关性
print("=== Effective Rank分析 ===")
corr = stable['effective_rank'].corr(stable['success'])
print(f"Pearson相关系数: {corr:.3f}")

# 2. 找出最佳阈值
# 用逻辑回归自动找阈值
X = stable[['effective_rank']].values
y = (stable['success'] >= 0.8).astype(int)

lr = LogisticRegression()
lr.fit(X, y)

# 计算决策边界
threshold = -lr.intercept_[0] / lr.coef_[0, 0]
print(f"\n最佳Effective Rank阈值: {threshold:.1f}")

# 3. 评估阈值性能
y_pred = (stable['effective_rank'] >= threshold).astype(int)
print("\n分类报告:")
print(classification_report(y, y_pred, target_names=['Low Success', 'High Success']))

# 4. 按ratio分组看effective_rank
print("\n各混合比例的平均Effective Rank:")
rank_by_ratio = stable.groupby('ratio')['effective_rank'].agg(['mean', 'std'])
print(rank_by_ratio)

# 5. 绘图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
scatter = plt.scatter(stable['effective_rank'], stable['success'], 
                     c=stable['ratio'], cmap='viridis', alpha=0.6)
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.1f}')
plt.axhline(0.8, color='orange', linestyle='--', label='Success=0.8')
plt.xlabel('Effective Rank')
plt.ylabel('Success Rate')
plt.title('Effective Rank vs Success')
plt.colorbar(scatter, label='Ratio (%)')
plt.legend()

plt.subplot(1, 3, 2)
ratios = rank_by_ratio.index
means = rank_by_ratio['mean']
stds = rank_by_ratio['std']
plt.errorbar(ratios, means, yerr=stds, marker='o', linewidth=2)
plt.axhline(threshold, color='red', linestyle='--')
plt.xlabel('Mixture Ratio (%)')
plt.ylabel('Effective Rank')
plt.title('Effective Rank by Ratio')

plt.subplot(1, 3, 3)
# ROC曲线
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y, stable['effective_rank'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Effective Rank')
plt.legend()

plt.tight_layout()
plt.savefig('effective_rank_analysis.png', dpi=300)
print("\n已保存: effective_rank_analysis.png")