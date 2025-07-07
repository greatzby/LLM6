# quick_diagnosis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=== 快速ER-Ratio独立性诊断 ===\n")

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])

# 1. 直接相关性
print("1. 相关性分析")
print("-" * 40)

# 计算每个ratio的平均ER（稳定期）
stable = merged[merged['iter'] >= 30000]
ratio_er = stable.groupby('ratio')['effective_rank'].agg(['mean', 'std']).reset_index()

# Pearson相关
corr, p_value = stats.pearsonr(ratio_er['ratio'], ratio_er['mean'])
print(f"Ratio-ER相关系数: {corr:.3f} (p={p_value:.4f})")

# 2. 决定系数
from sklearn.linear_model import LinearRegression
X = ratio_er[['ratio']]
y = ratio_er['mean']
reg = LinearRegression().fit(X, y)
r2 = reg.score(X, y)
print(f"R² (ER由Ratio解释): {r2:.3f}")

# 3. 残差分析
er_predicted = reg.predict(X)
residuals = y - er_predicted
residual_std = residuals.std()
print(f"ER残差标准差: {residual_std:.3f}")

# 4. 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 4.1 Ratio vs ER scatter
ax = axes[0, 0]
ax.scatter(ratio_er['ratio'], ratio_er['mean'], s=100, alpha=0.7)
ax.plot(ratio_er['ratio'], er_predicted, 'r--', label=f'R²={r2:.3f}')
ax.errorbar(ratio_er['ratio'], ratio_er['mean'], yerr=ratio_er['std'], 
            fmt='none', alpha=0.5, capsize=5)
ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('Mean ER (>30k iter)')
ax.set_title('ER vs Ratio Relationship')
ax.legend()
ax.grid(True, alpha=0.3)

# 4.2 残差图
ax = axes[0, 1]
ax.scatter(ratio_er['ratio'], residuals, s=100, alpha=0.7)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('ER Residuals')
ax.set_title('Residual Analysis')
ax.grid(True, alpha=0.3)

# 4.3 在控制ER后，ratio是否仍影响success？
ax = axes[1, 0]
# 将ER分箱
stable['er_bin'] = pd.qcut(stable['effective_rank'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

for er_q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = stable[stable['er_bin'] == er_q]
    if len(subset) > 50:
        ratio_effect = subset.groupby('ratio')['success'].mean()
        ax.plot(ratio_effect.index, ratio_effect.values, 'o-', 
                label=f'ER {er_q}', alpha=0.7)

ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('Success Rate')
ax.set_title('Ratio Effect within ER Quartiles')
ax.legend()
ax.grid(True, alpha=0.3)

# 4.4 交互效应热力图
ax = axes[1, 1]
pivot_table = stable.pivot_table(
    values='success', 
    index=pd.cut(stable['effective_rank'], bins=10),
    columns=pd.cut(stable['ratio'], bins=[0, 2, 5, 8, 11, 15, 20]),
    aggfunc='mean'
)

im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xlabel('Ratio Bins')
ax.set_ylabel('ER Bins')
ax.set_title('Success Rate Heatmap')

# 添加colorbar
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('quick_diagnosis.png', dpi=300)
print("\n已保存: quick_diagnosis.png")

# 5. 最终诊断
print("\n=== 诊断结论 ===")
print("-" * 40)

if r2 > 0.9:
    print("⚠️ 警告：ER几乎完全由Ratio决定 (R²>0.9)")
    print("   → 交互项可能是虚假的")
elif r2 > 0.7:
    print("⚠️ 注意：ER与Ratio高度相关 (R²>0.7)")
    print("   → 需要谨慎解释交互效应")
else:
    print("✓ 良好：ER有独立于Ratio的变化")
    print("   → 交互效应可能是真实的")

print(f"\n关键指标：")
print(f"  - 相关系数: {corr:.3f}")
print(f"  - R²: {r2:.3f}")
print(f"  - 残差标准差: {residual_std:.3f}")

# 6. 建议
if r2 > 0.8:
    print("\n建议的后续实验：")
    print("1. 使用不同的网络架构（改变ER基线）")
    print("2. 添加正则化项来控制ER")
    print("3. 在固定ER的条件下测试ratio效果")