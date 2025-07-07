import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

print("=== ER与Ratio关系深度分析 ===\n")

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])

# 1. ER随时间变化轨迹
print("1. ER随训练进程的变化")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1.1 不同ratio下ER的时间轨迹
ax = axes[0, 0]
for ratio in sorted(merged['ratio'].unique()):
    if ratio in [0, 5, 10, 15, 20]:  # 选择关键比例
        ratio_data = merged[merged['ratio'] == ratio]
        er_mean = ratio_data.groupby('iter')['effective_rank'].mean()
        er_std = ratio_data.groupby('iter')['effective_rank'].std()
        
        ax.plot(er_mean.index, er_mean.values, label=f'{ratio}%', linewidth=2)
        ax.fill_between(er_mean.index, 
                        er_mean.values - er_std.values,
                        er_mean.values + er_std.values, 
                        alpha=0.2)

ax.set_xlabel('Iteration')
ax.set_ylabel('Effective Rank')
ax.set_title('ER Trajectories by Mixture Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# 1.2 ER下降速率分析
ax = axes[0, 1]
er_drop_rates = {}

for ratio in sorted(merged['ratio'].unique()):
    ratio_data = merged[merged['ratio'] == ratio]
    er_by_iter = ratio_data.groupby('iter')['effective_rank'].mean()
    
    if len(er_by_iter) > 1:
        # 计算线性下降率
        X = er_by_iter.index.values.reshape(-1, 1)
        y = er_by_iter.values
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        er_drop_rates[ratio] = -slope  # 负斜率表示下降

# 绘制下降率
ratios = list(er_drop_rates.keys())
rates = list(er_drop_rates.values())
ax.bar(ratios, rates, alpha=0.7, color='coral')
ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('ER Decrease Rate')
ax.set_title('ER Decline Rate by Ratio')
ax.grid(True, alpha=0.3, axis='y')

# 1.3 ER的长期稳定值（>30k iterations）
ax = axes[1, 0]
stable_data = merged[merged['iter'] >= 30000]

# 计算每个ratio的ER统计
er_stats = stable_data.groupby('ratio')['effective_rank'].agg(['mean', 'std', 'count'])
er_stats = er_stats[er_stats['count'] >= 10]  # 确保足够样本

# 绘制ER均值和误差棒
ax.errorbar(er_stats.index, er_stats['mean'], yerr=er_stats['std'], 
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('Stable ER (>30k iter)')
ax.set_title('Long-term ER Values by Ratio')
ax.grid(True, alpha=0.3)

# 添加相关性信息
corr = np.corrcoef(er_stats.index, er_stats['mean'])[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

# 1.4 ER变异系数分析
ax = axes[1, 1]
cv_data = []

for ratio in sorted(merged['ratio'].unique()):
    ratio_data = merged[merged['ratio'] == ratio]
    if len(ratio_data) >= 20:
        # 计算不同时间段的CV
        for iter_range, label in [((0, 10000), 'Early'), 
                                  ((10000, 30000), 'Mid'), 
                                  ((30000, 100000), 'Late')]:
            subset = ratio_data[(ratio_data['iter'] >= iter_range[0]) & 
                               (ratio_data['iter'] < iter_range[1])]
            if len(subset) > 0:
                cv = subset['effective_rank'].std() / subset['effective_rank'].mean()
                cv_data.append({'ratio': ratio, 'phase': label, 'cv': cv})

cv_df = pd.DataFrame(cv_data)
cv_pivot = cv_df.pivot(index='ratio', columns='phase', values='cv')

# 热力图显示CV
sns.heatmap(cv_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
ax.set_title('ER Coefficient of Variation by Ratio and Phase')
ax.set_xlabel('Training Phase')
ax.set_ylabel('Mixture Ratio (%)')

plt.tight_layout()
plt.savefig('er_ratio_relationship.png', dpi=300, bbox_inches='tight')
print("已保存: er_ratio_relationship.png")

# 2. 统计独立性检验
print("\n2. ER-Ratio独立性统计检验")
print("-" * 50)

# 2.1 整体相关性
overall_corr = merged.groupby(['ratio', 'iter'])['effective_rank'].mean().reset_index()
correlation = overall_corr.groupby('ratio')['effective_rank'].mean().corr(
    overall_corr.groupby('ratio')['effective_rank'].mean().index
)
print(f"整体相关性 (Pearson): {correlation:.3f}")

# 2.2 偏相关分析（控制iteration）
from sklearn.preprocessing import StandardScaler

# 准备数据
analysis_data = merged[['ratio', 'iter', 'effective_rank', 'success']].copy()
scaler = StandardScaler()
analysis_data[['ratio_scaled', 'iter_scaled', 'er_scaled']] = scaler.fit_transform(
    analysis_data[['ratio', 'iter', 'effective_rank']]
)

# 计算偏相关（控制iteration）
def partial_correlation(data, x, y, z):
    """计算x和y的偏相关，控制z"""
    # 残差法
    reg_xz = LinearRegression().fit(data[[z]], data[x])
    reg_yz = LinearRegression().fit(data[[z]], data[y])
    
    residual_x = data[x] - reg_xz.predict(data[[z]])
    residual_y = data[y] - reg_yz.predict(data[[z]])
    
    return np.corrcoef(residual_x, residual_y)[0, 1]

partial_corr = partial_correlation(analysis_data, 'ratio_scaled', 'er_scaled', 'iter_scaled')
print(f"偏相关性 (控制iteration): {partial_corr:.3f}")

# 2.3 条件独立性检验
print("\n条件独立性检验:")

# 在固定ER范围内，ratio是否仍影响success？
er_bins = pd.qcut(merged['effective_rank'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
merged['er_bin'] = er_bins

for er_level in ['Low', 'Medium', 'High']:
    subset = merged[merged['er_bin'] == er_level]
    if len(subset) > 50:
        # 计算ratio和success的相关性
        ratio_success_corr = subset.groupby('ratio')['success'].mean().corr(
            subset.groupby('ratio')['success'].mean().index
        )
        print(f"  ER={er_level:10s}: Ratio-Success相关性 = {ratio_success_corr:.3f}")

# 2.4 方差分解
print("\n方差分解分析:")
from sklearn.metrics import r2_score

# 仅Ratio模型
X_ratio = merged[['ratio']]
y_success = merged['success']
model_ratio = LinearRegression().fit(X_ratio, y_success)
r2_ratio = r2_score(y_success, model_ratio.predict(X_ratio))

# 仅ER模型
X_er = merged[['effective_rank']]
model_er = LinearRegression().fit(X_er, y_success)
r2_er = r2_score(y_success, model_er.predict(X_er))

# 组合模型
X_both = merged[['ratio', 'effective_rank']]
model_both = LinearRegression().fit(X_both, y_success)
r2_both = r2_score(y_success, model_both.predict(X_both))

# 交互模型
merged['ratio_er'] = merged['ratio'] * merged['effective_rank']
X_interaction = merged[['ratio', 'effective_rank', 'ratio_er']]
model_interaction = LinearRegression().fit(X_interaction, y_success)
r2_interaction = r2_score(y_success, model_interaction.predict(X_interaction))

print(f"R² (仅Ratio):     {r2_ratio:.3f}")
print(f"R² (仅ER):        {r2_er:.3f}")
print(f"R² (Ratio+ER):    {r2_both:.3f}")
print(f"R² (含交互项):     {r2_interaction:.3f}")
print(f"交互项贡献:        {r2_interaction - r2_both:.3f}")

# 3. 保存详细分析结果
analysis_results = {
    'overall_correlation': correlation,
    'partial_correlation': partial_corr,
    'r2_ratio_only': r2_ratio,
    'r2_er_only': r2_er,
    'r2_both': r2_both,
    'r2_interaction': r2_interaction,
    'interaction_contribution': r2_interaction - r2_both
}

# 保存ER统计
er_stats.to_csv('er_statistics_by_ratio.csv')
print("\n已保存: er_statistics_by_ratio.csv")

print("\n=== 分析完成 ===")