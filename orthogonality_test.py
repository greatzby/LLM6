import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def partial_correlation(data, x, y, z):
    """计算x和y的偏相关，控制z"""
    from sklearn.linear_model import LinearRegression
    reg_xz = LinearRegression().fit(data[[z]], data[x])
    reg_yz = LinearRegression().fit(data[[z]], data[y])
    
    residual_x = data[x] - reg_xz.predict(data[[z]])
    residual_y = data[y] - reg_yz.predict(data[[z]])
    
    return np.corrcoef(residual_x, residual_y)[0, 1]

print("=== ER-Ratio正交性验证 ===\n")

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])

# 创建不同条件的子集
subsets = {
    'Different Seeds': [],
    'Different Iters': [],
    'Different Architectures': []  # 如果有的话
}

# 1. 相同ratio，不同seed的ER变化
print("1. 相同Ratio下的ER变异性")
print("-" * 50)

for ratio in [5, 10, 15]:
    ratio_data = merged[(merged['ratio'] == ratio) & (merged['iter'] >= 30000)]
    if len(ratio_data) > 0:
        er_by_seed = ratio_data.groupby('seed')['effective_rank'].mean()
        er_range = er_by_seed.max() - er_by_seed.min()
        er_cv = er_by_seed.std() / er_by_seed.mean()
        
        print(f"Ratio={ratio:2d}%: ER范围={er_range:.2f}, CV={er_cv:.3f}")
        
        # 这表明即使ratio相同，ER也有变化

# 2. 正交分解
print("\n2. 主成分分析")
print("-" * 50)

# 准备特征矩阵
features = merged[['ratio', 'effective_rank', 'iter']].copy()
features['ratio_er'] = features['ratio'] * features['effective_rank']
features['ratio_iter'] = features['ratio'] * features['iter']
features['er_iter'] = features['effective_rank'] * features['iter']

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA
pca = PCA()
pca.fit(features_scaled)

# 打印方差解释
print("各主成分解释方差比例:")
for i, var_exp in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_exp:.3f}")

# 如果前两个主成分解释了大部分方差，且ratio和ER在不同主成分上
# 说明它们是相对独立的

# 3. 条件独立性可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 3.1 固定ratio看ER分布
ax = axes[0, 0]
for ratio in [5, 10, 15]:
    ratio_data = merged[(merged['ratio'] == ratio) & (merged['iter'] >= 30000)]
    if len(ratio_data) > 20:
        ax.hist(ratio_data['effective_rank'], bins=20, alpha=0.5, 
                label=f'Ratio={ratio}%', density=True)

ax.set_xlabel('Effective Rank')
ax.set_ylabel('Density')
ax.set_title('ER Distribution at Fixed Ratios')
ax.legend()

# 3.2 固定ER范围看ratio效果
ax = axes[0, 1]
er_bins = pd.qcut(merged['effective_rank'], q=4)
merged['er_quartile'] = er_bins

for i, (q_name, q_label) in enumerate(zip(['Q1', 'Q2', 'Q3', 'Q4'], 
                                          ['Low', 'Mid-Low', 'Mid-High', 'High'])):
    quartile_data = merged[merged['er_quartile'].cat.codes == i]
    ratio_success = quartile_data.groupby('ratio')['success'].mean()
    
    if len(ratio_success) > 3:
        ax.plot(ratio_success.index, ratio_success.values, 
                'o-', label=f'ER {q_label}', alpha=0.7)

ax.set_xlabel('Mixture Ratio (%)')
ax.set_ylabel('Success Rate')
ax.set_title('Ratio Effect at Different ER Levels')
ax.legend()
ax.grid(True, alpha=0.3)

# 3.3 相互信息分析
ax = axes[1, 0]
from sklearn.feature_selection import mutual_info_regression

# 计算相互信息
mi_ratio = mutual_info_regression(merged[['ratio']], merged['success'], 
                                  random_state=42)[0]
mi_er = mutual_info_regression(merged[['effective_rank']], merged['success'], 
                              random_state=42)[0]
mi_both = mutual_info_regression(merged[['ratio', 'effective_rank']], 
                                merged['success'], random_state=42)

bars = ax.bar(['Ratio', 'ER', 'Ratio+ER\n(sum)'], 
               [mi_ratio, mi_er, mi_ratio + mi_er], 
               alpha=0.7, color=['blue', 'green', 'gray'])
ax.bar(3, mi_both.sum(), alpha=0.7, color='red', label='Actual')

ax.set_ylabel('Mutual Information with Success')
ax.set_title('Information Content Analysis')
ax.legend()

# 如果实际MI > 单独MI之和，说明有协同效应

# 3.4 残差分析
ax = axes[1, 1]

# 用ratio预测ER的残差
from sklearn.linear_model import LinearRegression
reg_ratio_er = LinearRegression().fit(merged[['ratio']], merged['effective_rank'])
er_residuals = merged['effective_rank'] - reg_ratio_er.predict(merged[['ratio']])

# 残差与success的关系
residual_bins = pd.qcut(er_residuals, q=5)
residual_success = merged.groupby(residual_bins)['success'].mean()

ax.plot(range(len(residual_success)), residual_success.values, 'o-', markersize=10)
ax.set_xlabel('ER Residual Quintile')
ax.set_ylabel('Success Rate')
ax.set_title('Success vs ER Residuals (after controlling Ratio)')
ax.grid(True, alpha=0.3)

# 如果残差仍影响success，说明ER有独立效应

plt.tight_layout()
plt.savefig('orthogonality_analysis.png', dpi=300)
print("\n已保存: orthogonality_analysis.png")

analysis_data = merged[['ratio', 'iter', 'effective_rank']].copy()
scaler = StandardScaler()
analysis_data[['ratio_scaled', 'iter_scaled', 'er_scaled']] = scaler.fit_transform(
    analysis_data[['ratio', 'iter', 'effective_rank']]
)

partial_corr = partial_correlation(analysis_data, 'ratio_scaled', 'er_scaled', 'iter_scaled')
print(f"\n偏相关性 (控制iteration): {partial_corr:.3f}")

# 最终判断
print("\n=== 独立性判断 ===")
if abs(partial_corr) < 0.3:
    print("✓ ER和Ratio基本独立")
elif abs(partial_corr) < 0.7:
    print("⚠ ER和Ratio有中度相关，但仍有独立变化")
else:
    print("✗ ER和Ratio高度相关，交互项可能不可靠")