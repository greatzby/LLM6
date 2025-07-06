# enhanced_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
import seaborn as sns

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])
stable = merged[merged['iter'] >= 10000]

# 1. 对比所有指标
print("=== 指标对比分析 ===")
metrics = ['effective_rank', 'WRC_star', 'BRC_star_adapted', 
           'direction_diversity', 'sigma_weight', 'collapse_score']

results = []
for metric in metrics:
    if metric in stable.columns:
        # 相关性
        corr = stable[metric].corr(stable['success'])
        
        # 对于success二分类的AUC
        y_binary = (stable['success'] >= 0.8).astype(int)
        try:
            if metric == 'effective_rank':
                auc = roc_auc_score(y_binary, stable[metric])
            else:
                auc = roc_auc_score(y_binary, -stable[metric])
        except:
            auc = np.nan
            
        # Spearman相关（捕捉非线性）
        spearman = stats.spearmanr(stable[metric], stable['success'])[0]
        
        results.append({
            'Metric': metric,
            'Pearson_r': f"{corr:.3f}",
            'Spearman_r': f"{spearman:.3f}",
            'AUC': f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 2. 结合ratio的二元预测模型
print("\n=== 组合模型分析 ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 准备特征
X_single = stable[['effective_rank']].values
X_combined = stable[['effective_rank', 'ratio']].values
y = (stable['success'] >= 0.8).astype(int)

# 标准化
scaler = StandardScaler()
X_single_scaled = scaler.fit_transform(X_single)
X_combined_scaled = scaler.fit_transform(X_combined)

# 交叉验证
lr_single = LogisticRegression(random_state=42)
lr_combined = LogisticRegression(random_state=42)

scores_single = cross_val_score(lr_single, X_single_scaled, y, cv=5, scoring='roc_auc')
scores_combined = cross_val_score(lr_combined, X_combined_scaled, y, cv=5, scoring='roc_auc')

print(f"仅Effective Rank: AUC = {scores_single.mean():.3f} ± {scores_single.std():.3f}")
print(f"ER + Ratio组合: AUC = {scores_combined.mean():.3f} ± {scores_combined.std():.3f}")

# 3. 可视化增强
plt.figure(figsize=(15, 10))

# 3.1 按ratio分组的ER分布（小提琴图）
plt.subplot(2, 3, 1)
ratio_groups = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
plot_data = []
plot_ratios = []
for r in ratio_groups:
    if r in stable['ratio'].values:
        data = stable[stable['ratio'] == r]['effective_rank'].values
        plot_data.extend(data)
        plot_ratios.extend([r] * len(data))

plot_df = pd.DataFrame({'Ratio': plot_ratios, 'Effective Rank': plot_data})
sns.violinplot(data=plot_df, x='Ratio', y='Effective Rank')
plt.axhline(66.3, color='red', linestyle='--', alpha=0.7, label='Threshold=66.3')
plt.xticks(rotation=45)
plt.title('Effective Rank Distribution by Ratio')
plt.legend()

# 3.2 ER vs Success的密度图
plt.subplot(2, 3, 2)
hexbin = plt.hexbin(stable['effective_rank'], stable['success'], 
                    gridsize=30, cmap='YlOrRd', mincnt=1)
plt.axvline(66.3, color='red', linestyle='--', alpha=0.7)
plt.axhline(0.8, color='orange', linestyle='--', alpha=0.7)
plt.xlabel('Effective Rank')
plt.ylabel('Success Rate')
plt.title('Density Plot: ER vs Success')
plt.colorbar(hexbin, label='Count')

# 3.3 条件概率：P(Success≥0.8 | ER)
plt.subplot(2, 3, 3)
er_bins = np.linspace(stable['effective_rank'].min(), stable['effective_rank'].max(), 20)
bin_centers = (er_bins[:-1] + er_bins[1:]) / 2
success_probs = []

for i in range(len(er_bins)-1):
    mask = (stable['effective_rank'] >= er_bins[i]) & (stable['effective_rank'] < er_bins[i+1])
    if mask.sum() > 10:  # 至少10个样本
        prob = (stable[mask]['success'] >= 0.8).mean()
        success_probs.append(prob)
    else:
        success_probs.append(np.nan)

plt.plot(bin_centers, success_probs, 'o-', linewidth=2, markersize=8)
plt.axvline(66.3, color='red', linestyle='--', alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Effective Rank')
plt.ylabel('P(Success ≥ 0.8)')
plt.title('Success Probability by ER')
plt.ylim(0, 1)

# 3.4 低混合比例区域的细节
plt.subplot(2, 3, 4)
low_ratio = stable[stable['ratio'] <= 6]
scatter = plt.scatter(low_ratio['effective_rank'], low_ratio['success'], 
                     c=low_ratio['ratio'], cmap='viridis', alpha=0.6, s=20)
plt.axvline(66.3, color='red', linestyle='--', alpha=0.7)
plt.axhline(0.8, color='orange', linestyle='--', alpha=0.7)
plt.xlabel('Effective Rank')
plt.ylabel('Success Rate')
plt.title('Low Ratio Region (≤6%)')
plt.colorbar(scatter, label='Ratio (%)')  # 直接用scatter对象

# 3.5 时间演化：选几个代表性案例
plt.subplot(2, 3, 5)
# 选择ratio=3的几个种子
example_data = merged[(merged['ratio'] == 3) & (merged['seed'].isin([42, 43, 44]))]
for seed in [42, 43, 44]:
    seed_data = example_data[example_data['seed'] == seed].sort_values('iter')
    if len(seed_data) > 0:
        plt.plot(seed_data['iter'], seed_data['effective_rank'], 
                alpha=0.7, label=f'Seed {seed}')
plt.axhline(66.3, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Effective Rank')
plt.title('ER Evolution (Ratio=3%)')
plt.legend()

# 3.6 指标对比雷达图
plt.subplot(2, 3, 6, projection='polar')
metrics_for_radar = ['effective_rank', 'WRC_star', 'direction_diversity', 
                     'sigma_weight', 'collapse_score']
correlations = []
aucs = []

for metric in metrics_for_radar:
    if metric in stable.columns:
        corr = abs(stable[metric].corr(stable['success']))
        correlations.append(corr)
        
        y_binary = (stable['success'] >= 0.8).astype(int)
        try:
            if metric == 'effective_rank':
                auc = roc_auc_score(y_binary, stable[metric])
            else:
                auc = roc_auc_score(y_binary, -stable[metric])
            aucs.append(auc)
        except:
            aucs.append(0.5)

angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False)
correlations = correlations + correlations[:1]
aucs = aucs + aucs[:1]
angles = np.concatenate([angles, [angles[0]]])

plt.plot(angles, correlations, 'o-', linewidth=2, label='|Correlation|')
plt.plot(angles, aucs, 's-', linewidth=2, label='AUC')
plt.fill(angles, correlations, alpha=0.25)
plt.fill(angles, aucs, alpha=0.25)
plt.xticks(angles[:-1], metrics_for_radar, size=8)
plt.ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.title('Metric Comparison')

plt.tight_layout()
plt.savefig('enhanced_er_analysis.png', dpi=300, bbox_inches='tight')
print("\n已保存: enhanced_er_analysis.png")

# 4. 生成LaTeX表格
print("\n=== LaTeX表格代码 ===")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Effective Rank by Mixture Ratio}")
print("\\begin{tabular}{ccc}")
print("\\hline")
print("Ratio (\\%) & Mean ER & Success $\\geq$ 0.8 (\\%) \\\\")
print("\\hline")
for ratio in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
    ratio_data = stable[stable['ratio'] == ratio]
    if len(ratio_data) > 0:
        mean_er = ratio_data['effective_rank'].mean()
        success_rate = (ratio_data['success'] >= 0.8).mean() * 100
        print(f"{ratio} & {mean_er:.1f} & {success_rate:.1f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")