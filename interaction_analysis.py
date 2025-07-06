# interaction_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])
stable = merged[merged['iter'] >= 10000]

# 1. 决策树分析：找出最优规则
print("=== 决策树分析 ===")
X = stable[['ratio', 'effective_rank']]
y = (stable['success'] >= 0.8).astype(int)

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
dt.fit(X, y)

# 可视化决策树
plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=['Ratio (%)', 'Effective Rank'], 
          class_names=['Low Success', 'High Success'],
          filled=True, rounded=True)
plt.title('Decision Tree: Predicting Success ≥ 0.8')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')

# 打印决策规则
print("\n决策规则：")
tree_rules = dt.tree_
def print_rules(node=0, depth=0):
    if tree_rules.feature[node] != -2:  # 非叶节点
        feature = ['Ratio', 'ER'][tree_rules.feature[node]]
        threshold = tree_rules.threshold[node]
        print(f"{'  '*depth}if {feature} <= {threshold:.2f}:")
        print_rules(tree_rules.children_left[node], depth+1)
        print(f"{'  '*depth}else:")
        print_rules(tree_rules.children_right[node], depth+1)
    else:  # 叶节点
        value = tree_rules.value[node][0]
        prob = value[1] / value.sum()
        print(f"{'  '*depth}Success prob = {prob:.2%}")

print_rules()

# 2. 交互效应分析
print("\n=== 交互效应分析 ===")
# 创建交互特征
stable['ratio_er_interaction'] = stable['ratio'] * stable['effective_rank']

# 逻辑回归with交互项
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features = ['ratio', 'effective_rank', 'ratio_er_interaction']
X_interact = scaler.fit_transform(stable[features])
y = (stable['success'] >= 0.8).astype(int)

lr = LogisticRegression()
lr.fit(X_interact, y)

print("逻辑回归系数（标准化后）：")
for feat, coef in zip(features, lr.coef_[0]):
    print(f"  {feat}: {coef:.3f}")

# 3. 2D概率图
print("\n生成2D概率图...")
ratio_range = np.linspace(0, 20, 50)
er_range = np.linspace(64, 72, 50)
ratio_grid, er_grid = np.meshgrid(ratio_range, er_range)

# 预测每个点的成功概率
grid_points = np.c_[ratio_grid.ravel(), er_grid.ravel()]
grid_interact = grid_points[:, 0] * grid_points[:, 1]
grid_features = np.c_[grid_points, grid_interact]
grid_features_scaled = scaler.transform(grid_features)

probs = lr.predict_proba(grid_features_scaled)[:, 1]
prob_grid = probs.reshape(ratio_grid.shape)

# 绘制
plt.figure(figsize=(12, 8))
contour = plt.contourf(ratio_grid, er_grid, prob_grid, levels=20, cmap='RdYlGn')
plt.colorbar(contour, label='P(Success ≥ 0.8)')

# 添加决策边界
contour_lines = plt.contour(ratio_grid, er_grid, prob_grid, 
                           levels=[0.5, 0.8], colors='black', linewidths=2)
plt.clabel(contour_lines, inline=True, fontsize=10)

# 叠加实际数据点
high_success = stable[stable['success'] >= 0.8]
low_success = stable[stable['success'] < 0.8]
plt.scatter(high_success['ratio'], high_success['effective_rank'], 
           c='green', marker='^', alpha=0.3, s=20, label='Success ≥ 0.8')
plt.scatter(low_success['ratio'], low_success['effective_rank'], 
           c='red', marker='v', alpha=0.3, s=20, label='Success < 0.8')

plt.xlabel('Mixture Ratio (%)')
plt.ylabel('Effective Rank')
plt.title('Success Probability Surface')
plt.legend()
plt.savefig('probability_surface.png', dpi=300, bbox_inches='tight')

# 4. 简化的实用规则
print("\n=== 实用决策规则 ===")
# 基于数据找出简单规则
rules = []

# 规则1：高ratio总是安全
high_ratio_success = (stable[stable['ratio'] >= 8]['success'] >= 0.8).mean()
rules.append(f"Rule 1: If ratio ≥ 8%, then success rate = {high_ratio_success:.1%}")

# 规则2：低ratio需要高ER
for r in [0, 1, 2, 3, 4, 5, 6]:
    ratio_data = stable[stable['ratio'] == r]
    if len(ratio_data) > 10:
        # 找出达到80%成功率需要的ER阈值
        success_data = ratio_data[ratio_data['success'] >= 0.8]
        if len(success_data) > 0:
            er_threshold = success_data['effective_rank'].quantile(0.1)  # 10分位数
            rules.append(f"Rule {r+2}: If ratio = {r}%, need ER ≥ {er_threshold:.1f}")

for rule in rules:
    print(rule)

# 5. 生成最终总结表
print("\n=== 论文用总结表 ===")
summary_data = []
for ratio in sorted(stable['ratio'].unique()):
    ratio_data = stable[stable['ratio'] == ratio]
    summary_data.append({
        'Ratio (%)': ratio,
        'Mean ER': ratio_data['effective_rank'].mean(),
        'Std ER': ratio_data['effective_rank'].std(),
        'Success Rate': ratio_data['success'].mean(),
        'P(Success≥0.8)': (ratio_data['success'] >= 0.8).mean(),
        'Safe (>90%)': 'Yes' if (ratio_data['success'] >= 0.8).mean() > 0.9 else 'No'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False, float_format='%.2f'))

# 保存为CSV
summary_df.to_csv('final_summary_table.csv', index=False)
print("\n已保存: final_summary_table.csv")