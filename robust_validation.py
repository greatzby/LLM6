import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from tqdm import tqdm

print("=== 交互模型稳健性验证 ===\n")

# 加载数据
df = pd.read_csv('collapse_metrics_updated.csv')
success_df = pd.read_csv('success_log.csv')
merged = pd.merge(df, success_df, on=['ratio', 'seed', 'iter'])
stable = merged[merged['iter'] >= 10000].copy()

# 准备数据
stable['ratio_er_interaction'] = stable['ratio'] * stable['effective_rank']
y = (stable['success'] >= 0.8).astype(int)

# 准备三种模型的特征
X_ratio_only = stable[['ratio']].values
X_er_only = stable[['effective_rank']].values
X_combined = stable[['ratio', 'effective_rank', 'ratio_er_interaction']].values

# 标准化
scaler_ratio = StandardScaler()
scaler_er = StandardScaler()
scaler_combined = StandardScaler()

X_ratio_scaled = scaler_ratio.fit_transform(X_ratio_only)
X_er_scaled = scaler_er.fit_transform(X_er_only)
X_combined_scaled = scaler_combined.fit_transform(X_combined)

# 1. 5折交叉验证
print("1. 5折交叉验证")
print("-" * 50)

models = {
    'Ratio Only': (X_ratio_scaled, LogisticRegression(random_state=42)),
    'ER Only': (X_er_scaled, LogisticRegression(random_state=42)),
    'Ratio + ER + Interaction': (X_combined_scaled, LogisticRegression(random_state=42))
}

cv_results = {}
for name, (X, model) in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    cv_results[name] = scores
    print(f"{name:25s}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")

# 2. Bootstrap验证（1000次）
print("\n2. Bootstrap验证 (1000次迭代)")
print("-" * 50)

n_bootstrap = 1000
bootstrap_results = {name: [] for name in models.keys()}

for i in tqdm(range(n_bootstrap), desc="Bootstrap进度"):
    # 生成bootstrap样本
    idx = np.random.choice(len(y), len(y), replace=True)
    
    for name, (X, model) in models.items():
        X_boot = X[idx]
        y_boot = y.iloc[idx]
        
        # 训练测试分割
        train_size = int(0.8 * len(y_boot))
        train_idx = np.random.choice(len(y_boot), train_size, replace=False)
        test_idx = np.setdiff1d(np.arange(len(y_boot)), train_idx)
        
        X_train, X_test = X_boot[train_idx], X_boot[test_idx]
        y_train, y_test = y_boot.iloc[train_idx], y_boot.iloc[test_idx]
        
        # 训练和评估
        model_copy = LogisticRegression(random_state=42)
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_pred)
            bootstrap_results[name].append(auc)
        except:
            pass  # 跳过无法计算AUC的情况

# 计算bootstrap统计
print("\nBootstrap结果:")
for name, aucs in bootstrap_results.items():
    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    print(f"{name:25s}: AUC = {mean_auc:.3f} ± {std_auc:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 3. 可视化Bootstrap分布
plt.figure(figsize=(12, 8))

# 3.1 Bootstrap AUC分布
plt.subplot(2, 2, 1)
for name, aucs in bootstrap_results.items():
    plt.hist(aucs, bins=50, alpha=0.6, label=name, density=True)
plt.xlabel('AUC', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Bootstrap AUC Distributions (1000 iterations)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 3.2 箱线图比较
plt.subplot(2, 2, 2)
data_for_box = [bootstrap_results[name] for name in models.keys()]
plt.boxplot(data_for_box, labels=['Ratio\nOnly', 'ER\nOnly', 'Combined'])
plt.ylabel('AUC', fontsize=12)
plt.title('Model Comparison (Bootstrap)', fontsize=14)
plt.grid(True, alpha=0.3)

# 3.3 ROC曲线（使用全数据）
plt.subplot(2, 2, 3)
for name, (X, model) in models.items():
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (Full Data)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 3.4 特征重要性（仅限组合模型）
plt.subplot(2, 2, 4)
combined_model = LogisticRegression(random_state=42)
combined_model.fit(X_combined_scaled, y)
feature_names = ['Ratio', 'Effective Rank', 'Ratio × ER']
coefs = combined_model.coef_[0]
colors = ['red' if c < 0 else 'green' for c in coefs]

plt.barh(feature_names, np.abs(coefs), color=colors, alpha=0.7)
plt.xlabel('|Coefficient| (Standardized)', fontsize=12)
plt.title('Feature Importance in Combined Model', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('robustness_validation.png', dpi=300, bbox_inches='tight')
print("\n已保存: robustness_validation.png")

# 4. 按Ratio分组的稳健性检验
print("\n3. 按Ratio分组的稳健性")
print("-" * 50)

ratio_groups = {
    'Low (0-5%)': stable[stable['ratio'] <= 5],
    'Medium (6-11%)': stable[(stable['ratio'] > 5) & (stable['ratio'] <= 11)],
    'High (12%+)': stable[stable['ratio'] >= 12]
}

for group_name, group_data in ratio_groups.items():
    if len(group_data) > 50:  # 确保有足够样本
        X_group = group_data[['ratio', 'effective_rank']].values
        X_group = np.column_stack([X_group, X_group[:, 0] * X_group[:, 1]])  # 添加交互项
        X_group_scaled = StandardScaler().fit_transform(X_group)
        y_group = (group_data['success'] >= 0.8).astype(int)
        
        if len(np.unique(y_group)) > 1:  # 确保有两类
            lr = LogisticRegression(random_state=42)
            try:
                scores = cross_val_score(lr, X_group_scaled, y_group, cv=3, scoring='roc_auc')
                print(f"{group_name:20s}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")
            except:
                print(f"{group_name:20s}: 无法计算（样本不足或单一类别）")

# 5. 保存详细结果
results_summary = {
    'Model': list(models.keys()),
    'CV_AUC_Mean': [cv_results[name].mean() for name in models.keys()],
    'CV_AUC_Std': [cv_results[name].std() for name in models.keys()],
    'Bootstrap_AUC_Mean': [np.mean(bootstrap_results[name]) for name in models.keys()],
    'Bootstrap_AUC_Std': [np.std(bootstrap_results[name]) for name in models.keys()],
    'Bootstrap_CI_Lower': [np.percentile(bootstrap_results[name], 2.5) for name in models.keys()],
    'Bootstrap_CI_Upper': [np.percentile(bootstrap_results[name], 97.5) for name in models.keys()]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('robustness_results.csv', index=False)
print("\n已保存: robustness_results.csv")

print("\n=== 稳健性验证完成 ===")