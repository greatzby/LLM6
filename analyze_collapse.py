#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_collapse.py  ───────────────────────────────────────────────
读取 collapse_metrics.csv (+ 可选 perf.csv) 产出一份快速报告：
  1. 描述性统计、指标分布、相关系数热图
  2. (可选) 预测成功率：
       • 逻辑回归 / XGBoost (若已安装) / 决策树
       • ROC / PR 曲线 + AUC
  3. 把所有结果写至 report/ 目录
--------------------------------------------------------------------
用法
  python analyze_collapse.py                         # 只有指标
  python analyze_collapse.py -p perf.csv             # 带表现文件
  python analyze_collapse.py -m collapse_metrics.csv -p perf.csv
依赖
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
--------------------------------------------------------------------
"""

import argparse, os, json, warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------- CLI ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv",
                    help="collapse_metrics.csv path")
parser.add_argument("-p", "--perf",   default=None,
                    help="performance file path (optional)")
parser.add_argument("-o", "--outdir", default="report",
                    help="output directory")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --------------------- 1  读取指标 --------------------------
df = pd.read_csv(args.metric)
print("Read metrics:", df.shape)

# ▽ 保存基本信息
with open(os.path.join(args.outdir, "shape.json"), "w") as w:
    json.dump(dict(rows=int(df.shape[0]), cols=int(df.shape[1])), w, indent=2)

# --------------------- 2  描述性统计 ------------------------
stats = df.describe().T
stats.to_csv(os.path.join(args.outdir, "stats.csv"))
print("Saved stats.csv")

# 指标分布图
plt.figure(figsize=(12, 6))
for i, col in enumerate([c for c in df.columns if df[c].dtype != object][:8]):
    plt.subplot(2, 4, i + 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "histograms.png"), dpi=300)
plt.close()

# 相关系数热图
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Pearson Correlation")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "corr_heatmap.png"), dpi=300)
plt.close()
print("Saved histograms.png & corr_heatmap.png")

# --------------------- 3  若有 perf 做预测 ------------------
summary_lines = []
if args.perf and os.path.exists(args.perf):
    perf = pd.read_csv(args.perf)
    print("Read perf:", perf.shape)

    # merge
    data = df.merge(perf, on=["ratio", "seed", "iter"], how="inner")
    print("Merged:", data.shape)

    # label 处理
    if data["success"].dtype != int and data["success"].max() <= 1.0:
        data["label"] = (data["success"] >= 0.8).astype(int)
    else:
        data["label"] = data["success"].astype(int)

    feature_cols = ['effective_rank', 'row_norm_std', 'WRC_star',
                    'direction_diversity', 'token_spread']  # 可自行增删
    X = data[feature_cols].fillna(0).values
    y = data["label"].values

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # ---------- Logistic Regression ----------
    clf = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    auc_lr = roc_auc_score(y_test, prob)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"LR AUC={auc_lr:.3f}")
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc_lr.png"), dpi=300)
    plt.close()

    # ---------- Decision Tree ----------
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                random_state=0).fit(X_train, y_train)
    prob_dt = dt.predict_proba(X_test)[:, 1]
    auc_dt = roc_auc_score(y_test, prob_dt)

    fpr, tpr, _ = roc_curve(y_test, prob_dt)
    plt.figure()
    plt.plot(fpr, tpr, label=f"DT AUC={auc_dt:.3f}", c="darkorange")
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc_dt.png"), dpi=300)
    plt.close()

    # 导出树结构 PNG
    plt.figure(figsize=(10, 6))
    plot_tree(dt, feature_names=feature_cols, class_names=["fail", "succ"],
              filled=True, rounded=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "decision_tree.png"), dpi=300)
    plt.close()

    # ---------- 文本总结 ----------
    summary_lines.append(f"LogReg  AUC = {auc_lr:.4f}")
    summary_lines.append(f"DecTree AUC = {auc_dt:.4f}")
    coef = dict(zip(feature_cols, clf.coef_[0]))
    summary_lines.append("LogReg coef (standardized):")
    summary_lines += [f"  {k}: {v:+.3f}" for k, v in coef.items()]

else:
    summary_lines.append("Performance file not provided → 只生成描述性统计")

# --------------------- 4  写入 summary.txt -----------------
with open(os.path.join(args.outdir, "summary.txt"), "w") as w:
    w.write("\n".join(summary_lines))
print("All done. See report/ directory.")