#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-p", "--perf",   default="success_log.csv")
parser.add_argument("--thr", type=float, default=0.6,   # 建议 0.8
                    help="success <= thr → fail")
args = parser.parse_args()

# ------------- 读数据
df = pd.read_csv(args.metric).merge(
        pd.read_csv(args.perf), on=["ratio", "seed", "iter"])

X = df[["WRC_star", "effective_rank"]].values          # ρ, d
y = (df["success"] <= args.thr).astype(int).values      # 1 = fail
groups = (df["ratio"].astype(str) + "-" + df["seed"].astype(str)).values

logo = LeaveOneGroupOut()
aucs, fold_names = [], []
for train_idx, test_idx in logo.split(X, y, groups):
    y_test = y[test_idx]
    if y_test.min() == y_test.max():          # 只有一个类别 → 跳过
        fold_names.append(groups[test_idx][0])
        aucs.append(np.nan)
        continue
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, y[train_idx])
    prob = clf.predict_proba(Xte)[:, 1]
    aucs.append(roc_auc_score(y_test, prob))
    fold_names.append(groups[test_idx][0])

# 打印结果
valid = [a for a in aucs if not np.isnan(a)]
print("Per-trajectory AUC")
for name, a in zip(fold_names, aucs):
    print(f"  {name:>10}:  {'-' if np.isnan(a) else f'{a:.4f}'}")

print(f"\nmean AUC (valid folds) = {np.mean(valid):.4f}")