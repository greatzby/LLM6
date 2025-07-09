#!/usr/bin/env python3
# static_rho_d_logit_group.py
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-p", "--perf",   default="success_log.csv")
parser.add_argument("--thr", type=float, default=0.5,
                    help="success ≤ thr → fail")
args = parser.parse_args()

# ---------- 读取并合并
met  = pd.read_csv(args.metric)
perf = pd.read_csv(args.perf)
df   = met.merge(perf, on=["ratio", "seed", "iter"])

X = df[["WRC_star", "effective_rank"]].values   # ρ, d
y = (df["success"] <= args.thr).astype(int).values
groups = (df["ratio"].astype(str) + "-" + df["seed"].astype(str)).values

# ---------- 5-fold group CV
cv   = GroupKFold(n_splits=5)
aucs = []
for train_idx, test_idx in cv.split(X, y, groups):
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, y[train_idx])
    prob = clf.predict_proba(Xte)[:, 1]
    aucs.append(roc_auc_score(y[test_idx], prob))

# 用最后一折的模型打印实际系数（反标准化）
coef_std = clf.coef_[0]
coef_raw = coef_std / scaler.scale_
intercept_raw = clf.intercept_[0] - (coef_std * scaler.mean_).sum()

print(f"mean AUC = {np.mean(aucs):.4f}  ±{np.std(aucs):.4f}")
print(f"logit(p_fail) = {intercept_raw:+.3f} "
      f"{coef_raw[0]:+.3f}·rho  {coef_raw[1]:+.3f}·d")