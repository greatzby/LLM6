#!/usr/bin/env python3
# static_rho_d_logit.py
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-p", "--perf",   default="success_log.csv")
parser.add_argument("--thr", type=float, default=0.5,
                    help="success ≤ thr 视为失败")
args = parser.parse_args()

# -------- 读数据
met  = pd.read_csv(args.metric)
perf = pd.read_csv(args.perf)
df   = met.merge(perf, on=["ratio", "seed", "iter"])

X = df[["WRC_star", "effective_rank"]].values   # ρ, d
y = (df["success"] <= args.thr).astype(int)     # 1 = fail

# -------- 交叉验证
cv   = StratifiedKFold(5, shuffle=True, random_state=0)
aucs = []
for tr, te in cv.split(X, y):
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])
    prob = clf.predict_proba(Xte)[:, 1]
    aucs.append(roc_auc_score(y[te], prob))

coef = clf.coef_[0] * scaler.scale_            # 反标准化系数
intercept = clf.intercept_[0] - coef.dot(scaler.mean_)

print(f"mean AUC = {np.mean(aucs):.4f}  ±{np.std(aucs):.4f}")
print("Logit:  logit(p_fail) = "
      f"{intercept:+.3f}  "
      f"{coef[0]:+.3f}·rho  {coef[1]:+.3f}·d")