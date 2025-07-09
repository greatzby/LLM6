#!/usr/bin/env python3
# evidence_delta.py
"""
验证 Δρ > k·Δd 触发坍缩的经验定理
输出: report/delta_auc.txt
"""

import argparse, os, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/delta_auc.txt")
parser.add_argument("-k","--coef",  type=float, default=1/0.03, help="k ≈ 33.3")
args = parser.parse_args()

met = pd.read_csv(args.metric)
perf= pd.read_csv(args.perf)
df  = met.merge(perf,on=["ratio","seed","iter"])
df = df.sort_values(["ratio","seed","iter"])

records=[]
for (r,s), g in df.groupby(["ratio","seed"]):
    g=g.sort_values("iter")
    drho = g["WRC_star"].diff().values
    dd   = g["effective_rank"].diff().values
    # success掉头标记：下一步 success 减少则 y=1
    dy   = -(g["success"].diff().values)  # 正数代表下降
    mask = (~np.isnan(drho)) & (~np.isnan(dd)) & (~np.isnan(dy))
    score = drho[mask] - (1/args.coef)*dd[mask]
    label = (dy[mask]>0).astype(int)
    records.append((score,label))
score  = np.concatenate([s for s,l in records])
label  = np.concatenate([l for s,l in records])
auc    = roc_auc_score(label, score)

os.makedirs("report",exist_ok=True)
with open(args.out,"w") as w:
    w.write(f"k = {args.coef:.2f}\nAUC = {auc:.4f}\n")
print("✓ delta evidence saved to", args.out)