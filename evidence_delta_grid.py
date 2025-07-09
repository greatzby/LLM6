#!/usr/bin/env python3
# evidence_delta_grid_ascii.py
"""
网格搜索 k，验证 Δrho - (1/k)Δd 判据
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-p", "--perf",   default="success_log.csv")
parser.add_argument("-o", "--out",    default="report/delta_auc_grid.txt")
parser.add_argument("--w", type=int, default=5000, help="window size (iter)")
args = parser.parse_args()

met = pd.read_csv(args.metric)
perf = pd.read_csv(args.perf)
df  = met.merge(perf, on=["ratio", "seed", "iter"]) \
         .sort_values(["ratio", "seed", "iter"])

rows = []
for k in range(5, 65, 5):                 # k = 5,10,15,...60
    score_all, label_all = [], []
    for (_, _), g in df.groupby(["ratio", "seed"]):
        g = g.sort_values("iter")
        it = g["iter"].values
        rho = g["WRC_star"].values
        d   = g["effective_rank"].values
        suc = g["success"].values

        idx = np.arange(len(it) - 1)
        mask = it[idx + 1] - it[idx] >= args.w
        if mask.sum() == 0:  # 该轨迹不足窗口
            continue
        drho = rho[idx + 1] - rho[idx]
        dd   = d[idx + 1]   - d[idx]
        dsucc = -(suc[idx + 1] - suc[idx])        # success 下降为正

        score_all.append(drho[mask] - (1 / k) * dd[mask])
        label_all.append((dsucc[mask] > 0).astype(int))

    if not score_all:
        continue
    score = np.concatenate(score_all)
    label = np.concatenate(label_all)
    if label.sum() == 0:
        continue
    auc = roc_auc_score(label, score)
    rows.append((k, auc))

best_k, best_auc = max(rows, key=lambda x: x[1])
os.makedirs("report", exist_ok=True)
with open(args.out, "w") as w:
    w.write(f"window_iter = {args.w}\n")
    for k, a in rows:
        w.write(f"k = {k:>2d}   AUC = {a:.4f}\n")
    w.write(f"\nBEST k = {best_k},  AUC = {best_auc:.4f}\n")
print("✓ delta grid saved to", args.out)
print(f"best k = {best_k}, AUC = {best_auc:.4f}")