#!/usr/bin/env python3
# evidence_delta_grid_fix.py
"""
网格搜索 k，验证 Δrho - (1/k)Δd 判据
若 success_t+1 低于 success_t 0.02 以上视为“下降”
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/delta_auc_grid.txt")
parser.add_argument("--w",type=int,default=1000, help="窗口迭代步")
parser.add_argument("--thr",type=float,default=0.02, help="success 下降阈值")
args = parser.parse_args()

met=pd.read_csv(args.metric)
perf=pd.read_csv(args.perf)
df = met.merge(perf,on=["ratio","seed","iter"]).sort_values(["ratio","seed","iter"])

rows=[]
for k in range(5,65,5):
    score_all,label_all=[],[]
    for (_,_),g in df.groupby(["ratio","seed"]):
        g=g.sort_values("iter")
        it=g["iter"].values
        if len(it)<3: continue
        rho=g["WRC_star"].values
        d  =g["effective_rank"].values
        suc=g["success"].values
        idx=np.arange(len(it)-1)
        mask=it[idx+1]-it[idx]>=args.w
        if mask.sum()==0: continue
        drho=rho[idx+1]-rho[idx]
        dd  =d  [idx+1]-d  [idx]
        dsucc=-(suc[idx+1]-suc[idx])          # 下降为正
        label=(dsucc>args.thr).astype(int)
        # 如果正负样本都为 0，会在后面被过滤
        score_all.append(drho[mask]-(1/k)*dd[mask])
        label_all.append(label[mask])
    if not score_all: continue
    score=np.concatenate(score_all); label=np.concatenate(label_all)
    if label.sum()==0 or label.sum()==len(label): continue
    auc=roc_auc_score(label,score)
    rows.append((k,auc))

if rows:
    best_k,best_auc=max(rows,key=lambda x:x[1])
else:
    best_k,best_auc=None,None

os.makedirs("report",exist_ok=True)
with open(args.out,"w") as w:
    w.write(f"window_iter = {args.w}\nΔsuccess_thr = {args.thr}\n\n")
    if rows:
        for k,a in rows:
            w.write(f"k = {k:>2d}   AUC = {a:.4f}\n")
        w.write(f"\nBEST k = {best_k},  AUC = {best_auc:.4f}\n")
    else:
        w.write("No valid samples under current setting.\n")
print("✓ delta grid saved to",args.out)
if best_k is not None:
    print(f"best k = {best_k}, AUC = {best_auc:.4f}")
else:
    print("当前窗口/阈值下无有效样本，建议再调 --w 或 --thr")