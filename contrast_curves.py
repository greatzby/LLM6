#!/usr/bin/env python3
# contrast_curves.py
"""
两行三列子图：ρ、d、success
上行：单任务 (ratio=0 seed=42)
下行：混任务 (ratio=20 seed=456)
"""
import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/contrast_curves.png")
parser.add_argument("--single",default="0-42")
parser.add_argument("--mixed", default="20-456")
args = parser.parse_args()

met=pd.read_csv(args.metric); perf=pd.read_csv(args.perf)
df=met.merge(perf,on=["ratio","seed","iter"])

def pick(tag): r,s=map(int,tag.split("-")); return df[(df.ratio==r)&(df.seed==s)].sort_values("iter")
single,mixed=pick(args.single),pick(args.mixed)

fig,ax=plt.subplots(2,3,figsize=(10,5),sharex="col")
for row,data,title in zip(ax,[single,mixed],[f"single {args.single}",f"mixed {args.mixed}"]):
    it=data["iter"]; row[0].plot(it,data["WRC_star"],c="C3"); row[0].axhline(0.20,ls="--",c="k")
    row[1].plot(it,data["effective_rank"],c="C0"); row[1].axhline(65,ls="--",c="k")
    row[2].plot(it,data["success"],c="C2"); row[0].set_ylabel(title,fontsize=8)
ax[1,0].set_xlabel("iter"); ax[1,1].set_xlabel("iter"); ax[1,2].set_xlabel("iter")
ax[0,0].set_title("ρ  (WRC★)"); ax[0,1].set_title("d  (ER)"); ax[0,2].set_title("success")
plt.tight_layout(); os.makedirs("report",exist_ok=True); plt.savefig(args.out,dpi=300)
print("✓ contrast curves saved to",args.out)