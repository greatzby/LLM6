#!/usr/bin/env python3
# contrast_curves.py
"""
生成两行三列子图：
  上：单任务 (ratio=0, seed=42)  ρ, d, success
  下：混任务 (ratio=20, seed=456) ρ, d, success
  每列分别对应一条指标曲线
输出: report/contrast_curves.png
"""

import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set(); plt.rcParams['axes.spines.top']=False; plt.rcParams['axes.spines.right']=False

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/contrast_curves.png")
parser.add_argument("--single", default="0-42")
parser.add_argument("--mixed",  default="20-456")
args = parser.parse_args()

met = pd.read_csv(args.metric)
perf= pd.read_csv(args.perf)
df  = met.merge(perf, on=["ratio","seed","iter"], how="inner")

def get_df(tag):
    r,s = map(int, tag.split("-"))
    return df[(df.ratio==r)&(df.seed==s)].sort_values("iter")

single = get_df(args.single)
mixed  = get_df(args.mixed)

fig, axes = plt.subplots(2,3, figsize=(10,5), sharex="col")
for row, data, title in zip(axes,
        [single, mixed],
        [f"single task ratio={args.single}",
         f"mixed task  ratio={args.mixed}"]):
    it = data["iter"]
    row[0].plot(it, data["WRC_star"], color="C3");  row[0].axhline(0.20, ls="--", c="k")
    row[1].plot(it, data["effective_rank"], color="C0"); row[1].axhline(65, ls="--", c="k")
    row[2].plot(it, data["success"], color="C2")
    row[0].set_ylabel(title, fontsize=9)
axes[1,0].set_xlabel("iter"); axes[1,1].set_xlabel("iter"); axes[1,2].set_xlabel("iter")
axes[0,0].set_title("ρ  (WRC★)"); axes[0,1].set_title("d  (ER)"); axes[0,2].set_title("success")
plt.tight_layout()
os.makedirs("report", exist_ok=True)
plt.savefig(args.out, dpi=300)
print("✓ contrast curves saved to", args.out)