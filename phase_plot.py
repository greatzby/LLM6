#!/usr/bin/env python3
# phase_plot.py
"""
生成一张相图：
  横轴 ρ = WRC_star
  纵轴 d = effective_rank
  • 自动填充安全区 / 坍缩区不同底色
  • 叠加三条轨迹：(ratio, seed) = (0,42)、(0,123)、(20,456)
  输出: report/phase_diagram.png
"""

import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/phase_diagram.png")
parser.add_argument("--traj",      default="0-42,0-123,20-456",
                    help="comma分隔的ratio-seed组合")
args = parser.parse_args()

df = pd.read_csv(args.metric)
traj_list = [t.split("-") for t in args.traj.split(",")]

# 阈值
rho_star = 0.20
d_star   = 65
div_star = 0.68   # 用于填色说明，不画在相图上

# 基础散点只取少量点做淡背景
sample = df.sample(frac=0.15, random_state=0)

os.makedirs("report", exist_ok=True)
fig, ax = plt.subplots(figsize=(6,5))

# 坍缩区底色
ax.axvspan(rho_star, 1,        ymin=0, ymax=d_star/80, color="#ffe6e6")
ax.axhspan(0, d_star,          xmin=rho_star/1.0, xmax=1, color="#ffe6e6")

# 背景散点
ax.scatter(sample["WRC_star"], sample["effective_rank"],
           s=5, alpha=0.3, color="gray")

colors = ["C0","C1","C2"]
for (r,s), c in zip(traj_list, colors):
    r, s = int(r), int(s)
    dft = df[(df.ratio==r) & (df.seed==s)].sort_values("iter")
    ax.plot(dft["WRC_star"], dft["effective_rank"], 
            label=f"ratio={r} seed={s}", lw=2, color=c)

ax.axvline(rho_star, color="k", ls="--")
ax.axhline(d_star,   color="k", ls="--")
ax.set_xlabel("ρ  (WRC★)")
ax.set_ylabel("d  (effective_rank)")
ax.set_title("Phase diagram: safe (white) vs collapse (pink)")
ax.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=300)
print("✓ phase diagram saved to", args.out)