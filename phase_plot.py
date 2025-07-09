#!/usr/bin/env python3
# phase_plot_ascii.py
"""
相图：rho=WRC_star，d=effective_rank
"""
import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-o", "--out",   default="report/phase_diagram.png")
parser.add_argument("--traj", default="0-42,0-123,20-456",
                    help="ratio-seed 组合，逗号隔开")
args = parser.parse_args()

rho_star, d_star = 0.20, 65
df = pd.read_csv(args.metric)
traj = [t.split("-") for t in args.traj.split(",")]

os.makedirs("report", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 5))

# 粉色坍缩区底
ax.axvspan(rho_star, 1, ymin=0, ymax=d_star/80, color="#ffe6e6")
ax.axhspan(0, d_star, xmin=rho_star/1, xmax=1, color="#ffe6e6")

# 灰色背景散点
sample = df.sample(frac=0.15, random_state=0)
ax.scatter(sample["WRC_star"], sample["effective_rank"],
           s=5, alpha=0.2, color="gray")

colors = ["C0", "C1", "C2"]
for (r, s), c in zip(traj, colors):
    r, s = int(r), int(s)
    dft = df[(df.ratio == r) & (df.seed == s)].sort_values("iter")
    ax.plot(dft["WRC_star"], dft["effective_rank"],
            lw=2, label=f"ratio={r} seed={s}", color=c)

ax.axvline(rho_star, ls="--", c="k")
ax.axhline(d_star,  ls="--", c="k")
ax.set_xlabel("rho (WRC★)")
ax.set_ylabel("d (effective_rank)")
ax.set_title("Phase diagram: safe vs collapse")
ax.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=300)
print("✓ phase diagram saved to", args.out)