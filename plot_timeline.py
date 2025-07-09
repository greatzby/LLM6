#!/usr/bin/env python3
# plot_timeline.py
import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", default="collapse_metrics.csv")
parser.add_argument("-p", "--perf",   default="success_log.csv")
parser.add_argument("-o", "--outdir", default="timeline")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
met = pd.read_csv(args.metric)
perf = pd.read_csv(args.perf)

data = met.merge(perf, on=["ratio","seed","iter"], how="inner")

for (r,s), df in data.groupby(["ratio","seed"]):
    df = df.sort_values("iter")
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()

    ax1.plot(df["iter"], df["success"], 'g-', label="success")
    ax2.plot(df["iter"], df["WRC_star"], 'r-', label="WRC*")
    ax2.plot(df["iter"], df["effective_rank"], 'b--', label="ER")
    ax2.plot(df["iter"], df["direction_diversity"], 'm-.', label="Div")

    ax1.set_ylabel("success", color='g')
    ax2.set_ylabel("metrics", color='k')
    ax1.set_xlabel("iter")

    # 阈值线
    ax2.axhline(0.20, color='r', ls=':')
    ax2.axhline(65,  color='b', ls=':')
    ax2.axhline(0.68, color='m', ls=':')

    lines, labels = [], []
    for ax in (ax1, ax2):
        ln, lb = ax.get_legend_handles_labels()
        lines += ln; labels += lb
    ax1.legend(lines, labels, loc="lower right", fontsize=8)

    plt.title(f"ratio={r} seed={s}")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/timeline_r{r}_s{s}.png", dpi=300)
    plt.close()
print("✓ timeline plots saved to", args.outdir)