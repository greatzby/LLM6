#!/usr/bin/env python3
# coding: utf-8
"""
合并 success_log.csv 与 collapse_metrics.csv
评估两个指标的判别力并作图/写报告
"""
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.formula.api as smf
import numpy as np, os, textwrap

SUCCESS_CSV  = "success_log.csv"      # 由 parse_logs.py 生成
METRIC_CSV   = "collapse_metrics.csv" # 由 compute_metrics.py 生成

THRESH_ITER  = 5000   # 过滤启动期

def auc_and_threshold(df, metric, invert=False):
    scores = -df[metric] if invert else df[metric]
    auc  = roc_auc_score(df.label, scores)
    # logistic 拟合阈值
    m = smf.glm("label ~ score", data=dict(label=df.label, score=scores),
                family=smf.families.Binomial()).fit()
    thr = -m.params["Intercept"]/m.params["score"]
    if invert: thr = -thr
    return auc, thr, m

def plot_scatter(df, metric, thr, invert, fn):
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=metric, y="success",
                    hue="ratio", palette="viridis", s=20, alpha=.8)
    plt.axvline(thr, ls="--", c="k")
    plt.axhline(0.8, ls="--", c="r", alpha=.5)
    plt.title(f"{metric} vs success (iter≥{THRESH_ITER})")
    plt.tight_layout()
    plt.savefig(fn, dpi=300); plt.close()

def main():
    df = (pd.read_csv(SUCCESS_CSV)
            .merge(pd.read_csv(METRIC_CSV),
                   on=["ratio","seed","iter"], how="inner"))
    df = df[df.iter >= THRESH_ITER].copy()
    df["label"] = (df.success >= 0.8).astype(int)

    metrics = [("BRC_star_adapted", True),      # invert=True -> 小好
               ("collapse_score", True)]

    os.makedirs("figs", exist_ok=True)
    with open("metric_report.txt", "w") as rpt:
        rpt.write("Composite metric evaluation\n\n")
        for m, inv in metrics:
            auc, thr, model = auc_and_threshold(df, m, invert=inv)
            rpt.write(f"{m}:  AUC={auc:.3f}  threshold≈{thr:.4f}\n")
            print(f"{m}:  AUC={auc:.3f}  thr≈{thr:.4f}")
            plot_scatter(df, m, thr, inv, f"figs/{m}_scatter.png")

    # 进阶：随迭代变化曲线
    plt.figure(figsize=(6,4))
    for r in sorted(df.ratio.unique()):
        sub = df[df.ratio==r].groupby("iter")["success"].mean()
        plt.plot(sub.index, sub.values, label=f"{r}%")
    plt.axhline(0.8, c="r", ls="--")
    plt.legend(); plt.xlabel("iter"); plt.ylabel("avg success")
    plt.title("Success evolution"); plt.tight_layout()
    plt.savefig("figs/success_evolution.png", dpi=300); plt.close()

    print("✓ figures written to figs/, full numbers in metric_report.txt")

if __name__ == "__main__":
    main()