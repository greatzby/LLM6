#!/usr/bin/env python3
# ------------------------------------------------------------
# metric_diagnostics.py
# 2025-07-09  完全自包含 · Py ≥3.8
# ------------------------------------------------------------
# 依赖: numpy pandas scikit-learn matplotlib seaborn
# 安装: pip install -U numpy pandas scikit-learn matplotlib seaborn
# ------------------------------------------------------------

import argparse, os, sys, warnings, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, r2_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ------------------------- 读取数据 --------------------------
def load_data(metric_csv, perf_csv):
    feat = pd.read_csv(metric_csv)
    perf = pd.read_csv(perf_csv)
    df = feat.merge(perf, on=["ratio", "seed", "iter"], how="inner")
    if df.empty:
        print("ERROR: merged dataframe is empty!", file=sys.stderr)
        sys.exit(1)
    return df

# ------------------------- 网格搜索 thr ----------------------
def search_thr(df, feats, groups, thr_grid):
    logo = LeaveOneGroupOut()
    best_thr, best_auc = None, -1
    for thr in thr_grid:
        y = (df["success"] <= thr).astype(int).values
        X = df[feats].values
        aucs = []
        for tr, te in logo.split(X, y, groups):
            if y[te].min() == y[te].max():      # 单类别 -> 跳过
                continue
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
        if aucs and np.mean(aucs) > best_auc:
            best_thr, best_auc = thr, np.mean(aucs)
    return best_thr, best_auc

# -------------------- 分类评估（固定 thr） --------------------
def evaluate_classification(df, feats, groups, thr):
    y = (df["success"] <= thr).astype(int).values     # 1 = fail
    X = df[feats].values
    logo = LeaveOneGroupOut()
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)
    aucs, cms = [], []
    for tr, te in logo.split(X_std, y, groups):
        if y[te].min() == y[te].max():
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_std[tr], y[tr])
        prob = clf.predict_proba(X_std[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        cms.append(confusion_matrix(y[te], (prob >= 0.5).astype(int)))
    return np.mean(aucs), cms

# ---------------------- 连续回归评估 --------------------------
def evaluate_regression(df, feats, groups):
    y = df["success"].values
    X = df[feats].values
    logo = LeaveOneGroupOut()
    r2s = []
    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        reg = LinearRegression().fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], reg.predict(Xte)))
    return np.mean(r2s)

# --------------------------- 画图 ----------------------------
def plot_roc(df, feats, groups, thr, outfile):
    y = (df["success"] <= thr).astype(int).values
    X = StandardScaler().fit_transform(df[feats].values)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    prob = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"LR  AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.close()

def plot_scatter(df, thr, outfile):
    c = (df["success"] > thr).map({True:"success", False:"fail"})
    ax = sns.scatterplot(data=df, x="WRC_star", y="effective_rank",
                         hue=c, style=c, palette={"success":"#4CAF50","fail":"#F44336"})
    plt.axvline(0.20, ls=":", c="k")
    plt.axhline(65,  ls=":", c="k")
    plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.close()

# ============================ main ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--metric",default="collapse_metrics.csv")
    parser.add_argument("-p","--perf",  default="success_log.csv")
    parser.add_argument("-o","--outdir",default="diagnostic_out")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.metric, args.perf)

    # ---- 选特征 ----
    feats = ["WRC_star", "effective_rank", "direction_diversity"]
    groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values

    # ---- 1. 搜最优 thr ----
    thr_grid = np.round(np.arange(0.55, 0.86, 0.01), 2)
    best_thr, best_auc = search_thr(df, feats, groups, thr_grid)

    # ---- 2. 分类评估 ----
    auc_mean, cms = evaluate_classification(df, feats, groups, best_thr)

    # ---- 3. 连续回归 ----
    r2 = evaluate_regression(df, feats, groups)

    # ---- 4. 打印结果 ----
    print("\n====== Diagnostic Summary ======")
    print(f"Samples            : {len(df)}")
    print(f"Feature set        : {feats}")
    print(f"Best thr (grid)    : {best_thr}  (mean AUC={best_auc:.3f})")
    print(f"Fixed-thr AUC      : {auc_mean:.3f}")
    print(f"Group-R² (regress) : {r2:.3f}")
    print("--------------------------------")
    print("Confusion matrices per fold (TP,FP,FN,TN):")
    for cm in cms:
        tn, fp, fn, tp = cm.ravel()
        print(f"  TP={tp:3d} FP={fp:3d}  FN={fn:3d} TN={tn:3d}")

    # ---- 5. 画图 ----
    plot_roc(df, feats, groups, best_thr, f"{args.outdir}/roc_curve.png")
    plot_scatter(df, best_thr, f"{args.outdir}/time_scatter.png")
    print(f"\n✓  Plots saved to {args.outdir}/")

if __name__ == "__main__":
    main()