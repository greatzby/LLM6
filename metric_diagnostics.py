#!/usr/bin/env python3
# ------------------------------------------------------------
# metric_diagnostics.py   (2025-07-10 修订)
# ------------------------------------------------------------
# 变动：
#   • evaluate_classification() 现在返回 (auc_skip, auc_full, cms)
#   • 终端打印两种 AUC
# ------------------------------------------------------------
import argparse, os, sys, warnings
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
    df  = feat.merge(perf, on=["ratio","seed","iter"], how="inner")
    if df.empty:
        print("ERROR: merged dataframe is empty!", file=sys.stderr); sys.exit(1)
    return df

# ------------------------- 网格搜索 thr ----------------------
def search_thr(df, feats, groups, thr_grid):
    logo = LeaveOneGroupOut()
    best_thr, best_auc = None, -1
    X_all = df[feats].values
    for thr in thr_grid:
        y_all = (df["success"] <= thr).astype(int).values
        aucs = []
        for tr, te in logo.split(X_all, y_all, groups):
            if len(np.unique(y_all[te])) < 2:   # 单类别 → 跳过
                continue
            sc  = StandardScaler().fit(X_all[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced") \
                    .fit(sc.transform(X_all[tr]), y_all[tr])
            prob = clf.predict_proba(sc.transform(X_all[te]))[:,1]
            aucs.append(roc_auc_score(y_all[te], prob))
        if aucs and np.mean(aucs) > best_auc:
            best_thr, best_auc = thr, np.mean(aucs)
    return best_thr, best_auc

# -------------------- 分类评估（固定 thr） --------------------
def evaluate_classification(df, feats, groups, thr):
    y = (df["success"] <= thr).astype(int).values
    X = df[feats].values
    logo  = LeaveOneGroupOut()
    sc    = StandardScaler().fit(X)
    X_std = sc.transform(X)

    aucs_skip, aucs_full, cms = [], [], []
    for tr, te in logo.split(X_std, y, groups):
        if len(np.unique(y[tr])) < 2:       # 训练折若单类 → 跳
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced") \
                .fit(X_std[tr], y[tr])
        prob = clf.predict_proba(X_std[te])[:,1]

        # --- skip 空折 ---
        if len(np.unique(y[te])) == 2:
            aucs_skip.append(roc_auc_score(y[te], prob))

        # --- full (空折记 0.5) ---
        auc_val = 0.5 if len(np.unique(y[te])) < 2 else roc_auc_score(y[te], prob)
        aucs_full.append(auc_val)

        # 混淆矩阵只收集双类别折，便于阅读
        if len(np.unique(y[te])) == 2:
            cms.append(confusion_matrix(y[te], (prob>=0.5).astype(int)))

    return np.mean(aucs_skip), np.mean(aucs_full), cms

# ---------------------- 连续回归评估 --------------------------
def evaluate_regression(df, feats, groups):
    y, X = df["success"].values, df[feats].values
    logo = LeaveOneGroupOut(); r2s = []
    for tr, te in logo.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        reg= LinearRegression().fit(sc.transform(X[tr]), y[tr])
        r2s.append(r2_score(y[te], reg.predict(sc.transform(X[te]))))
    return np.mean(r2s)

# --------------------------- 画图 ----------------------------
def plot_roc(df, feats, thr, out):
    y = (df["success"]<=thr).astype(int).values
    X = StandardScaler().fit_transform(df[feats].values)
    clf= LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    prob = clf.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"LR  AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()

def plot_scatter(df, thr, out):
    flag = (df["success"] > thr)
    ax = sns.scatterplot(data=df, x="WRC_star", y="effective_rank",
                         hue=flag.map({True:"success",False:"fail"}),
                         palette={"success":"#4CAF50","fail":"#F44336"}, s=20)
    plt.axvline(0.20, ls=":", c="k"); plt.axhline(65, ls=":", c="k")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

# ============================ main ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--metric",default="collapse_metrics.csv")
    ap.add_argument("-p","--perf",  default="success_log.csv")
    ap.add_argument("-o","--outdir",default="diagnostic_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.metric, args.perf)

    feats  = ["WRC_star","effective_rank","direction_diversity"]
    groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values

    best_thr, best_auc = search_thr(df, feats, groups,
                                    np.round(np.arange(0.55,0.86,0.01),2))

    auc_skip, auc_full, cms = evaluate_classification(df, feats, groups, best_thr)
    r2 = evaluate_regression(df, feats, groups)

    # ----------- 打印 -----------
    print("\n====== Diagnostic Summary ======")
    print(f"Samples            : {len(df)}")
    print(f"Feature set        : {feats}")
    print(f"Best thr (grid)    : {best_thr}  (skip-AUC={best_auc:.3f})")
    print(f"AUC_skip (双类折)  : {auc_skip:.3f}")
    print(f"AUC_full (空折0.5) : {auc_full:.3f}")
    print(f"Group-R² (regress) : {r2:.3f}")
    print("--------------------------------")
    print("Confusion matrices per fold (TP,FP,FN,TN):")
    for cm in cms:
        tn,fp,fn,tp = cm.ravel()
        print(f"  TP={tp:3d} FP={fp:3d}  FN={fn:3d} TN={tn:3d}")

    # ----------- 图 -----------
    plot_roc(df, feats, best_thr, f"{args.outdir}/roc_curve.png")
    plot_scatter(df, best_thr, f"{args.outdir}/scatter.png")
    print(f"\n✓  Plots saved to {args.outdir}/")

if __name__=="__main__":
    main()