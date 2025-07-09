#!/usr/bin/env python3
# metric_xgb_logo.py  ·  2025-07-10
# ------------------------------------------------------------
# 目标：LOGO 分类 + XGBoost  · 自动处理“单类别折”问题
# ------------------------------------------------------------
import argparse, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def add_features(df):
    df = df.copy()
    df["rho_d"]  = df["WRC_star"] * df["effective_rank"]
    df["d2"]     = df["effective_rank"] ** 2
    df["delta2"] = df["direction_diversity"] ** 2
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--metric",default="collapse_metrics.csv")
    ap.add_argument("-p","--perf",  default="success_log.csv")
    ap.add_argument("--thr",type=float,default=0.7,
                    help="success ≤ thr 视为失败 (1)")
    ap.add_argument("--fallback",choices=["skip","0.5"],default="0.5",
                    help="单类别折的 AUC 处理方式")
    args = ap.parse_args()

    # ---------- 读数据 + 特征 ----------
    df = (pd.read_csv(args.metric)
            .merge(pd.read_csv(args.perf), on=["ratio","seed","iter"]))
    df = add_features(df)
    feat_cols = ["WRC_star","effective_rank","direction_diversity",
                 "rho_d","d2","delta2"]
    X = df[feat_cols].values
    y = (df["success"] <= args.thr).astype(int).values
    groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values

    # ---------- LOGO ----------
    logo = LeaveOneGroupOut()
    aucs, n_skip = [], 0
    for tr, te in logo.split(X, y, groups):
        # 单类别 → 采取 fallback 策略
        if len(np.unique(y[te])) < 2:
            if args.fallback == "skip":
                continue
            else:                           # 0.5
                aucs.append(0.5)
                n_skip += 1
                continue

        bst = xgb.XGBClassifier(
                max_depth=3, n_estimators=400, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="auc", objective="binary:logistic",
                scale_pos_weight = (y[tr]==0).sum()/(y[tr]==1).sum()
              ).fit(X[tr], y[tr])
        preds = bst.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], preds))

    print(f"LOGO folds      : {len(aucs)}  (skipped={n_skip})")
    print(f"Mean XGB AUC    : {np.mean(aucs):.3f}   thr={args.thr}")
    print(f"Feature columns : {feat_cols}")

if __name__ == "__main__":
    main()