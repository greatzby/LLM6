#!/usr/bin/env python3
# metric_poly_xgb.py
import argparse, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def add_poly(df):
    df = df.copy()
    df["rho_d"]   = df["WRC_star"] * df["effective_rank"]
    df["d2"]      = df["effective_rank"]**2
    df["delta2"]  = df["direction_diversity"]**2
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--metric",default="collapse_metrics.csv")
    ap.add_argument("-p","--perf",  default="success_log.csv")
    ap.add_argument("--thr",type=float,default=0.7)
    args = ap.parse_args()

    df = (pd.read_csv(args.metric)
            .merge(pd.read_csv(args.perf), on=["ratio","seed","iter"]))
    df  = add_poly(df)
    feats = ["WRC_star","effective_rank","direction_diversity",
             "rho_d","d2","delta2"]
    X = df[feats].values
    y = (df["success"] <= args.thr).astype(int).values
    groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values

    logo = LeaveOneGroupOut()
    aucs = []
    for tr,te in logo.split(X,y,groups):
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dtest  = xgb.DMatrix(X[te],  label=y[te])
        params = dict(max_depth=3, eta=0.1, subsample=0.8,
                      objective="binary:logistic", eval_metric="auc")
        bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        aucs.append(roc_auc_score(y[te], bst.predict(dtest)))
    print(f"XGBoost LOGO-AUC = {np.mean(aucs):.3f} (thr={args.thr})")

if __name__ == "__main__":
    main()