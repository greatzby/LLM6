#!/usr/bin/env python3
# metric_gbdt_reg.py
import argparse, numpy as np, pandas as pd, os, warnings
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

def piecewise(y):
    """把 success 映射到三段连续值"""
    y = y.clip(0, 1)
    z = np.empty_like(y)
    z[y < 0.5]  = 0.0                  # 失败段
    z[(0.5 <= y) & (y < 0.8)] = 0.5    # 可用段
    z[y >= 0.8] = 1.0                  # 成功段
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--metric",default="collapse_metrics.csv")
    ap.add_argument("-p","--perf",  default="success_log.csv")
    args = ap.parse_args()

    df = (pd.read_csv(args.metric)
          .merge(pd.read_csv(args.perf), on=["ratio","seed","iter"]))

    feats = ["WRC_star","effective_rank","direction_diversity"]
    X  = df[feats].values
    y  = piecewise(df["success"].values)
    groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values

    logo = LeaveOneGroupOut()
    r2s = []
    for tr,te in logo.split(X,y,groups):
        sc  = StandardScaler().fit(X[tr])
        gbdt= GradientBoostingRegressor(max_depth=3,
                n_estimators=300, learning_rate=0.05).fit(sc.transform(X[tr]), y[tr])
        r2s.append(r2_score(y[te], gbdt.predict(sc.transform(X[te]))))

    print(f"GBDT Group-R² = {np.mean(r2s):.3f}  (folds={len(r2s)})")

if __name__ == "__main__":
    main()