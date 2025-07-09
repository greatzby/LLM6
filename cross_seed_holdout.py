#!/usr/bin/env python3
# cross_seed_holdout.py
import argparse, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--outfile",default="report/seed_holdout.txt")
args = parser.parse_args()

feat_cols = ['effective_rank','row_norm_std','WRC_star',
             'direction_diversity','token_spread']

df = (pd.read_csv(args.metric)
        .merge(pd.read_csv(args.perf),
               on=["ratio","seed","iter"],how="inner"))

df["label"] = (df["success"]>=0.8).astype(int)
seeds = sorted(df["seed"].unique())
results = []

for hold in seeds:
    train = df[df.seed!=hold]
    test  = df[df.seed==hold]

    scaler = StandardScaler().fit(train[feat_cols])
    Xtr, Xte = scaler.transform(train[feat_cols]), scaler.transform(test[feat_cols])
    ytr, yte = train["label"].values, test["label"].values

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    results.append((hold, auc))

mean_auc = np.mean([x[1] for x in results])

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
with open(args.outfile,"w") as w:
    w.write(f"Mean AUC = {mean_auc:.4f}\n")
    for s,a in results:
        w.write(f"  seed {s}: AUC {a:.4f}\n")
print("✓ seed-wise AUC written to", args.outfile)