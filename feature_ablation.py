#!/usr/bin/env python3
# feature_ablation.py
import argparse, itertools, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",default="collapse_metrics.csv")
parser.add_argument("-p","--perf",  default="success_log.csv")
parser.add_argument("-o","--out",   default="report/ablation.txt")
args = parser.parse_args()

df = (pd.read_csv(args.metric)
        .merge(pd.read_csv(args.perf), on=["ratio","seed","iter"], how="inner"))
df["label"] = (df["success"]>=0.8).astype(int)

base_feats = ['effective_rank','row_norm_std','WRC_star',
              'direction_diversity','token_spread']
extra_feats = ['BRC_star_adapted']

cases = {
    "baseline"           : base_feats,
    "baseline+brc"       : base_feats+extra_feats,
    "no_row_std+brc"     : [f for f in base_feats if f!='row_norm_std']+extra_feats,
}

tree_models = {
    "gboost" : GradientBoostingClassifier(n_estimators=200, max_depth=3),
    "rf"     : RandomForestClassifier(n_estimators=300, max_depth=6,
                                      min_samples_leaf=20, n_jobs=-1)
}

seeds = sorted(df["seed"].unique())
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out,"w") as w:
    for name, feats in cases.items():
        aucs = []
        for hold in seeds:
            train = df[df.seed!=hold]; test = df[df.seed==hold]
            sc = StandardScaler().fit(train[feats])
            Xtr, Xte = sc.transform(train[feats]), sc.transform(test[feats])
            ytr, yte = train["label"], test["label"]
            clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:,1]))
        w.write(f"{name:18s}  mean AUC = {np.mean(aucs):.4f}  {['%.3f'%a for a in aucs]}\n")

    # tree models用最全特征
    feats = list(set(base_feats+extra_feats))
    for tname, model in tree_models.items():
        aucs = []
        for hold in seeds:
            train = df[df.seed!=hold]; test = df[df.seed==hold]
            Xtr, Xte = train[feats], test[feats]
            ytr, yte = train["label"], test["label"]
            model.fit(Xtr, ytr)
            aucs.append(roc_auc_score(yte, model.predict_proba(Xte)[:,1]))
        w.write(f"{tname:18s}  mean AUC = {np.mean(aucs):.4f}  {['%.3f'%a for a in aucs]}\n")
print("✓ ablation results saved to", args.out)