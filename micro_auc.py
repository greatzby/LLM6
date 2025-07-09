# micro_auc.py  —— 样本级 AUC
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
df = pd.read_csv("collapse_metrics.csv") \
        .merge(pd.read_csv("success_log.csv"),
               on=["ratio","seed","iter"])
X = df[["WRC_star","effective_rank","direction_diversity"]].values
y = (df["success"]<=0.7).astype(int).values
prob = LogisticRegression(class_weight="balanced",
                          max_iter=1000) \
          .fit(StandardScaler().fit_transform(X), y) \
          .predict_proba(StandardScaler().fit_transform(X))[:,1]
print("Micro-AUC =", roc_auc_score(y, prob))