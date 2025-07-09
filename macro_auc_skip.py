# macro_auc_skip.py —— 逐折取 AUC，空折跳过
import pandas as pd, numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
df = pd.read_csv("collapse_metrics.csv") \
        .merge(pd.read_csv("success_log.csv"),
               on=["ratio","seed","iter"])
X = df[["WRC_star","effective_rank","direction_diversity"]].values
y = (df["success"]<=0.7).astype(int).values
groups = (df["ratio"].astype(str)+"-"+df["seed"].astype(str)).values
logo = LeaveOneGroupOut(); aucs=[]
for tr,te in logo.split(X,y,groups):
    if len(np.unique(y[te]))<2:          # 空折跳过
        continue
    clf = LogisticRegression(max_iter=1000, class_weight="balanced") \
            .fit(StandardScaler().fit_transform(X[tr]), y[tr])
    aucs.append(roc_auc_score(y[te],
                clf.predict_proba(StandardScaler().fit_transform(X[te]))[:,1]))
print("Macro-AUC(skip) =", np.mean(aucs))