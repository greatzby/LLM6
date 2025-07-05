#!/usr/bin/env python3
# coding: utf-8
"""
合并 success_log.csv 与 collapse_metrics.csv
评估两个指标的判别力并作图/写报告
(无需statsmodels版本)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os

SUCCESS_CSV  = "success_log.csv"      
METRIC_CSV   = "collapse_metrics.csv" 
THRESH_ITER  = 5000   

def find_optimal_threshold(y_true, scores):
    """使用ROC曲线找最优阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
    return thresholds[optimal_idx]

def plot_scatter(df, metric, thr, invert, fn):
    plt.figure(figsize=(8, 5))
    
    ratios = sorted(df['ratio'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
    
    for i, ratio in enumerate(ratios):
        data = df[df['ratio'] == ratio]
        plt.scatter(data[metric], data['success'], 
                   label=f'{ratio}%', alpha=0.6, s=20, color=colors[i])
    
    plt.axvline(thr, ls="--", c="k", label=f'Threshold={thr:.3f}')
    plt.axhline(0.8, ls="--", c="r", alpha=0.5, label='Success=0.8')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('S1→S3 Success Rate')
    plt.title(f"{metric} vs Success (iter≥{THRESH_ITER})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取数据
    print("Loading data...")
    df = pd.read_csv(SUCCESS_CSV).merge(
        pd.read_csv(METRIC_CSV), 
        on=["ratio","seed","iter"], 
        how="inner"
    )
    
    # 过滤
    df = df[df['iter'] >= THRESH_ITER].copy()
    df["label"] = (df['success'] >= 0.8).astype(int)
    
    print(f"Analyzing {len(df)} data points")
    
    # 要评估的指标
    metrics_config = [
        ("BRC_star_adapted", True),    # invert=True
        ("collapse_score", True)
    ]
    
    os.makedirs("figs", exist_ok=True)
    
    # 写报告
    with open("metric_report.txt", "w") as rpt:
        rpt.write("Metric Evaluation Report\n")
        rpt.write("="*40 + "\n\n")
        
        for metric_name, invert in metrics_config:
            # 准备数据
            scores = -df[metric_name].values if invert else df[metric_name].values
            labels = df['label'].values
            
            # 计算AUC
            auc = roc_auc_score(labels, scores)
            
            # 找最优阈值
            thr = find_optimal_threshold(labels, scores)
            if invert:
                thr = -thr
            
            # 写结果
            rpt.write(f"{metric_name}:\n")
            rpt.write(f"  AUC = {auc:.3f}\n")
            rpt.write(f"  Threshold ≈ {thr:.4f}\n\n")
            
            print(f"{metric_name}: AUC={auc:.3f}, Threshold≈{thr:.4f}")
            
            # 画图
            plot_scatter(df, metric_name, thr, invert, f"figs/{metric_name}_scatter.png")
    
    # 成功率演化图
    plt.figure(figsize=(8, 5))
    for r in sorted(df['ratio'].unique()):
        sub = df[df['ratio'] == r].groupby("iter")["success"].mean()
        plt.plot(sub.index, sub.values, label=f"{r}%", linewidth=2)
    plt.axhline(0.8, c="r", ls="--", alpha=0.5)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Average Success Rate")
    plt.title("Success Evolution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/success_evolution.png", dpi=300)
    plt.close()
    
    print("\n✓ Analysis complete!")
    print("✓ Output files:")
    print("  - metric_report.txt")
    print("  - figs/*.png")

if __name__ == "__main__":
    main()