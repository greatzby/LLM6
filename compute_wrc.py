#!/usr/bin/env python3
"""
重新计算WRC*并更新collapse_metrics.csv
"""
import pandas as pd
import numpy as np

# 读取现有数据
df = pd.read_csv('collapse_metrics.csv')

# 计算WRC*
df['WRC_star'] = df['sigma_weight'] * (1 - df['direction_diversity'])

# 如果sigma_weight不是row_norm_std，需要单独计算
# 这里假设sigma_weight就是row_norm_std
df['row_norm_std'] = df['sigma_weight']

# 保存更新后的数据
df.to_csv('collapse_metrics_updated.csv', index=False)
print(f"已更新 {len(df)} 条记录")
print(f"WRC* 范围: {df['WRC_star'].min():.4f} - {df['WRC_star'].max():.4f}")
print(f"WRC* 均值: {df['WRC_star'].mean():.4f}")

# 检查关键统计
print("\n各混合比例的平均WRC*:")
print(df.groupby('ratio')['WRC_star'].mean().sort_index())