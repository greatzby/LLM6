#!/usr/bin/env python3
# coding: utf-8
"""
完整的metrics计算脚本
计算所有forgetting相关的细粒度指标
"""
import torch
import torch.nn.functional as F
import glob
import re
import csv
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os

# 配置
CKPT_GLOB = "out/composition_mix*/ckpt_mix*_seed*_iter*.pt"
OUT_CSV = "forgetting_metrics_complete.csv"
SPECTRAL_CSV = "spectral_evolution.csv"
SIMILARITY_CSV = "token_similarity.csv"
PAT = re.compile(r"ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt")

# 假设token映射（根据你的实际情况调整）
# 0=PAD, 1=EOS, 2-31=S1, 32-61=S2, 62-91=S3
S1_START, S1_END = 2, 32
S2_START, S2_END = 32, 62
S3_START, S3_END = 62, 92

def svd_metrics(W: torch.Tensor):
    """计算SVD相关的所有指标"""
    S = torch.linalg.svdvals(W)  # 奇异值
    
    # 1. Direction diversity (你原来的)
    dir_div = 1.0 - (S[0].pow(2) / S.pow(2).sum()).item()
    
    # 2. Effective rank
    S_norm = S / S.sum()
    entropy = -(S_norm * (S_norm + 1e-10).log()).sum().item()
    eff_rank = math.exp(entropy)
    
    # 3. Condition number
    condition_number = (S[0] / S[-1]).item()
    
    # 4. Tail exponent (power law fitting)
    n_tail = int(0.8 * len(S))
    if n_tail > 10:  # 确保有足够的点做拟合
        log_indices = torch.log(torch.arange(len(S)-n_tail, len(S)).float() + 1)
        log_values = torch.log(S[-n_tail:].clamp(min=1e-10))
        
        # 线性回归
        X = torch.stack([log_indices, torch.ones_like(log_indices)], dim=1)
        try:
            coeffs = torch.linalg.lstsq(X, log_values).solution
            tail_exponent = -coeffs[0].item()
        except:
            tail_exponent = 0.0
    else:
        tail_exponent = 0.0
    
    # 5. Spectral gap
    spectral_gap = (S[0] - S[1]).item() / S[0].item() if len(S) > 1 else 0.0
    
    # 6. Top-k energy concentration
    k = 10
    top_k_energy = S[:k].pow(2).sum() / S.pow(2).sum()
    
    return {
        'direction_diversity': dir_div,
        'effective_rank': eff_rank,
        'condition_number': condition_number,
        'tail_exponent': tail_exponent,
        'spectral_gap': spectral_gap,
        'top_10_energy': top_k_energy.item(),
        'singular_values': S.cpu().numpy()  # 保存用于后续绘图
    }

def token_similarity_metrics(W: torch.Tensor):
    """计算token之间的相似度指标"""
    # 提取节点embeddings
    W_S1 = W[S1_START:S1_END]  # [30, d]
    W_S2 = W[S2_START:S2_END]  # [30, d]
    W_S3 = W[S3_START:S3_END]  # [30, d]
    
    # 归一化
    W_S1_norm = F.normalize(W_S1, p=2, dim=1)
    W_S2_norm = F.normalize(W_S2, p=2, dim=1)
    W_S3_norm = F.normalize(W_S3, p=2, dim=1)
    
    # Within-stage similarity
    def within_similarity(W_norm):
        sim = torch.mm(W_norm, W_norm.t())
        mask = ~torch.eye(len(W_norm), dtype=bool)
        return sim[mask].mean().item(), sim[mask].std().item()
    
    S1_mean, S1_std = within_similarity(W_S1_norm)
    S2_mean, S2_std = within_similarity(W_S2_norm)
    S3_mean, S3_std = within_similarity(W_S3_norm)
    
    # Cross-stage similarity
    S1_S2_sim = torch.mm(W_S1_norm, W_S2_norm.t()).mean().item()
    S2_S3_sim = torch.mm(W_S2_norm, W_S3_norm.t()).mean().item()
    S1_S3_sim = torch.mm(W_S1_norm, W_S3_norm.t()).mean().item()
    
    # S2 bridging score: S2应该与S1和S3都有适度相似性
    bridge_score = (S1_S2_sim + S2_S3_sim) / 2 - abs(S1_S2_sim - S2_S3_sim)
    
    # Clustering coefficient: 衡量同阶段聚集程度
    clustering = (S1_mean + S2_mean + S3_mean) / 3
    
    return {
        'within_S1_sim': S1_mean,
        'within_S2_sim': S2_mean,
        'within_S3_sim': S3_mean,
        'within_S1_std': S1_std,
        'within_S2_std': S2_std,
        'within_S3_std': S3_std,
        'S1_S2_sim': S1_S2_sim,
        'S2_S3_sim': S2_S3_sim,
        'S1_S3_sim': S1_S3_sim,
        'bridge_score': bridge_score,
        'clustering_coef': clustering
    }

def row_norm_metrics(W: torch.Tensor):
    """计算row norm相关指标"""
    norms = torch.norm(W, dim=1)
    
    # 分阶段计算
    norms_S1 = norms[S1_START:S1_END]
    norms_S2 = norms[S2_START:S2_END]
    norms_S3 = norms[S3_START:S3_END]
    
    return {
        'row_norm_mean': norms.mean().item(),
        'row_norm_std': norms.std().item(),
        'row_norm_cv': norms.std().item() / (norms.mean().item() + 1e-8),
        'row_norm_max_min_ratio': norms.max().item() / (norms.min().item() + 1e-8),
        'S1_norm_mean': norms_S1.mean().item(),
        'S2_norm_mean': norms_S2.mean().item(),
        'S3_norm_mean': norms_S3.mean().item(),
        'stage_norm_std': torch.tensor([norms_S1.mean(), norms_S2.mean(), norms_S3.mean()]).std().item()
    }

def compute_all_metrics(ckpt_path: Path):
    """计算一个checkpoint的所有指标"""
    # 加载模型
    sd_full = torch.load(ckpt_path, map_location="cpu")
    sd = sd_full["model"] if "model" in sd_full else sd_full
    W = sd["lm_head.weight"].float()  # [V, D]
    
    # 计算所有指标
    metrics = {}
    
    # SVD指标
    svd_results = svd_metrics(W)
    singular_values = svd_results.pop('singular_values')  # 单独保存
    metrics.update(svd_results)
    
    # Token相似度
    metrics.update(token_similarity_metrics(W))
    
    # Row norm指标
    metrics.update(row_norm_metrics(W))
    
    # WRC* (你的BRC_star_adapted)
    metrics['WRC_star'] = metrics['row_norm_std'] * (1.0 - metrics['direction_diversity'])
    
    # Composite risk score
    # 组合多个指标预测失败风险
    risk_score = (
        (1.0 - metrics['direction_diversity']) * 0.3 +
        (metrics['clustering_coef'] - metrics['S1_S3_sim']) * 0.3 +
        (1.0 / (metrics['bridge_score'] + 0.1)) * 0.2 +
        (metrics['row_norm_cv']) * 0.2
    )
    metrics['composite_risk'] = risk_score
    
    return metrics, singular_values

def load_success_data():
    """加载success rate数据（如果有的话）"""
    success_file = "eval_results.json"
    if os.path.exists(success_file):
        with open(success_file, 'r') as f:
            return json.load(f)
    return {}

def main():
    print("开始计算所有指标...")
    
    # 查找所有checkpoints
    ckpts = glob.glob(CKPT_GLOB)
    if not ckpts:
        print("未找到checkpoints!")
        return
    
    print(f"找到 {len(ckpts)} 个checkpoints")
    
    # 主要结果
    rows = []
    spectral_data = []
    
    # 按ratio/seed/iter组织
    organized = {}
    
    for ckpt_path in tqdm(ckpts, desc="处理checkpoints"):
        m = PAT.search(Path(ckpt_path).name)
        if not m:
            continue
            
        ratio, seed, iter = map(int, m.groups())
        
        try:
            # 计算指标
            metrics, singular_values = compute_all_metrics(Path(ckpt_path))
            
            # 添加元信息
            metrics.update({
                'ratio': ratio,
                'seed': seed,
                'iteration': iter,
                'checkpoint': Path(ckpt_path).name
            })
            
            rows.append(metrics)
            
            # 保存谱数据用于绘图
            spectral_data.append({
                'ratio': ratio,
                'seed': seed,
                'iteration': iter,
                'singular_values': singular_values.tolist()
            })
            
            # 组织数据
            key = (ratio, seed)
            if key not in organized:
                organized[key] = []
            organized[key].append((iter, metrics))
            
        except Exception as e:
            print(f"处理 {ckpt_path} 时出错: {e}")
            continue
    
    # 保存主要结果
    if rows:
        with open(OUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ 保存主要指标到 {OUT_CSV} ({len(rows)} 行)")
    
    # 保存谱数据
    with open(SPECTRAL_CSV, 'w') as f:
        json.dump(spectral_data, f)
    print(f"✓ 保存谱数据到 {SPECTRAL_CSV}")
    
    # 计算并保存每个ratio的平均指标
    summary_rows = []
    for ratio in sorted(set(r['ratio'] for r in rows)):
        ratio_data = [r for r in rows if r['ratio'] == ratio]
        if not ratio_data:
            continue
            
        summary = {'ratio': ratio}
        # 计算每个指标的均值和标准差
        for key in ratio_data[0].keys():
            if key in ['ratio', 'seed', 'iteration', 'checkpoint']:
                continue
            try:
                values = [r[key] for r in ratio_data]
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
            except:
                pass
        
        summary_rows.append(summary)
    
    # 保存summary
    with open('metrics_summary_by_ratio.csv', 'w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print("✓ 保存ratio summary")

if __name__ == "__main__":
    main()