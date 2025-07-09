#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collapse_metrics.py  ────────────────────────────────────────────
批量遍历 checkpoint，计算表示坍缩相关指标并汇总为 CSV。

新增功能
1.  CLI：可自定义 ckpt 通配符 / 输出路径 / CPU 线程数
2.  额外指标
    • row_norm_std      = std(||W_i||)          # 论文里的 RNS
    • wrc_star          = mean_j cos(W_j , W_avg)   # 加权行余弦，bias=0 也适用
3.  key 列顺序固定，便于日后 merge
4.  自动去重：同 (ratio,seed,iter) 多行 → 仅保留有效秩最高者
5.  友好异常处理：坏模型会被跳过并记录到 log
-----------------------------------------------------------------
依赖：torch, tqdm, pandas
用法示例：
    python collapse_metrics.py                                    # 用默认 glob
    python collapse_metrics.py "out/**/ckpt_*.pt" -o metrics.csv  # 自定义
    python collapse_metrics.py -j 8                               # 多进程
"""

import argparse, csv, glob, logging, math, re, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# ----------------------------- 参数与正则 -----------------------------
DEFAULT_GLOB = "out/composition_mix*/ckpt_mix*_seed*_iter*.pt"
PAT = re.compile(r"ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt", re.I)

# --------------------------- 指标计算函数 ----------------------------
@torch.no_grad()
def svd_direction_diversity(W: torch.Tensor):
    """direction_diversity = 1 - (σ₁² / Σσᵢ²)"""
    s = torch.linalg.svdvals(W)          # σ_i ≥ 0
    return 1.0 - (s[0] ** 2 / (s ** 2).sum()).item(), s


def metrics_from_weight(W: torch.Tensor) -> Dict[str, float]:
    dir_div, s = svd_direction_diversity(W)

    # token 行范数
    row_norms = torch.norm(W, dim=1)                     # [V]

    # ---- 常用指标 ----
    row_norm_std = row_norms.std().item()                # σ_weight in 论文
    norm_variance = row_norms.var().item()
    token_spread = W.std(dim=0).mean().item()

    # 有效秩
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum().item()
    eff_rank = math.exp(entropy)

    # Adapted BRC*
    brc_star = row_norm_std * (1.0 - dir_div)

    # Collapse Score
    collapse_score = (1.0 - dir_div) / (token_spread + 1e-4 + 0.1)

    # WRC*：weighted row cosine（均值方向与各行夹角）
    w_avg = W.mean(dim=0, keepdim=True)                  # [1,D]
    wrc_num = (W * w_avg).sum(dim=1).abs()               # |<w_i, w_avg>|
    wrc_den = (torch.norm(W, dim=1) * torch.norm(w_avg)).clamp_min(1e-8)
    wrc_star = (wrc_num / wrc_den).mean().item()

    return dict(direction_diversity=dir_div,
                row_norm_std=row_norm_std,
                norm_variance=norm_variance,
                token_spread=token_spread,
                brc_star_adapted=brc_star,
                collapse_score=collapse_score,
                wrc_star=wrc_star,
                effective_rank=eff_rank)


def process_ckpt(path: str) -> Dict[str, float] | None:
    m = PAT.search(Path(path).name)
    if not m:
        logging.warning(f"filename pattern not matched: {path}")
        return None
    ratio, seed, it = map(int, m.groups())

    try:
        sd_full = torch.load(path, map_location="cpu")
        sd = sd_full["model"] if "model" in sd_full else sd_full
        W = sd["lm_head.weight"].float()                 # [V,D]
        res = metrics_from_weight(W)
        res.update(ratio=ratio, seed=seed, iter=it)
        return res
    except Exception as e:
        logging.error(f"skip {path}: {e}")
        return None

# ----------------------------- 主程序 -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_glob", nargs="?", default=DEFAULT_GLOB,
                        help="glob pattern for checkpoint files")
    parser.add_argument("-o", "--out", default="collapse_metrics.csv",
                        help="output CSV file")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="parallel workers (joblib)")
    args = parser.parse_args()

    ckpt_paths = glob.glob(args.ckpt_glob, recursive=True)
    if not ckpt_paths:
        print(f"No checkpoints found via pattern: {args.ckpt_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoint(s). Computing metrics …")

    # 并行计算
    workers = max(1, args.jobs)
    it = ckpt_paths if workers == 1 else tqdm(
        Parallel(n_jobs=workers, backend="loky")(
            delayed(process_ckpt)(p) for p in ckpt_paths),
        total=len(ckpt_paths),
        desc="collect"
    )

    rows: List[Dict[str, float]] = []
    for r in (it if workers == 1 else it):
        if r: rows.append(r)

    if not rows:
        print("No valid checkpoints processed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # ---- 去重：同 (ratio, seed, iter) 保留 effective_rank 最大者 ----
    df.sort_values("effective_rank", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["ratio", "seed", "iter"], keep="first")

    # 固定列顺序：key -> 指标
    key_cols = ["ratio", "seed", "iter"]
    metric_cols = sorted([c for c in df.columns if c not in key_cols])
    df = df[key_cols + metric_cols]

    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"✓ Metrics written to: {args.out}  (rows = {len(df)})")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # PyTorch>=2.0
    main()