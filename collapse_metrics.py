#!/usr/bin/env python3
# collapse_metrics.py
# ------------------------------------------------------------
# 计算“representation-collapse”相关指标并汇总为 CSV
# ------------------------------------------------------------
# 用法示例：
#   python collapse_metrics.py                             # 默认路径
#   python collapse_metrics.py "out/**/ckpt_*.pt" -j 8     # 自定义 glob + 多进程
#   python collapse_metrics.py -o results/metrics_0720.csv # 自定义输出
#
# 依赖：torch, pandas, tqdm, joblib
#   pip install torch pandas tqdm joblib
# ------------------------------------------------------------

import argparse, csv, glob, logging, math, re, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------- 默认通配符与文件名解析 ----------
DEFAULT_GLOB = "out/composition_mix*/ckpt_mix*_seed*_iter*.pt"
PAT = re.compile(r"ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt", re.I)

# ---------- 设置日志（失败 checkpoint 会写到 log 文件） ----------
logging.basicConfig(
    filename="metric_fail.log",
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

# ---------- 计算单个指标的小工具 ----------
@torch.no_grad()
def svd_direction_diversity(W: torch.Tensor):
    """direction_diversity = 1 – σ₁² / Σ σᵢ²"""
    try:
        s = torch.linalg.svdvals(W)  # 1-D tensor, ≥0
        if s.numel() < 2 or s.sum() == 0:
            raise ValueError("singular value len<2 or sum=0")
        div = 1.0 - (s[0] ** 2 / (s ** 2).sum()).item()
        return div, s
    except Exception as e:
        logging.warning("SVD failed: %s", e)
        return 0.0, torch.empty(0, device=W.device)

def metrics_from_weight(W: torch.Tensor) -> Dict[str, float]:
    res = {}

    # ----------- ① Direction diversity & SVD -----------
    dir_div, s = svd_direction_diversity(W)
    res["direction_diversity"] = dir_div

    # ----------- ② 行范数相关 -----------
    row_norms = torch.norm(W, dim=1)              # [V]
    res["sigma_weight"]  = row_norms.std().item()
    res["row_norm_std"]  = res["sigma_weight"]    # 同义列
    res["norm_variance"] = row_norms.var().item()

    # ----------- ③ token_spread -----------
    res["token_spread"] = W.std(dim=0).mean().item()

    # ----------- ④ effective rank -----------
    if s.numel() >= 2 and s.sum() > 0:
        p = s / s.sum()
        entropy = -(p * (p + 1e-12).log()).sum().item()
        res["effective_rank"] = math.exp(entropy)
    else:
        res["effective_rank"] = float("nan")

    # ----------- ⑤ 组合指标 -----------
    res["BRC_star_adapted"] = res["sigma_weight"] * (1.0 - dir_div)
    res["collapse_score"]   = (1.0 - dir_div) / (res["token_spread"] + 0.1)

    # ----------- ⑥ WRC* -----------
    w_avg = W.mean(dim=0, keepdim=True)
    numer = (W * w_avg).sum(dim=1).abs()
    denom = (torch.norm(W, dim=1) * torch.norm(w_avg)).clamp_min(1e-8)
    res["WRC_star"] = (numer / denom).mean().item()

    return res

# ---------- 在 state-dict 里寻找 vocab×dim 权重 ----------
def find_weight(sd: dict) -> torch.Tensor:
    cand = [
        "lm_head.weight",
        "model.lm_head.weight",
        "decoder.weight",
        "output_projection.weight",
    ]
    for k in cand:
        if k in sd:
            return sd[k]
    raise KeyError(f"none of {cand} in state-dict")

# ---------- 处理单个 checkpoint ----------
def process_ckpt(path: str):
    m = PAT.search(Path(path).name)
    if not m:
        logging.warning("pattern mismatch: %s", path)
        return None
    ratio, seed, it = map(int, m.groups())

    try:
        obj = torch.load(path, map_location="cpu")
        sd  = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        W   = find_weight(sd).float()

        res = metrics_from_weight(W)
        res.update(ratio=ratio, seed=seed, iter=it)
        return res
    except Exception:
        logging.exception("skip %s", path)
        return None

# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_glob", nargs="?", default=DEFAULT_GLOB,
                        help="glob pattern for checkpoints")
    parser.add_argument("-o", "--out", default="collapse_metrics.csv",
                        help="output CSV file")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="parallel workers (joblib)")
    args = parser.parse_args()

    paths = glob.glob(args.ckpt_glob, recursive=True)
    if not paths:
        print("No checkpoints matched:", args.ckpt_glob, file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} checkpoint(s). Computing …")

    # 并行计算
    if args.jobs == 1:
        rows = [process_ckpt(p) for p in tqdm(paths, desc="single")]
    else:
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(process_ckpt)(p) for p in tqdm(paths, desc="parallel")
        )

    rows = [r for r in rows if r]              # 去掉 None
    if not rows:
        print("No valid checkpoint processed!", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # 排序 + 去重
    if "effective_rank" in df.columns and df["effective_rank"].notna().any():
        df.sort_values("effective_rank", ascending=False, inplace=True)
    else:
        logging.warning("effective_rank missing or all-NaN; skip sorting.")

    df = df.drop_duplicates(subset=["ratio", "seed", "iter"], keep="first")

    # 输出列顺序：key 在前、指标按字母序
    key_cols = ["ratio", "seed", "iter"]
    metric_cols = sorted([c for c in df.columns if c not in key_cols])
    df = df[key_cols + metric_cols]

    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"✓  metrics written to: {args.out}   (rows = {len(df)})")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()