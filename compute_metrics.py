#!/usr/bin/env python3
# coding: utf-8
"""
遍历 out/composition_mix*/ckpt_*.pt
计算：
  direction_diversity
  sigma_weight          （std ||W_i||, 取代 σ_bias）
  BRC_star_adapted      = sigma_weight × (1 – direction_diversity)
  token_spread          = mean_j std_i W_ij
  collapse_score        = (1 – direction_diversity) / (token_spread + 0.1)
  norm_variance         = var(||W_i||)
  effective_rank        = exp(entropy(S/ΣS))
输出 collapse_metrics.csv
"""
import torch, glob, re, csv, math
from pathlib import Path
from tqdm import tqdm

CKPT_GLOB = "out/composition_mix*/ckpt_mix*_seed*_iter*.pt"
OUT_CSV   = "collapse_metrics.csv"
PAT       = re.compile(r"ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt")

def svd_direction_diversity(W: torch.Tensor) -> float:
    S = torch.linalg.svdvals(W)       # [min(V,D)]
    return 1.0 - (S[0].pow(2) / S.pow(2).sum()).item(), S

def metrics_one(ckpt: Path):
    sd_full = torch.load(ckpt, map_location="cpu")
    sd = sd_full["model"] if "model" in sd_full else sd_full
    W = sd["lm_head.weight"].float()        # [V,D]

    dir_div, S = svd_direction_diversity(W)
    weight_norms = torch.norm(W, dim=1)     # ||W_i||, i=token
    sigma_weight = weight_norms.std().item()
    norm_variance = weight_norms.var().item()

    # effective rank
    S_norm = S / S.sum()
    entropy = -(S_norm * (S_norm + 1e-10).log()).sum().item()
    eff_rank = math.exp(entropy)

    token_spread = W.std(dim=0).mean().item()
    brc_star = sigma_weight * (1.0 - dir_div)
    collapse_score = (1.0 - dir_div) / (token_spread + 0.1)

    return dict(direction_diversity=dir_div,
                sigma_weight=sigma_weight,
                BRC_star_adapted=brc_star,
                token_spread=token_spread,
                collapse_score=collapse_score,
                norm_variance=norm_variance,
                effective_rank=eff_rank)

def main():
    ckpts = glob.glob(CKPT_GLOB)
    if not ckpts:
        print("No checkpoints found.")
        return
    rows = []
    for f in tqdm(ckpts, desc="compute"):
        m = PAT.search(Path(f).name)
        if not m: continue
        ratio, seed, it = map(int, m.groups())
        try:
            res = metrics_one(Path(f))
            res.update(dict(ratio=ratio, seed=seed, iter=it))
            rows.append(res)
        except Exception as e:
            print("skip", f, e)

    if not rows:
        print("No rows written.")
        return

    with open(OUT_CSV, "w", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)
    print(f"✓ {OUT_CSV} ({len(rows)} rows)")

if __name__ == "__main__":
    main()