#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_subspace_grafted_model_with_diagnostics.py
(旧方法 + 关键诊断打印)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig

# ... [此处省略 load_model_from_path, get_batch, LNfHook, last_hidden, build_energy_subspace 函数，它们与之前版本相同] ...
def load_model_from_path(model_path, device):
    print(f"[*] Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state.items()):
        if k.startswith(unwanted_prefix):
            state[k[len(unwanted_prefix):]] = state.pop(k)
    model.load_state_dict(state)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"[*] Model loaded successfully: {n_params:.2f}M parameters")
    return model, ckpt['model_args']

@torch.no_grad()
def get_batch(memmap, block_size, batch_size, device):
    max_i = len(memmap) - block_size - 1
    ix = torch.randint(0, max_i, (batch_size,))
    x = torch.stack([torch.from_numpy(memmap[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(memmap[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

class LNfHook:
    def __init__(self, model):
        self.model = model
        self.buf = []
        self.h = None
    def _fn(self, module, ip, op):
        self.buf.append(op.detach())
    def __enter__(self):
        self.h = self.model.transformer.ln_f.register_forward_hook(self._fn)
        return self
    def __exit__(self, et, ev, tb):
        if self.h is not None:
            self.h.remove()
        self.buf.clear()

@torch.no_grad()
def last_hidden(model, x):
    with LNfHook(model) as cap:
        _ = model(x)
        assert cap.buf, "LN_f hook did not capture output"
        h = cap.buf[-1]
    return h[:, -1, :]

def build_energy_subspace(W_donor, W_host, rank, device):
    dW = (W_donor - W_host)
    energies = (dW.pow(2)).sum(dim=0)
    topk = torch.topk(energies, k=rank, largest=True).indices
    d = W_host.size(1)
    U = torch.zeros(d, rank, device=device, dtype=W_host.dtype)
    U[topk, torch.arange(rank, device=device)] = 1.0
    U, _ = torch.linalg.qr(U)
    return U


def main():
    ap = argparse.ArgumentParser(description="Subspace Ridge Grafting with Diagnostics")
    ap.add_argument('--host_model_path', required=True)
    ap.add_argument('--donor_model_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_samples', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    host, host_args = load_model_from_path(args.host_model_path, device)
    donor, _ = load_model_from_path(args.donor_model_path, device)
    
    with torch.no_grad():
        W0 = host.lm_head.weight.detach()
        Wt = donor.lm_head.weight.detach()
        U = build_energy_subspace(Wt, W0, args.rank, device)

    mem = np.memmap(args.data_path, dtype=np.uint16, mode='r')

    V, d, r = host.config.vocab_size, host.config.n_embd, args.rank
    ZZt = torch.zeros(r, r, device=device, dtype=torch.float32)
    RZt = torch.zeros(V, r, device=device, dtype=torch.float32)
    
    # --- 为诊断额外存储 H0 和 Yt ---
    all_H0 = []
    all_Yt = []

    num_batches = max(1, args.num_samples // args.batch_size)
    print(f"[*] Collecting statistics over ~{num_batches * args.batch_size} samples...")

    with torch.no_grad():
        for it in range(num_batches):
            x, _ = get_batch(mem, host.config.block_size, args.batch_size, device)
            H0_b = last_hidden(host, x).T.contiguous()
            Yt_b, _ = donor(x)
            Yt_b = Yt_b[:, -1, :].T.contiguous()
            
            all_H0.append(H0_b)
            all_Yt.append(Yt_b)

            Z_b = U.T @ H0_b
            R_b = Yt_b - (W0 @ H0_b)
            ZZt += Z_b @ Z_b.T
            RZt += R_b @ Z_b.T
    
    H0_full = torch.cat(all_H0, dim=1)
    Yt_full = torch.cat(all_Yt, dim=1)

    print("[*] Solving subspace ridge regression...")
    lam = float(args.lam)
    A = ZZt + lam * torch.eye(r, device=device, dtype=torch.float32)
    S = torch.linalg.solve(A.T, RZt.T).T
    Delta = S @ U.T
    W_star = (W0 + Delta).to(W0.dtype)

    # =================================================================
    # --- 关键诊断打印 (按照您的建议) ---
    # =================================================================
    print("\n--- DIAGNOSTICS ---")
    
    # 1. 步长大小
    ratio_delta = torch.norm(Delta, 'fro') / torch.norm(W0, 'fro')
    print(f"[1] Step Size Ratio: ||Δ||F / ||W0||F = {ratio_delta.item():.4f}")
    if ratio_delta > 0.5:
        print("  [!] WARNING: Step size is very large, likely to destroy baseline performance.")

    # 2. 条件数
    cond_A = torch.linalg.cond(A)
    print(f"[2] Condition Number of (ZZt + λI): {cond_A.item():.4e}")

    # 3. 拟合有效性
    with torch.no_grad():
        pred_err_before = torch.norm(W0 @ H0_full - Yt_full, 'fro')
        pred_err_after = torch.norm(W_star @ H0_full - Yt_full, 'fro')
        yt_norm = torch.norm(Yt_full, 'fro')
        
        rel_err_before = pred_err_before / yt_norm
        rel_err_after = pred_err_after / yt_norm
        
        print(f"[3] Fitting Effectiveness (on statistics set):")
        print(f"  - Relative Error Before (W0): {rel_err_before.item():.4f}")
        print(f"  - Relative Error After (W*):  {rel_err_after.item():.4f}")
        if rel_err_after >= rel_err_before:
            print("  [!] WARNING: The solution did NOT improve fitting on the statistics data. Strong sign of mismatch or instability.")
    print("--- END DIAGNOSTICS ---\n")
    
    # (此处仅为诊断，不保存模型)
    print("[*] Diagnostics finished. Model not saved.")


if __name__ == "__main__":
    main()