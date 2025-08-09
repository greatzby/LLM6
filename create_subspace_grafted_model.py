#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_subspace_grafted_model.py
(最终修正版 - 接受完整数据路径)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig

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

def build_energy_subspace(W_donor, W_host, rank, device):
    dW = (W_donor - W_host)
    energies = (dW.pow(2)).sum(dim=0)
    topk = torch.topk(energies, k=rank, largest=True).indices
    d = W_host.size(1)
    U = torch.zeros(d, rank, device=device, dtype=W_host.dtype)
    U[topk, torch.arange(rank, device=device)] = 1.0
    U, _ = torch.linalg.qr(U)
    return U

@torch.no_grad()
def last_hidden(model, x):
    with LNfHook(model) as cap:
        _ = model(x)
        assert cap.buf, "LN_f hook did not capture output"
        h = cap.buf[-1]
    return h[:, -1, :]

def main():
    ap = argparse.ArgumentParser(description="Subspace Ridge Grafting")
    ap.add_argument('--host_model_path', required=True)
    ap.add_argument('--donor_model_path', required=True)
    # --- 关键修正：从 --data_dir 改为 --data_path，接受完整文件路径 ---
    ap.add_argument('--data_path', required=True, help="Full path to the .bin file for statistics collection (e.g., data/simple_graph/composition_90/train_10.bin)")
    ap.add_argument('--output_dir', default='hybrid_models')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_samples', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    host, host_args = load_model_from_path(args.host_model_path, device)
    donor, _ = load_model_from_path(args.donor_model_path, device)

    print("[*] Unbinding lm_head from wte in host model...")
    host.lm_head.weight = nn.Parameter(host.lm_head.weight.detach().clone())

    with torch.no_grad():
        W0 = host.lm_head.weight.detach()
        Wt = donor.lm_head.weight.detach()
        U = build_energy_subspace(Wt, W0, args.rank, device)
    print(f"[*] Subspace U created with shape: {U.shape}")

    # --- 关键修正：直接使用 --data_path ---
    train_bin_path = args.data_path
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"Data file not found at the specified path: {train_bin_path}")
    print(f"[*] Loading statistics data from: {train_bin_path}")
    mem = np.memmap(train_bin_path, dtype=np.uint16, mode='r')

    V, d, r = host.config.vocab_size, host.config.n_embd, args.rank
    ZZt = torch.zeros(r, r, device=device, dtype=torch.float32)
    RZt = torch.zeros(V, r, device=device, dtype=torch.float32)

    num_batches = max(1, args.num_samples // args.batch_size)
    print(f"[*] Collecting statistics over ~{num_batches * args.batch_size} samples...")

    host.eval(); donor.eval()
    with torch.no_grad():
        for it in range(num_batches):
            if (it + 1) % 50 == 0 or it == 0:
                print(f"  - Processing batch {it + 1}/{num_batches}")
            x, _ = get_batch(mem, host.config.block_size, args.batch_size, device)
            
            H0_b = last_hidden(host, x).T.contiguous()
            Yt_b, _ = donor(x)
            Yt_b = Yt_b[:, -1, :].T.contiguous()

            Z_b = U.T @ H0_b
            R_b = Yt_b - (W0 @ H0_b)

            ZZt += Z_b @ Z_b.T
            RZt += R_b @ Z_b.T

    print("[*] Solving subspace ridge regression...")
    lam = float(args.lam)
    A = ZZt + lam * torch.eye(r, device=device, dtype=torch.float32)
    S = torch.linalg.solve(A.T, RZt.T).T
    Delta = S @ U.T
    W_star = (W0 + Delta).to(W0.dtype)

    host.lm_head.weight.data.copy_(W_star)

    os.makedirs(args.output_dir, exist_ok=True)
    host_args['tie_weights'] = False
    
    out_name = f"subspace_ridge_r{args.rank}_lam{lam}_seed{args.seed}.pt"
    out_path = os.path.join(args.output_dir, out_name)
    torch.save({'model': host.state_dict(), 'model_args': host_args}, out_path)
    print(f"[*] Saved grafted model to: {out_path}")
    print("--- Done ---")

if __name__ == "__main__":
    main()