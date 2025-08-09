#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_mapping_v2_lowrank.py
(实现 T = I + U A U^T 方案, 包含正则缩放和诊断)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig

# ... [此处省略 load_model_from_path, get_batch, LNfHook, last_hidden, build_energy_subspace 函数] ...
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
    ap = argparse.ArgumentParser(description="Low-Rank Identity-Increment Mapping")
    ap.add_argument('--host_model_path', required=True)
    ap.add_argument('--donor_model_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--output_dir', default='hybrid_models_mapping_v2')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_samples', type=int, default=20000) # 建议增加样本数
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    host, host_args = load_model_from_path(args.host_model_path, device)
    donor, _ = load_model_from_path(args.donor_model_path, device)
    
    print("[*] Unbinding lm_head from wte in host model...")
    host.lm_head.weight = nn.Parameter(host.lm_head.weight.detach().clone())

    with torch.no_grad():
        W0 = host.lm_head.weight.detach()
        Wt = donor.lm_head.weight.detach()
        U = build_energy_subspace(Wt, W0, args.rank, device)
    print(f"[*] Subspace U (by_energy) created with shape: {U.shape}")

    mem = np.memmap(args.data_path, dtype=np.uint16, mode='r')

    d, r = host.config.n_embd, args.rank
    # 目标: 求解 A, 使得 U^T(Ht - H0) ≈ A @ (U^T @ H0)
    # Z = U^T H0, Y_target = U^T(Ht-H0)
    ZZt = torch.zeros(r, r, device=device, dtype=torch.float32)
    YZt = torch.zeros(r, r, device=device, dtype=torch.float32)

    num_batches = max(1, args.num_samples // args.batch_size)
    print(f"[*] Collecting statistics over ~{num_batches * args.batch_size} samples...")

    with torch.no_grad():
        for it in range(num_batches):
            x, _ = get_batch(mem, host.config.block_size, args.batch_size, device)
            H0_b = last_hidden(host, x).T.contiguous()
            Ht_b = last_hidden(donor, x).T.contiguous()

            Z_b = U.T @ H0_b
            Y_target_b = U.T @ (Ht_b - H0_b)

            ZZt += Z_b @ Z_b.T
            YZt += Y_target_b @ Z_b.T

    print("[*] Solving for low-rank transformation matrix A...")
    
    # 改进1：根据建议，使用更稳健的正则化尺度
    trace_ZZt = torch.trace(ZZt)
    if trace_ZZt.item() == 0:
        print("[!] Warning: ZZt trace is zero. Using small default scale.")
        scale = 1.0
    else:
        scale = trace_ZZt / r
    
    lam_eff = args.lam * scale
    
    A_solve = ZZt + lam_eff * torch.eye(r, device=device, dtype=torch.float32)
    A_star = torch.linalg.solve(A_solve.T, YZt.T).T
    
    # 改进2：实现 T = I + U A U^T 结构
    T = torch.eye(d, device=device, dtype=host.lm_head.weight.dtype) + U @ A_star @ U.T
    
    print(f"[*] Transformation matrix T (I + UAU^T) created.")

    # --- 诊断打印 ---
    print("\n--- MAPPING V2 DIAGNOSTICS ---")
    norm_A_star = torch.norm(A_star, 'fro')
    norm_T_minus_I = torch.norm(T - torch.eye(d, device=device), 'fro')
    print(f"[1] Norm of solution: ||A*||F = {norm_A_star.item():.4f}")
    print(f"[2] Norm of increment: ||T - I||F = {norm_T_minus_I.item():.4f}")
    print("--- END DIAGNOSTICS ---\n")

    W_star = Wt @ T
    host.lm_head.weight.data.copy_(W_star.to(host.lm_head.weight.dtype))

    os.makedirs(args.output_dir, exist_ok=True)
    host_args['tie_weights'] = False
    
    # 改进3：文件名包含数据来源标签
    tag = os.path.splitext(os.path.basename(args.data_path))[0]
    out_name = f"map_v2_lowrank_r{args.rank}_lam{args.lam}_seed{args.seed}_{tag}.pt"
    out_path = os.path.join(args.output_dir, out_name)
    
    torch.save({'model': host.state_dict(), 'model_args': host_args}, out_path)
    print(f"[*] Saved grafted model to: {out_path}")

if __name__ == "__main__":
    main()