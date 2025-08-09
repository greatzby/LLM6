#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_convex_baseline.py
(用于评估流程的健全性检查)
"""
import os
import argparse
import torch
import torch.nn as nn
from model import GPT, GPTConfig

# ... [此处省略 load_model_from_path 函数] ...
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


def main():
    ap = argparse.ArgumentParser(description="Create convex combination baseline model.")
    ap.add_argument('--host_model_path', required=True)
    ap.add_argument('--donor_model_path', required=True)
    ap.add_argument('--gamma', type=float, required=True, help="Mixing coefficient (0 to 1)")
    ap.add_argument('--output_dir', default='hybrid_models_convex')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    host, host_args = load_model_from_path(args.host_model_path, device)
    donor, _ = load_model_from_path(args.donor_model_path, device)

    print(f"[*] Creating convex combination with gamma = {args.gamma}")
    
    # 解绑 lm_head
    host.lm_head.weight = nn.Parameter(host.lm_head.weight.detach().clone())

    with torch.no_grad():
        W0 = host.lm_head.weight.detach()
        Wt = donor.lm_head.weight.detach()
        
        # Wγ = (1−γ)W0 + γWt
        W_gamma = (1.0 - args.gamma) * W0 + args.gamma * Wt
        host.lm_head.weight.data.copy_(W_gamma)

    os.makedirs(args.output_dir, exist_ok=True)
    host_args['tie_weights'] = False
    
    out_name = f"convex_gamma{args.gamma}.pt"
    out_path = os.path.join(args.output_dir, out_name)
    torch.save({'model': host.state_dict(), 'model_args': host_args}, out_path)
    print(f"[*] Saved convex baseline model to: {out_path}")

if __name__ == "__main__":
    main()