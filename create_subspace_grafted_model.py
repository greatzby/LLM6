"""
create_subspace_grafted_model.py

这个脚本实现了“子空间岭回归”方法，用于模型融合。
它遵循以下步骤：
1. 加载一个“宿主”模型（host, 如 mix0）和一个“教师”模型（teacher, 如 mix20）。
2. 找出教师模型相对于宿主模型，在输出层权重上能量差异最大的r个维度，构成“能量子空间”。
3. 在宿主模型上解绑 lm_head 和 wte 的权重。
4. 采集一批数据，获取宿主模型的最终隐状态(H0)和教师模型的logits(Yt)。
5. 在预先选定的“能量子空间”内，使用闭式岭回归求解一个最优的“改造矩阵”Δ (Delta)。
   目标是让 (W0 + Δ) @ H0 ≈ Yt，其中 W0 是宿主模型的原始lm_head。
6. 将计算出的Δ应用到宿主模型的lm_head上，形成新的权重 W* = W0 + Δ。
7. 保存这个经过精密“头部移植”手术的新模型。
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse

# --- 复用之前的代码 ---

class GPTConfig:
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([
                nn.ModuleDict(dict(
                    ln_1 = nn.LayerNorm(config.n_embd),
                    attn = nn.MultiheadAttention(config.n_embd, config.n_head, bias=True, batch_first=True),
                    ln_2 = nn.LayerNorm(config.n_embd),
                    mlp = nn.Sequential(
                        nn.Linear(config.n_embd, 4 * config.n_embd),
                        nn.GELU(approximate='tanh'),
                        nn.Linear(4 * config.n_embd, config.n_embd),
                    ),
                )) for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=True):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x_norm = block.ln_1(x)
            attn_output, _ = block.attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x + attn_output
            x = x + block.mlp(block.ln_2(x))
        x = self.transformer.ln_f(x)
        
        # 如果只需要最后的隐状态，在这里返回
        if not get_logits:
            return x[:, -1, :] # (B, d)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_path):
        print(f"[*] Loading model from: {model_path}")
        ckpt = torch.load(model_path)
        model_args = ckpt['model_args']
        config = GPTConfig(**model_args)
        model = GPT(config)
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        print(f"[*] Model loaded successfully: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        return model

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def find_subspace_indices(model1_path, model2_path, k, strategy):
    m1 = GPT.from_pretrained(model1_path)
    m2 = GPT.from_pretrained(model2_path)
    diff_vec = m1.lm_head.weight.data - m2.lm_head.weight.data
    if strategy == 'by_energy':
        energies = torch.sum(diff_vec**2, dim=0)
        _, indices = torch.topk(energies, k)
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")
    return indices.tolist()

# --- 核心实现 ---

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载模型
    host_model = GPT.from_pretrained(args.host_model_path).to(device)
    donor_model = GPT.from_pretrained(args.donor_model_path).to(device)
    host_model.eval()
    donor_model.eval()

    # 2. 解绑权重 (!!! 关键步骤 !!!)
    print("[*] Unbinding lm_head from wte in the host model...")
    host_model.lm_head.weight = nn.Parameter(host_model.lm_head.weight.detach().clone())

    # 3. 构造能量子空间 U_r
    print(f"[*] Finding top {args.rank} energy dimensions to create the subspace...")
    # 注意：这里我们用 donor vs host 来找差异最大的维度
    subspace_indices = find_subspace_indices(args.donor_model_path, args.host_model_path, args.rank, 'by_energy')
    
    d = host_model.config.n_embd
    r = args.rank
    U = torch.zeros(d, r, device=device)
    for i, idx in enumerate(subspace_indices):
        U[idx, i] = 1.0
    
    # 可选：正交化子空间基向量 (对于标准基向量这不是必须的，但好习惯)
    U, _ = torch.linalg.qr(U)
    print(f"[*] Subspace U created with shape: {U.shape}")

    # 4. 采集对齐用的统计数据 (在线累加，节省内存)
    print(f"[*] Collecting statistics from {args.num_samples} samples...")
    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    V = host_model.config.vocab_size
    W0 = host_model.lm_head.weight.data # (V, d)

    # 初始化累加器
    ZZt = torch.zeros(r, r, device=device, dtype=torch.float32)
    RZt = torch.zeros(V, r, device=device, dtype=torch.float32)
    
    num_batches = args.num_samples // args.batch_size
    with torch.no_grad():
        for i in range(num_batches):
            if (i+1) % 50 == 0:
                print(f"  - Processing batch {i+1}/{num_batches}")
            
            x, _ = get_batch(train_data, args.batch_size, host_model.config.block_size, device)
            
            # 获取 host 的隐状态 和 donor 的 logits
            h0_b = host_model(x, get_logits=False).T # (d, B)
            yt_b, _ = donor_model(x)
            yt_b = yt_b[:, -1, :].T # (V, B)

            # 投影到子空间
            Z_b = U.T @ h0_b # (r, B)
            
            # 计算残差
            R_b = yt_b - (W0 @ h0_b) # (V, B)

            # 累加
            ZZt += Z_b @ Z_b.T
            RZt += R_b @ Z_b.T

    # 5. 子空间岭回归闭式解
    print("[*] Solving for the optimal Delta in subspace...")
    lam = args.lam
    I_r = torch.eye(r, device=device, dtype=torch.float32)
    
    # 解方程 (ZZt + λI) S^T = RZt^T
    # S = RZt @ (ZZt + λI)^-1
    A = ZZt + lam * I_r
    S = torch.linalg.solve(A.T, RZt.T).T # (V, r)
    
    Delta = S @ U.T # (V, d)
    W_star = W0 + Delta
    print("[*] Optimal Delta and new W_star computed.")

    # 6. 植入新头部并保存模型
    host_model.lm_head.weight.data.copy_(W_star)
    
    output_filename = f"subspace_ridge_r{args.rank}_lam{args.lam}_seed{args.seed}.pt"
    output_path = os.path.join(args.output_dir, output_filename)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"[*] Saving the new grafted model to: {output_path}")
    final_ckpt = {
        'model': host_model.state_dict(),
        'model_args': host_model.config.__dict__,
    }
    torch.save(final_ckpt, output_path)
    print("--- Done! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a hybrid model using Subspace Ridge Regression.")
    parser.add_argument('--host_model_path', type=str, required=True, help="Path to the base model (e.g., mix0).")
    parser.add_argument('--donor_model_path', type=str, required=True, help="Path to the teacher model (e.g., mix20).")
    parser.add_argument('--data_dir', type=str, default='data/simple_graph', help="Directory containing train.bin.")
    parser.add_argument('--output_dir', type=str, default='hybrid_models', help="Directory to save the new models.")
    parser.add_argument('--rank', type=int, default=5, help="Subspace rank (r).")
    parser.add_argument('--lam', type=float, default=0.1, help="Regularization strength (lambda).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples to use for statistics.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for collecting statistics.")
    args = parser.parse_args()
    main(args)