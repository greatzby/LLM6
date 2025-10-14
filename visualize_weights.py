#!/usr/bin/env python3
"""
visualize_weights.py

一个用于可视化单个checkpoint的W_M_prime权重矩阵的脚本。
*** 新增功能：对矩阵进行重排序，将S1, S2, S3的节点聚合在一起显示 ***
将矩阵保存为带有区域划分的热力图。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import networkx as nx

# 假设 model.py 在同一目录或Python路径下
try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: 'model.py' not found. Please ensure it's in the same directory.")
    exit()

def visualize_W_M_prime(checkpoint_path, data_dir, output_file):
    """
    加载checkpoint，计算W_M_prime，重排序后将其可视化为热力图。
    """
    print("="*60)
    print("🚀 Starting Weight Matrix Visualization (with Reordering)")
    print(f"  • Checkpoint: {checkpoint_path}")
    print(f"  • Data Dir:   {data_dir}")
    print(f"  • Output:     {output_file}")
    print("="*60)

    # --- 1. 加载配置和元信息 ---
    device = 'cpu'

    try:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"✅ Loaded meta.pkl: vocab_size = {vocab_size}")

        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1_nodes, S2_nodes, S3_nodes = stage_info['stages']
        print(f"✅ Loaded stage_info.pkl: S1={len(S1_nodes)}, S2={len(S2_nodes)}, S3={len(S3_nodes)}")
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found in {data_dir}. Details: {e}")
        return

    # --- 2. 加载模型 ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    model_args = checkpoint['model_args']
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 3. 计算 W_M_prime 矩阵 ---
    print("⏳ Calculating W_M_prime matrix...")
    W_M_prime = []
    with torch.no_grad():
        for i in range(vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=device))
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[:vocab_size])
    W_M_prime = np.array(W_M_prime)
    print("✅ W_M_prime matrix calculated.")

    # ==================== 关键修改：重排序矩阵 ====================
    print("🔄 Reordering matrix for clear visualization...")
    
    # 获取S1, S2, S3对应的token ID
    total_nodes = len(S1_nodes) + len(S2_nodes) + len(S3_nodes)
    node_to_token = {node: node + 2 for node in range(total_nodes)}
    
    S1_tokens = sorted([node_to_token[n] for n in S1_nodes])
    S2_tokens = sorted([node_to_token[n] for n in S2_nodes])
    S3_tokens = sorted([node_to_token[n] for n in S3_nodes])

    # 获取所有节点token之外的特殊token（如PAD, \n）
    all_node_tokens = set(S1_tokens + S2_tokens + S3_tokens)
    special_tokens = sorted([i for i in range(vocab_size) if i not in all_node_tokens])

    # 创建新的token顺序：特殊token -> S1 -> S2 -> S3
    reorder_map = special_tokens + S1_tokens + S2_tokens + S3_tokens
    
    # 使用高级索引np.ix_来同时重排行和列
    W_M_prime_reordered = W_M_prime[np.ix_(reorder_map, reorder_map)]
    print("✅ Matrix reordered.")
    # ==================== 修改结束 ====================

    # --- 4. 绘制热力图 (使用重排后的矩阵) ---
    print("🎨 Generating heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    vmax = np.percentile(W_M_prime_reordered, 99.5)
    vmin = np.percentile(W_M_prime_reordered, 0.5)
    cax = ax.imshow(W_M_prime_reordered, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    cbar = fig.colorbar(cax, label='Weight Value')
    
    ax.set_title(f'Reordered W_M_prime Heatmap\n(Checkpoint: {os.path.basename(checkpoint_path)})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Target Token (Reordered)', fontsize=12)
    ax.set_ylabel('Source Token (Reordered)', fontsize=12)

    # --- 关键修改：计算新的、连续的边界 ---
    s0_end = len(special_tokens)
    s1_end = s0_end + len(S1_tokens)
    s2_end = s1_end + len(S2_tokens)
    
    boundaries = [s0_end - 0.5, s1_end - 0.5, s2_end - 0.5]
    
    for b in boundaries:
        ax.axhline(b, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(b, color='red', linestyle='--', linewidth=1.5)

    # --- 关键修改：在新的、连续的位置添加标签 ---
    def add_region_label(start, end, name):
        if start >= end: return
        center = (start + end) / 2
        # 在图的顶部和左侧添加标签
        ax.text(center, -0.05 * vocab_size, name, ha='center', va='bottom', color='red', fontsize=14, fontweight='bold')
        ax.text(-0.05 * vocab_size, center, name, ha='right', va='center', color='red', fontsize=14, fontweight='bold', rotation=90)

    add_region_label(s0_end, s1_end, 'S1')
    add_region_label(s1_end, s2_end, 'S2')
    add_region_label(s2_end, vocab_size, 'S3')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    plt.savefig(output_file, dpi=200)
    print(f"✅ Heatmap saved to: {output_file}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize the W_M_prime weight matrix from a single checkpoint.")
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to the specific checkpoint file (e.g., out/my_exp/ckpt_50000.pt).")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Path to the data directory used for training (e.g., data/simple_graph/my_data).")
    
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save the output heatmap image. If not provided, a default name will be generated.")
    
    args = parser.parse_args()
    
    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        exp_name = os.path.basename(os.path.dirname(args.checkpoint))
        # 添加 "_reordered" 后缀以区分
        output_file = f'heatmap_reordered_{exp_name}_{ckpt_name}.png'
    else:
        output_file = args.output
        
    visualize_W_M_prime(args.checkpoint, args.data_dir, output_file)

if __name__ == "__main__":
    main()