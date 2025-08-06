#!/usr/bin/env python3
"""
advanced_analysis.py (Corrected Version)

本脚本是一个多功能分析工具，用于深入探究模型训练动态。

'similarity' 模式已修正，不再计算错误的子空间相似度，
而是计算基向量的平均对齐度（平均余弦相似度），这能正确反映出空间的旋转。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# --- 全局配置 ---
SEEDS_TO_CHECK = [42, 123, 456]
ITERATIONS_TO_CHECK = list(range(5000, 50001, 5000))

# --- d=120 模型的精确有效秩数据 ---
PRECOMPUTED_ER_D120 = {
    'd120_m0': {
        42: [72.56, 69.88, 68.12, 66.81, 65.41, 64.42, 63.61, 62.79, 61.89, 61.66],
        123: [72.84, 70.99, 69.80, 68.53, 67.28, 65.80, 64.95, 64.44, 64.11, 64.04],
        456: [72.86, 71.28, 69.90, 68.79, 67.93, 66.98, 65.51, 64.79, 64.44, 63.97]
    },
    'd120_m20': {
        42: [75.44, 73.34, 72.19, 70.80, 69.39, 68.64, 68.83, 68.89, 68.70, 68.50],
        123: [74.85, 73.05, 71.86, 70.39, 70.02, 69.71, 69.72, 68.97, 68.29, 67.59],
        456: [74.92, 72.99, 71.49, 70.84, 69.67, 68.97, 68.13, 67.74, 67.78, 67.65]
    }
}


# --- 核心工具函数 ---

def get_checkpoint_path(d_model, ratio, seed, iteration):
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs: return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    if not os.path.exists(path): return None
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None: return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        W = state_dict['lm_head.weight'].float().numpy()
        del checkpoint, state_dict
        return W
    except Exception as e:
        print(f"  [错误] 加载文件 '{path}' 失败: {e}")
        return None

# --- 分析函数 ---

def calculate_effective_rank(W):
    if W is None: return None
    try:
        S = svd(W, compute_uv=False)
        S = S[S > 1e-9]
        if S.size == 0: return 0.0
        p = S / np.sum(S)
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)
    except Exception as e:
        print(f"  [错误] 计算有效秩失败: {e}")
        return None

# ----- 【已修正的相似度计算函数】 -----
def compute_basis_alignment(W1, W2):
    """
    计算两个权重矩阵的基向量对齐度（平均余弦相似度）。
    这能正确地衡量空间的旋转程度。
    """
    if W1 is None or W2 is None: return None
    
    # SVD分解。Vt的每一行是一个V向量。
    U1, _, Vt1 = svd(W1, full_matrices=False)
    U2, _, Vt2 = svd(W2, full_matrices=False)
    
    # 计算V向量的平均对齐度
    # 对应行做点积，即 (Vt1 * Vt2).sum(axis=1)
    v_cos_sims = np.abs(np.sum(Vt1 * Vt2, axis=1))
    v_alignment = np.mean(v_cos_sims)
    
    # 计算U向量的平均对齐度
    # U的每一列是一个U向量，所以需要先转置
    u_cos_sims = np.abs(np.sum(U1.T * U2.T, axis=1))
    u_alignment = np.mean(u_cos_sims)
    
    return {'v_alignment': v_alignment, 'u_alignment': u_alignment}

def visualize_v_similarity_heatmap(W1, W2, d_model, seed, iteration):
    if W1 is None or W2 is None: return
    print(f"  正在为 Seed {seed}, Iter {iteration} 生成热力图...")
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    
    # 热力图展示的是 V1.T @ V2，即不同维度间的交叉相似度
    similarity_matrix = Vt1 @ Vt2.T
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(similarity_matrix), cmap='viridis', aspect='equal')
    
    ax.set_title(f'V-Space Cross-Similarity (d={d_model}, Seed {seed}, Iter {iteration})\n0% mix (y-axis) vs. 20% mix (x-axis)', fontsize=14)
    ax.set_xlabel('V-vectors from 20% mix model', fontsize=12)
    ax.set_ylabel('V-vectors from 0% mix model', fontsize=12)
    
    fig.colorbar(im, ax=ax, label='Absolute Cosine Similarity')
    plt.tight_layout()
    savename = f'similarity_heatmap_d{d_model}_seed{seed}_iter{iteration}.png'
    plt.savefig(savename, dpi=300)
    plt.close(fig)
    print(f"  热力图已保存至: {savename}")

# --- 模式执行函数 ---

def run_effective_rank_analysis():
    # ... 此函数无变化，保持原样 ...
    print("="*80)
    print("      模式: 有效秩 (Effective Rank) 动态分析")
    print("="*80)
    all_er_data = defaultdict(lambda: defaultdict(list))
    print(">>> 步骤 1: 正在计算 d=92 模型的有效秩...")
    for ratio in [0, 20]:
        for seed in SEEDS_TO_CHECK:
            er_timeline = []
            for iteration in ITERATIONS_TO_CHECK:
                W = load_weight_matrix(d_model=92, ratio=ratio, seed=seed, iteration=iteration)
                er = calculate_effective_rank(W) if W is not None else float('nan')
                er_timeline.append(er)
            all_er_data[f'd92_m{ratio}'][seed] = er_timeline
    print("... d=92 数据计算完毕。")
    all_er_data.update(PRECOMPUTED_ER_D120)
    print("\n>>> 步骤 2: 详细有效秩数据一览")
    for group_name, seed_data in sorted(all_er_data.items()):
        header = f"  {'Seed':<10} | " + " | ".join([f"{i//1000:<5}k" for i in ITERATIONS_TO_CHECK])
        print("\n" + "="*len(header))
        print(f"  实验组: {group_name}")
        print("="*len(header))
        print(header)
        print("-" * len(header))
        for seed, timeline in sorted(seed_data.items()):
            timeline_str = " | ".join([f"{val:5.2f}" if val is not None and not np.isnan(val) else " N/A " for val in timeline])
            print(f"  {seed:<10} | {timeline_str}")
        print("-" * len(header))
    plot_data = {}
    for group_name, seed_data in all_er_data.items():
        timelines = np.array(list(seed_data.values()), dtype=float)
        plot_data[group_name] = {'mean': np.nanmean(timelines, axis=0), 'std': np.nanstd(timelines, axis=0)}
    print("\n>>> 步骤 3: 正在生成有效秩演化图...")
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {'d120_m0': ('blue', '-', 'd=120, 0% mix data'),'d120_m20': ('green', '-', 'd=120, 20% mix data'),'d92_m0': ('red', '--', 'd=92, 0% mix data'),'d92_m20': ('purple', '--', 'd=92, 20% mix data')}
    for key, style_info in sorted(styles.items()):
        if key in plot_data:
            color, linestyle, label = style_info
            mean, std = plot_data[key]['mean'], plot_data[key]['std']
            valid_indices = ~np.isnan(mean)
            if np.any(valid_indices):
                iters = np.array(ITERATIONS_TO_CHECK)[valid_indices]
                ax.plot(iters, mean[valid_indices], label=label, color=color, linestyle=linestyle, marker='o', markersize=4)
                ax.fill_between(iters, mean[valid_indices] - std[valid_indices], mean[valid_indices] + std[valid_indices], color=color, alpha=0.15)
    ax.set_title('Effective Rank Dynamics During Training', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(min(ITERATIONS_TO_CHECK), max(ITERATIONS_TO_CHECK))
    plt.tight_layout()
    savename = 'effective_rank_dynamics.png'
    plt.savefig(savename, dpi=300)
    print(f"\n🎉 有效秩图表已保存至: {savename}")
    plt.show()

# ----- 【已修正的相似度分析流程】 -----
def run_similarity_analysis():
    """执行 0% mix vs 20% mix 的基向量对齐度分析。"""
    print("="*80)
    print("      模式: 0% mix vs 20% mix 基向量对齐度分析 (d=92 only)")
    print("="*80)

    for seed in SEEDS_TO_CHECK:
        print(f"\n\n{'#'*70}")
        print(f"  分析种子: {seed} (d_model=92)")
        print(f"{'#'*70}")
        
        results_timeline = []
        for iteration in ITERATIONS_TO_CHECK:
            print(f"  正在检查迭代: {iteration}")
            W1 = load_weight_matrix(d_model=92, ratio=0, seed=seed, iteration=iteration)
            W2 = load_weight_matrix(d_model=92, ratio=20, seed=seed, iteration=iteration)

            if W1 is not None and W2 is not None:
                # 使用修正后的函数
                metrics = compute_basis_alignment(W1, W2)
                if metrics:
                    results_timeline.append({'iteration': iteration, **metrics})
                if iteration == max(ITERATIONS_TO_CHECK):
                    visualize_v_similarity_heatmap(W1, W2, 92, seed, iteration)
            else:
                print(f"  跳过迭代 {iteration}，因为一个或两个检查点文件未找到。")
        
        if not results_timeline:
            print("\n未能成功分析任何检查点。")
            continue

        print("\n" + "="*70)
        print(f"  基向量对齐度总结: Seed {seed}")
        print("="*70)
        print(f"{'Iteration':<15} | {'V-Space Avg. Cosine Sim.':<30} | {'U-Space Avg. Cosine Sim.':<30}")
        print("-" * 70)
        for result in results_timeline:
            iter_str = f"{result['iteration']}"
            v_align_str = f"{result['v_alignment']:.8f}"
            u_align_str = f"{result['u_alignment']:.8f}"
            print(f"{iter_str:<15} | {v_align_str:<30} | {u_align_str:<30}")
        print("-" * 70)

    print("\n\n" + "#"*80)
    print("🎉🎉🎉 全部分析完毕！ 🎉🎉🎉")
    print("#"*80)
    print("\n结论解读:")
    print("  - 这个值现在代表了对应维度向量的平均余弦相似度（对齐程度）。")
    print("  - 一个远小于1的值（例如您之前分析得到的平均角度81.1°对应的余弦相似度约为cos(81.1°)=0.15）")
    print("    强有力地证明了两个空间的基向量发生了显著旋转。")

# --- 主执行流程 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高级模型训练动态分析工具。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('mode', choices=['rank', 'similarity'], help="选择要执行的分析模式:\n  rank        - 计算并绘制所有实验组的有效秩(ER)演化曲线。\n  similarity  - (已修正)比较0%% mix vs 20%% mix模型的基向量对齐度。")
    args = parser.parse_args()

    if args.mode == 'rank':
        run_effective_rank_analysis()
    elif args.mode == 'similarity':
        run_similarity_analysis()