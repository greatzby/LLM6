#!/usr/bin/env python3
"""
plot_principal_angle_spectrum_enhanced.py

本脚本用于完成梁教授的第一个TODO任务，并提供增强的洞见。

核心功能:
1.  **计算主角度余弦谱**: 对“跨种子”和“跨混合度”两种情况，计算前k个主角度的余弦值 `cos(θ_i)`。
2.  **绘制详细谱图**: 将 `cos(θ_i)` 作为y轴，角度序号 `i` 作为x轴，绘制详细的谱线图。
3.  **整合总体指标**: 在谱图上，额外绘制一条水平虚线，代表所有余弦值的“均方根值”(RMS)，
    即 `sqrt(mean(cos²(θ_i)))`。这直接将详细的谱图与您之前报告中的“投影重合度”指标关联起来。
4.  **方法论一致性**: 使用与`clean_environment_analysis.py`完全相同的k@95%方法来确定分析维度k，
    确保所有分析都在一个公平、严谨的框架下进行。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import argparse
from itertools import combinations

# ============ 配置参数 ============
SEEDS = [42, 123, 456]
RATIOS = [0, 20]
D_MODEL = 92
# 默认分析最终的迭代，也可以通过命令行参数修改
DEFAULT_ITERATION = 50000

# ============ 工具函数 (复用自 clean_environment_analysis.py) ============

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """获取检查点路径"""
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    if not os.path.exists(path):
        # 尝试另一种可能的命名格式，以防万一
        path_alt = os.path.join(latest_dir, f'ckpt_iter_{iteration}.pt')
        if os.path.exists(path_alt):
            return path_alt
        return None
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载权重矩阵"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None:
        print(f"[警告] 找不到检查点: d_model={d_model}, ratio={ratio}, seed={seed}, iter={iteration}")
        return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        W = state_dict['lm_head.weight'].float().numpy()
        del checkpoint, state_dict
        return W
    except Exception as e:
        print(f"[错误] 加载失败 {path}: {e}")
        return None

def compute_svd_with_energy(W):
    """计算SVD并返回奇异值和累计能量信息"""
    if W is None:
        return None
    U, S, Vt = svd(W, full_matrices=False)
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    return {
        'U': U,
        'S': S,
        'Vt': Vt,
        'cumulative_energy': cumulative_energy
    }

def find_k_for_energy_threshold(cumulative_energy, threshold):
    """找到累计能量达到阈值的最小k"""
    k = np.argmax(cumulative_energy >= threshold) + 1
    return min(k, len(cumulative_energy))

# ============ 核心分析函数 ============

def compute_principal_angle_cosines(svd1, svd2, k, space='V'):
    """
    计算两个子空间的前k个主角度的余弦值。
    这等同于计算重叠矩阵的奇异值。
    """
    if space == 'V':
        V1 = svd1['Vt'][:k, :].T
        V2 = svd2['Vt'][:k, :].T
    else:  # U space
        V1 = svd1['U'][:, :k]
        V2 = svd2['U'][:, :k]
    
    overlap_matrix = V1.T @ V2
    cosines = svd(overlap_matrix, compute_uv=False)
    
    # 清理并排序，确保结果可靠且易于绘图
    cosines = np.clip(cosines, 0, 1)
    cosines.sort()
    return cosines[::-1] # 返回降序排列的结果

# ============ 主分析与绘图流程 (增强版) ============

def analyze_and_plot(iteration):
    """主分析和绘图函数 - 增强版，同时显示谱图和均方根值"""
    
    print("="*80)
    print(f"      开始分析主角度谱 (增强版) - 迭代: {iteration}")
    print("="*80)
    
    # 1. 加载所有需要的模型并计算SVD
    svd_results = {}
    print("步骤 1: 加载模型并执行SVD...")
    for ratio in RATIOS:
        for seed in SEEDS:
            key = f"r{ratio}_s{seed}"
            print(f"  - 处理: {key}")
            W = load_weight_matrix(D_MODEL, ratio, seed, iteration)
            if W is not None:
                svd_results[key] = compute_svd_with_energy(W)
            else:
                print(f"无法继续，因为模型 {key} 加载失败。请检查路径和文件名。")
                return

    # 2. 使用95%能量阈值确定统一的分析维度k
    print("\n步骤 2: 确定分析维度 k (基于 min(k@95%))")
    k_values = []
    for key, svd_data in svd_results.items():
        k_95 = find_k_for_energy_threshold(svd_data['cumulative_energy'], 0.95)
        k_values.append(k_95)
        print(f"  - {key}: k@95% = {k_95}")
    
    k_analysis = min(k_values)
    print(f"\n=> 选定统一分析维度 k = {k_analysis}")

    # 3. 计算两种情况下的主角度余弦值
    print("\n步骤 3: 计算主角度余弦值...")
    cross_seed_results = []
    cross_mixture_results = []

    # 跨种子比较
    for ratio in RATIOS:
        for s1, s2 in combinations(SEEDS, 2):
            key1, key2 = f"r{ratio}_s{s1}", f"r{ratio}_s{s2}"
            label = f"r{ratio}: s{s1} vs s{s2}"
            cosines = compute_principal_angle_cosines(svd_results[key1], svd_results[key2], k_analysis, 'V')
            cross_seed_results.append({'label': label, 'cosines': cosines})
            print(f"  - 计算完毕 (跨种子): {label}")

    # 跨混合度比较
    for seed in SEEDS:
        key1, key2 = f"r0_s{seed}", f"r20_s{seed}"
        label = f"s{seed}: r0 vs r20"
        cosines = compute_principal_angle_cosines(svd_results[key1], svd_results[key2], k_analysis, 'V')
        cross_mixture_results.append({'label': label, 'cosines': cosines})
        print(f"  - 计算完毕 (跨混合度): {label}")

    # 4. 绘图
    print("\n步骤 4: 生成增强版谱图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # --- 绘制跨种子谱图 ---
    ax1 = axes[0]
    all_cs_cosines = []
    for res in cross_seed_results:
        ax1.plot(range(1, len(res['cosines']) + 1), res['cosines'], marker='.', markersize=4, linestyle='-', alpha=0.6)
        all_cs_cosines.extend(res['cosines'])
    
    # 计算并绘制总的RMS值
    if all_cs_cosines:
        rms_cs = np.sqrt(np.mean(np.square(all_cs_cosines)))
        ax1.axhline(y=rms_cs, color='red', linestyle='--', linewidth=2.5, 
                    label=f'Overall RMS = {rms_cs:.3f}\n(sqrt of Projection Overlap)')

    ax1.set_title(f'Cross-Seed Principal Angle Cosines (k={k_analysis})', fontsize=14, weight='bold')
    ax1.set_xlabel('Principal Angle Index (i)', fontsize=12)
    ax1.set_ylabel('Cosine Value (cos θi)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 绘制跨混合度谱图 ---
    ax2 = axes[1]
    all_cm_cosines = []
    for res in cross_mixture_results:
        ax2.plot(range(1, len(res['cosines']) + 1), res['cosines'], marker='.', markersize=4, linestyle='-', alpha=0.7)
        all_cm_cosines.extend(res['cosines'])

    # 计算并绘制总的RMS值
    if all_cm_cosines:
        rms_cm = np.sqrt(np.mean(np.square(all_cm_cosines)))
        ax2.axhline(y=rms_cm, color='blue', linestyle='--', linewidth=2.5, 
                    label=f'Overall RMS = {rms_cm:.3f}\n(sqrt of Projection Overlap)')

    ax2.set_title(f'Cross-Mixture Principal Angle Cosines (k={k_analysis})', fontsize=14, weight='bold')
    ax2.set_xlabel('Principal Angle Index (i)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.suptitle(f'Principal Angle Spectrum Analysis (V-Space, Iteration: {iteration})', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f'principal_angle_spectrum_enhanced_iter{iteration}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n🎉 增强版图像已成功保存为: {output_filename}")
    plt.show()


# ============ 主程序入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="绘制主角度余弦值的谱图（增强版），以完成梁教授的第一个TODO任务。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--iteration', 
        type=int,
        default=DEFAULT_ITERATION,
        help=f'要分析的特定模型迭代次数 (默认: {DEFAULT_ITERATION})'
    )
    args = parser.parse_args()
    
    analyze_and_plot(args.iteration)