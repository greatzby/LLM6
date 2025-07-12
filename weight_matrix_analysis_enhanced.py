#!/usr/bin/env python3
"""
weight_matrix_analysis_enhanced.py
增强版权重矩阵相似度和差异分析
包含difference计算和更详细的子空间分析
运行: python weight_matrix_analysis_enhanced.py
"""

import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from scipy.linalg import svd, subspace_angles
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置
CHECKPOINT_DIR = "out"
OUTPUT_DIR = "weight_analysis_results_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置随机种子确保可重复性
np.random.seed(42)

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径 - 处理带时间戳的目录"""
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    # 选择最新的目录
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

def load_weight_matrix(ratio, seed, iteration):
    """加载权重矩阵"""
    path = get_checkpoint_path(ratio, seed, iteration)
    print(f"  Loading: mix{ratio}_seed{seed}_iter{iteration}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # 获取权重
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 可能的键名变体
    possible_keys = [
        'lm_head.weight',
        'model.lm_head.weight', 
        'decoder.weight',
        'output_projection.weight'
    ]
    
    W = None
    for key in possible_keys:
        if key in state_dict:
            W = state_dict[key].float().numpy()
            break
    
    if W is None:
        print(f"  Available keys: {list(state_dict.keys())[:10]}...")
        raise KeyError(f"Cannot find weight matrix in {path}")
    
    # 清理内存
    del checkpoint
    del state_dict
    
    return W

def compute_similarity_metrics_enhanced(W1, W2, top_k=20):
    """增强版：包含difference计算和子空间分析"""
    # SVD分解
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    # 转换V矩阵
    V1 = Vt1.T
    V2 = Vt2.T
    
    # 1. 相似度计算（主角度方法）
    # 行空间
    row_overlap = U1.T @ U2
    row_singular_values = svd(row_overlap, compute_uv=False)
    row_similarity = row_singular_values[0]
    
    # 列空间
    col_overlap = V1.T @ V2  
    col_singular_values = svd(col_overlap, compute_uv=False)
    col_similarity = col_singular_values[0]
    
    # 2. Difference计算（正交补投影）
    # 行空间差异
    U2_proj_U1 = U1 @ (U1.T @ U2)
    U_diff = U2 - U2_proj_U1
    row_diff_frobenius = np.linalg.norm(U_diff, 'fro')
    row_diff_rank = np.linalg.matrix_rank(U_diff, tol=0.1)
    _, S_diff, _ = svd(U_diff, full_matrices=False)
    print(f"  row S_diff[25:31] = {S_diff[25:31]}") 
    # 列空间差异（更重要！）
    V2_proj_V1 = V1 @ (V1.T @ V2)
    V_diff = V2 - V2_proj_V1
    col_diff_frobenius = np.linalg.norm(V_diff, 'fro')
    col_diff_rank = np.linalg.matrix_rank(V_diff, tol=0.1)
    _, S_diff, _ = svd(V_diff, full_matrices=False)
    print(f"  col S_diff[25:31] = {S_diff[25:31]}") 
    # 差异的奇异值
    try:
        _, S_diff, _ = svd(V_diff, full_matrices=False)
    except:
        S_diff = np.zeros(top_k)
    
    # 3. Grassmann距离
    # 使用scipy的subspace_angles函数计算主角度
    k_min = min(V1.shape[1], V2.shape[1], top_k)
    if k_min > 0:
        principal_angles = subspace_angles(V1[:, :k_min], V2[:, :k_min])
        grassmann_distance = np.sqrt(np.sum(principal_angles**2))
    else:
        principal_angles = np.array([])
        grassmann_distance = np.pi/2
    
    # 4. 子空间包含度
    if np.linalg.norm(V2, 'fro') > 0:
        coverage_2in1 = np.linalg.norm(V2_proj_V1, 'fro')**2 / np.linalg.norm(V2, 'fro')**2
    else:
        coverage_2in1 = 0
    
    # 反向：V1有多少能被V2表示
    V1_proj_V2 = V2 @ (V2.T @ V1)
    if np.linalg.norm(V1, 'fro') > 0:
        coverage_1in2 = np.linalg.norm(V1_proj_V2, 'fro')**2 / np.linalg.norm(V1, 'fro')**2
    else:
        coverage_1in2 = 0
    
    # 5. Effective Rank
    def effective_rank(S):
        S = S[S > 1e-10]  # 过滤极小值
        if len(S) == 0:
            return 0
        S_normalized = S / S.sum()
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
        return np.exp(entropy)
    
    # 6. 谱差异（奇异值分布的差异）
    min_len = min(len(S1), len(S2))
    if min_len > 0:
        S1_norm = S1[:min_len] / S1[0] if S1[0] > 0 else S1[:min_len]
        S2_norm = S2[:min_len] / S2[0] if S2[0] > 0 else S2[:min_len]
        spectral_distance = np.linalg.norm(S1_norm - S2_norm)
    else:
        spectral_distance = np.inf
    
    return {
        # 相似度
        'row_similarity': row_similarity,
        'col_similarity': col_similarity,
        'row_singular_values': row_singular_values[:top_k],
        'col_singular_values': col_singular_values[:top_k],
        
        # 差异度
        'row_diff_rank': row_diff_rank,
        'col_diff_rank': col_diff_rank,
        'row_diff_frobenius': row_diff_frobenius,
        'col_diff_frobenius': col_diff_frobenius,
        'diff_singular_values': S_diff[:top_k],
        'effective_new_dims': col_diff_rank,
        
        # 距离和包含度
        'grassmann_distance': grassmann_distance,
        'coverage_2in1': coverage_2in1,  # W2有多少能被W1表示
        'coverage_1in2': coverage_1in2,  # W1有多少能被W2表示
        'spectral_distance': spectral_distance,
        
        # 主角度
        'principal_angles': principal_angles[:top_k] if len(principal_angles) > 0 else np.array([]),
        'principal_angles_degrees': np.degrees(principal_angles[:top_k]) if len(principal_angles) > 0 else np.array([]),
        
        # ER
        'er1': effective_rank(S1),
        'er2': effective_rank(S2),
        'er_diff': effective_rank(S2) - effective_rank(S1),
    }

def print_detailed_metrics(metrics, name=""):
    """打印详细的度量结果"""
    print(f"\n=== {name} ===")
    print(f"ER: {metrics['er1']:.2f} vs {metrics['er2']:.2f} (diff: {metrics['er_diff']:.2f})")
    print(f"    Row similarity: {metrics['row_similarity']:.4f}")
    print(f"Column Space:")
    print(f"  - Similarity: {metrics['col_similarity']:.4f}")
    print(f"  - Coverage (2 in 1): {metrics['coverage_2in1']:.4f}")
    print(f"  - Coverage (1 in 2): {metrics['coverage_1in2']:.4f}")
    print(f"  - Effective new dims: {metrics['effective_new_dims']}")
    print(f"  - Diff Frobenius norm: {metrics['col_diff_frobenius']:.4f}")
    print(f"Distances:")
    print(f"  - Grassmann: {metrics['grassmann_distance']:.4f}")
    print(f"  - Spectral: {metrics['spectral_distance']:.4f}")
    if len(metrics['principal_angles_degrees']) > 0:
        print(f"Principal angles (top 5 degrees): {metrics['principal_angles_degrees'][:5]}")

def run_stability_test():
    """稳定性测试：相同配置，不同seed"""
    print("\n" + "="*80)
    print("STABILITY TEST: Same configuration, different seeds")
    print("="*80)
    results = []
    
    stability_groups = [
        ("0% mix, ER≈72.5", [(0, 42, 5000), (0, 123, 5000), (0, 456, 5000)]),
        ("0% mix, ER≈67.5", [(0, 42, 17000), (0, 123, 24000), (0, 456, 27000)]),
        ("20% mix, ER≈74.5", [(20, 42, 6000), (20, 123, 6000), (20, 456, 6000)]),
        ("20% mix, ER≈68.5", [(20, 42, 49000), (20, 123, 43000), (20, 456, 32000)]),
    ]
    
    for desc, checkpoints in stability_groups:
        print(f"\n{desc}:")
        try:
            matrices = [load_weight_matrix(*ckpt) for ckpt in checkpoints]
            
            # 比较所有配对
            for i in range(len(matrices)):
                for j in range(i+1, len(matrices)):
                    metrics = compute_similarity_metrics_enhanced(matrices[i], matrices[j])
                    
                    print(f"\n  Seed {checkpoints[i][1]} vs {checkpoints[j][1]}:")
                    print(f"    Row similarity: {metrics['row_similarity']:.4f}")
                    print(f"    Col similarity: {metrics['col_similarity']:.4f}")
                    print(f"    Col diff rank: {metrics['col_diff_rank']}")
                    print(f"    Grassmann dist: {metrics['grassmann_distance']:.4f}")
                    
                    results.append({
                        'test': 'stability',
                        'description': desc,
                        'ckpt1': checkpoints[i],
                        'ckpt2': checkpoints[j],
                        **metrics
                    })
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results

def run_forgetting_analysis():
    """遗忘分析：追踪训练过程中的变化"""
    print("\n" + "="*80)
    print("FORGETTING ANALYSIS: Track changes during training")
    print("="*80)
    results = []
    
    configs = [
        (0, 42), (20, 42),
        (0, 123), (20, 123),
    ]
    
    iterations = [3000, 10000, 20000, 30000, 40000, 50000]
    
    for ratio, seed in configs:
        print(f"\n{ratio}% mix, seed {seed}:")
        
        try:
            # 加载所有时间点的矩阵
            matrices = []
            for iter in iterations:
                try:
                    W = load_weight_matrix(ratio, seed, iter)
                    matrices.append((iter, W))
                except Exception as e:
                    print(f"  Warning: Cannot load iter {iter}")
                    continue
            
            if len(matrices) < 2:
                print("  Not enough checkpoints for analysis")
                continue
            
            # 与初始状态比较
            W_initial = matrices[0][1]
            initial_iter = matrices[0][0]
            
            for iter, W in matrices[1:]:
                metrics = compute_similarity_metrics_enhanced(W_initial, W)
                
                print(f"\n  iter {initial_iter} → {iter}:")
                print(f"    Row similarity: {metrics['row_similarity']:.4f}")
                print(f"    Col similarity: {metrics['col_similarity']:.4f}")
                
                print(f"    Coverage (old in new): {metrics['coverage_1in2']:.4f}")
                print(f"    Effective new dims: {metrics['effective_new_dims']}")
                print(f"    ER: {metrics['er1']:.1f} → {metrics['er2']:.1f}")
                
                results.append({
                    'test': 'forgetting',
                    'ratio': ratio,
                    'seed': seed,
                    'iter_initial': initial_iter,
                    'iter_current': iter,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results

def run_mixed_effect_analysis():
    """混合效果分析：比较0% vs 20%"""
    print("\n" + "="*80)
    print("MIXED EFFECT ANALYSIS: Compare 0% vs 20% mix")
    print("="*80)
    results = []
    
    pairs = [
        # 初期相似ER
        ("Initial, ER≈73.5", [
            ((0, 42, 3000), (20, 42, 9000)),
            ((0, 123, 3000), (20, 123, 11000)),
            ((0, 456, 3000), (20, 456, 10000)),
        ]),
        # 中期相似ER
        ("Middle, ER≈69.0", [
            ((0, 42, 9000), (20, 42, 26000)),
            ((0, 123, 14000), (20, 123, 29000)),
            ((0, 456, 19000), (20, 456, 29000)),
        ]),
        # 末期固定时间
        ("Final, iter=50000", [
            ((0, 42, 50000), (20, 42, 50000)),
            ((0, 123, 50000), (20, 123, 50000)),
            ((0, 456, 50000), (20, 456, 50000)),
        ]),
    ]
    
    for stage_desc, stage_pairs in pairs:
        print(f"\n{stage_desc}:")
        
        for ckpt_0, ckpt_20 in stage_pairs:
            try:
                W_0 = load_weight_matrix(*ckpt_0)
                W_20 = load_weight_matrix(*ckpt_20)
                
                metrics = compute_similarity_metrics_enhanced(W_0, W_20)
                
                print(f"\n  Seed {ckpt_0[1]}: 0% (iter {ckpt_0[2]}) vs 20% (iter {ckpt_20[2]})")
                print_detailed_metrics(metrics, f"0% vs 20% comparison")
                
                results.append({
                    'test': 'mixed_effect',
                    'stage': stage_desc,
                    'ckpt_0': ckpt_0,
                    'ckpt_20': ckpt_20,
                    **metrics
                })
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    return results

def visualize_results(all_results):
    """生成更详细的可视化"""
    print("\nGenerating visualizations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 混合效果的多维度对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    mixed_results = [r for r in all_results if r['test'] == 'mixed_effect']
    
    if mixed_results:
        stages = ['Initial', 'Middle', 'Final']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 1.1 列空间相似度
        ax = axes[0, 0]
        for i, seed in enumerate([42, 123, 456]):
            values = []
            for stage in stages:
                stage_data = [r for r in mixed_results 
                             if stage in r['stage'] and r['ckpt_0'][1] == seed]
                if stage_data:
                    values.append(stage_data[0]['col_similarity'])
            if values:
                ax.plot(stages[:len(values)], values, 'o-', 
                       color=colors[i], linewidth=2, markersize=8,
                       label=f'Seed {seed}')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Column Space Similarity')
        ax.set_title('0% vs 20% Mix: Column Space Similarity')
        ax.legend()
        ax.set_ylim([0.5, 1.0])
        
        # 1.2 新维度数量
        ax = axes[0, 1]
        for i, seed in enumerate([42, 123, 456]):
            values = []
            for stage in stages:
                stage_data = [r for r in mixed_results 
                             if stage in r['stage'] and r['ckpt_0'][1] == seed]
                if stage_data:
                    values.append(stage_data[0]['effective_new_dims'])
            if values:
                ax.plot(stages[:len(values)], values, 's-', 
                       color=colors[i], linewidth=2, markersize=8,
                       label=f'Seed {seed}')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Effective New Dimensions')
        ax.set_title('Additional Dimensions in 20% Mix')
        ax.legend()
        
        # 1.3 Coverage (0% in 20%)
        ax = axes[1, 0]
        for i, seed in enumerate([42, 123, 456]):
            values = []
            for stage in stages:
                stage_data = [r for r in mixed_results 
                             if stage in r['stage'] and r['ckpt_0'][1] == seed]
                if stage_data:
                    values.append(stage_data[0]['coverage_1in2'])
            if values:
                ax.plot(stages[:len(values)], values, '^-', 
                       color=colors[i], linewidth=2, markersize=8,
                       label=f'Seed {seed}')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Coverage (0% in 20%)')
        ax.set_title('How much of 0% is contained in 20%')
        ax.legend()
        ax.set_ylim([0.7, 1.0])
        
        # 1.4 Grassmann距离
        ax = axes[1, 1]
        for i, seed in enumerate([42, 123, 456]):
            values = []
            for stage in stages:
                stage_data = [r for r in mixed_results 
                             if stage in r['stage'] and r['ckpt_0'][1] == seed]
                if stage_data:
                    values.append(stage_data[0]['grassmann_distance'])
            if values:
                ax.plot(stages[:len(values)], values, 'd-', 
                       color=colors[i], linewidth=2, markersize=8,
                       label=f'Seed {seed}')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Grassmann Distance')
        ax.set_title('Subspace Distance Evolution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mixed_effect_multi_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 主角度分布（Final stage）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    final_results = [r for r in mixed_results if 'Final' in r['stage']]
    
    if final_results:
        for i, r in enumerate(final_results):
            if len(r['principal_angles_degrees']) > 0:
                angles = r['principal_angles_degrees'][:10]
                ax.plot(range(1, len(angles)+1), angles, 'o-', 
                       label=f"Seed {r['ckpt_0'][1]}", linewidth=2, markersize=6)
        
        ax.set_xlabel('Principal Angle Index')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Principal Angles between 0% and 20% Subspaces (Final Stage)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/principal_angles_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 稳定性测试热图
    stability_results = [r for r in all_results if r['test'] == 'stability']
    if stability_results:
        # 创建相似度矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 按描述分组
        for ax, metric, title in [(ax1, 'col_similarity', 'Column Space Similarity'),
                                  (ax2, 'grassmann_distance', 'Grassmann Distance')]:
            descriptions = list(set(r['description'] for r in stability_results))
            
            for j, desc in enumerate(descriptions):
                desc_results = [r for r in stability_results if r['description'] == desc]
                if desc_results:
                    seeds = sorted(list(set([r['ckpt1'][1] for r in desc_results] + 
                                           [r['ckpt2'][1] for r in desc_results])))
                    
                    matrix = np.ones((len(seeds), len(seeds)))
                    for r in desc_results:
                        i1 = seeds.index(r['ckpt1'][1])
                        i2 = seeds.index(r['ckpt2'][1])
                        matrix[i1, i2] = r[metric]
                        matrix[i2, i1] = r[metric]
                    
                    im = ax.imshow(matrix, cmap='RdBu_r' if metric == 'col_similarity' else 'viridis',
                                  vmin=0, vmax=1 if metric == 'col_similarity' else None)
                    ax.set_xticks(range(len(seeds)))
                    ax.set_yticks(range(len(seeds)))
                    ax.set_xticklabels(seeds)
                    ax.set_yticklabels(seeds)
                    ax.set_title(f"{title}\n{desc}")
                    
                    # 添加数值标注
                    for i in range(len(seeds)):
                        for j in range(len(seeds)):
                            if i != j:
                                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/stability_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved!")

def save_detailed_results(all_results):
    """保存详细结果到多个CSV文件"""
    if not all_results:
        print("No results to save!")
        return
    
    # 1. 主结果文件
    main_rows = []
    for r in all_results:
        row = {
            'test_type': r['test'],
            'row_similarity': r['row_similarity'],
            'col_similarity': r['col_similarity'],
            'effective_new_dims': r['effective_new_dims'],
            'col_diff_rank': r['col_diff_rank'],
            'col_diff_frobenius': r['col_diff_frobenius'],
            'coverage_2in1': r['coverage_2in1'],
            'coverage_1in2': r['coverage_1in2'],
            'grassmann_distance': r['grassmann_distance'],
            'spectral_distance': r['spectral_distance'],
            'er1': r.get('er1', np.nan),
            'er2': r.get('er2', np.nan),
            'er_diff': r.get('er_diff', np.nan),
        }
        
        if r['test'] == 'stability':
            row.update({
                'description': r['description'],
                'seed1': r['ckpt1'][1],
                'iter1': r['ckpt1'][2],
                'seed2': r['ckpt2'][1],
                'iter2': r['ckpt2'][2],
            })
        elif r['test'] == 'forgetting':
            row.update({
                'ratio': r['ratio'],
                'seed': r['seed'],
                'iter_initial': r['iter_initial'],
                'iter_current': r['iter_current'],
            })
        elif r['test'] == 'mixed_effect':
            row.update({
                'stage': r['stage'],
                'seed': r['ckpt_0'][1],
                'iter_0': r['ckpt_0'][2],
                'iter_20': r['ckpt_20'][2],
            })
        
        main_rows.append(row)
    
    df_main = pd.DataFrame(main_rows)
    df_main.to_csv(f"{OUTPUT_DIR}/similarity_analysis_main.csv", index=False)
    
    # 2. 混合效果详细分析
    mixed_results = [r for r in all_results if r['test'] == 'mixed_effect']
    if mixed_results:
        mixed_df = pd.DataFrame(mixed_results)
        mixed_df.to_csv(f"{OUTPUT_DIR}/mixed_effect_detailed.csv", index=False)
    
    # 3. 汇总统计
    summary = []
    
    # 稳定性测试汇总
    stability_results = [r for r in all_results if r['test'] == 'stability']
    if stability_results:
        for desc in set(r['description'] for r in stability_results):
            desc_results = [r for r in stability_results if r['description'] == desc]
            summary.append({
                'analysis': 'stability',
                'condition': desc,
                'col_similarity_mean': np.mean([r['col_similarity'] for r in desc_results]),
                'col_similarity_std': np.std([r['col_similarity'] for r in desc_results]),
                'grassmann_distance_mean': np.mean([r['grassmann_distance'] for r in desc_results]),
                'n_comparisons': len(desc_results)
            })
    
    # 混合效果汇总
    if mixed_results:
        for stage in ['Initial', 'Middle', 'Final']:
            stage_results = [r for r in mixed_results if stage in r['stage']]
            if stage_results:
                summary.append({
                    'analysis': 'mixed_effect',
                    'condition': stage,
                    'col_similarity_mean': np.mean([r['col_similarity'] for r in stage_results]),
                    'col_similarity_std': np.std([r['col_similarity'] for r in stage_results]),
                    'effective_new_dims_mean': np.mean([r['effective_new_dims'] for r in stage_results]),
                    'effective_new_dims_std': np.std([r['effective_new_dims'] for r in stage_results]),
                    'coverage_1in2_mean': np.mean([r['coverage_1in2'] for r in stage_results]),
                    'er_diff_mean': np.mean([r['er_diff'] for r in stage_results]),
                    'n_comparisons': len(stage_results)
                })
    
    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(f"{OUTPUT_DIR}/analysis_summary.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  - similarity_analysis_main.csv: All results")
    print(f"  - mixed_effect_detailed.csv: Detailed mixed effect analysis")
    print(f"  - analysis_summary.csv: Summary statistics")
    
    # 打印关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if mixed_results:
        final_results = [r for r in mixed_results if 'Final' in r['stage']]
        if final_results:
            print("\nFinal Stage (50k iterations) - 0% vs 20% comparison:")
            for r in final_results:
                print(f"\nSeed {r['ckpt_0'][1]}:")
                print(f"  Column similarity: {r['col_similarity']:.4f}")
                print(f"  Coverage (0% in 20%): {r['coverage_1in2']:.4f}")
                print(f"  Effective new dims: {r['effective_new_dims']}")
                print(f"  ER difference: {r['er_diff']:.2f}")
                print(f"  Grassmann distance: {r['grassmann_distance']:.4f}")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("Enhanced Weight Matrix Similarity Analysis")
    print("="*80)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # 验证关键checkpoints
    print("\nVerifying key checkpoints...")
    test_configs = [(0, 42, 3000), (20, 42, 3000), (0, 42, 50000), (20, 42, 50000)]
    all_found = True
    
    for ratio, seed, iter in test_configs:
        try:
            path = get_checkpoint_path(ratio, seed, iter)
            print(f"✓ Found: mix{ratio}_seed{seed}_iter{iter}")
        except Exception as e:
            print(f"✗ Missing: mix{ratio}_seed{seed}_iter{iter}")
            all_found = False
    
    if not all_found:
        print("\nWarning: Some checkpoints are missing. Results may be incomplete.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    all_results = []
    
    # 运行三个测试
    tests = [
        ("Stability Test", run_stability_test),
        ("Forgetting Analysis", run_forgetting_analysis),
        ("Mixed Effect Analysis", run_mixed_effect_analysis),
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            results = test_func()
            all_results.extend(results)
            print(f"{test_name} completed: {len(results)} comparisons")
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存和可视化
    if all_results:
        save_detailed_results(all_results)
        visualize_results(all_results)
        print(f"\n✓ Analysis complete! Total comparisons: {len(all_results)}")
        print(f"✓ Check {OUTPUT_DIR}/ for detailed results.")
    else:
        print("\n✗ No results generated. Please check your checkpoints.")

if __name__ == "__main__":
    main()