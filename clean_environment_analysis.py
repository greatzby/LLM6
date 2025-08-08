#!/usr/bin/env python3
"""
clean_environment_analysis.py

用于判定实验的"无菌环境"（Clean Environment）。

核心功能：
1. 计算累计能量阈值下的k值（k@90%, k@95%, k@97%）
2. 基于能量阈值的k计算子空间重合度
3. 计算能谱差异（JS散度）
4. 计算能量加权相似度
5. 综合判定随机种子是否会干扰mix差异的结论

判定标准：
- 跨种子（同配比）的方差应该小
- 跨配比（同种子）的差异应该大
- 这样才能证明随机种子不会混淆mix效应
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import pandas as pd

# ============ 配置参数 ============
SEEDS = [42, 123, 456]
RATIOS = [0, 20]
D_MODEL = 92
CHECK_ITERATIONS = [10000, 20000, 30000, 40000, 50000]  # 检查多个训练阶段
ENERGY_THRESHOLDS = [0.90, 0.95, 0.97]  # 能量累计阈值

# ============ 工具函数 ============

def get_checkpoint_path(d_model, ratio, seed, iteration):
    """获取检查点路径"""
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    if not os.path.exists(path):
        return None
    return path

def load_weight_matrix(d_model, ratio, seed, iteration):
    """加载权重矩阵"""
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None:
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
        'total_energy': total_energy,
        'cumulative_energy': cumulative_energy
    }

def find_k_for_energy_threshold(cumulative_energy, threshold):
    """找到累计能量达到阈值的最小k"""
    k = np.argmax(cumulative_energy >= threshold) + 1
    return min(k, len(cumulative_energy))

def compute_effective_rank(S):
    """计算有效秩"""
    S_positive = S[S > 1e-9]
    if len(S_positive) == 0:
        return 0
    p = S_positive**2 / np.sum(S_positive**2)
    H = -np.sum(p * np.log(p + 1e-10))
    return np.exp(H)

# ============ 核心分析函数 ============

def compute_subspace_overlap(svd1, svd2, k, space='V'):
    """
    计算子空间重合度 (basis-invariant)
    sim_proj = trace(P1 * P2) / k
    """
    if space == 'V':
        V1 = svd1['Vt'][:k, :].T
        V2 = svd2['Vt'][:k, :].T
    else:  # U space
        V1 = svd1['U'][:, :k]
        V2 = svd2['U'][:, :k]
    
    # 计算投影矩阵的迹
    overlap = V1.T @ V2
    singular_values = svd(overlap, compute_uv=False)
    sim_proj = np.mean(singular_values**2)  # 平均cos²
    return sim_proj

def compute_spectrum_difference(svd1, svd2, k):
    """
    计算能谱差异（JS散度）
    """
    # 归一化top-k奇异值为概率分布
    s1 = svd1['S'][:k]**2
    s2 = svd2['S'][:k]**2
    p1 = s1 / np.sum(s1)
    p2 = s2 / np.sum(s2)
    
    # JS散度
    js_div = jensenshannon(p1, p2)**2  # 平方使其在[0,1]范围
    
    # L2距离作为补充
    l2_dist = np.linalg.norm(p1 - p2)
    
    return {'js_divergence': js_div, 'l2_distance': l2_dist}

def compute_energy_weighted_similarity(svd1, svd2, k, space='V'):
    """
    计算能量加权相似度
    sim_E = Σ_i (σ1_i * σ2_i * |<v1_i, v2_i>|²) / sqrt(Σσ1² * Σσ2²)
    """
    if space == 'V':
        V1 = svd1['Vt'][:k, :].T
        V2 = svd2['Vt'][:k, :].T
    else:
        V1 = svd1['U'][:, :k]
        V2 = svd2['U'][:, :k]
    
    s1 = svd1['S'][:k]
    s2 = svd2['S'][:k]
    
    # 计算每对基向量的内积
    dot_products = np.abs(np.diag(V1.T @ V2))**2
    
    # 能量加权
    weighted_sim = np.sum(s1 * s2 * dot_products) / np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    
    return weighted_sim

# ============ 主分析流程 ============

def analyze_clean_environment():
    """主分析函数"""
    
    print("="*80)
    print("          无菌环境（Clean Environment）综合分析")
    print("="*80)
    
    all_results = []
    
    for iteration in CHECK_ITERATIONS:
        print(f"\n{'='*70}")
        print(f"   分析迭代: {iteration}")
        print(f"{'='*70}")
        
        # 1. 加载所有模型并计算SVD
        svd_results = {}
        for ratio in RATIOS:
            for seed in SEEDS:
                print(f"加载模型: ratio={ratio}%, seed={seed}")
                W = load_weight_matrix(D_MODEL, ratio, seed, iteration)
                if W is not None:
                    svd_data = compute_svd_with_energy(W)
                    svd_results[f"r{ratio}_s{seed}"] = svd_data
                    
                    # 报告有效秩和能量阈值k
                    erank = compute_effective_rank(svd_data['S'])
                    print(f"  - 有效秩: {erank:.2f}")
                    for thresh in ENERGY_THRESHOLDS:
                        k_thresh = find_k_for_energy_threshold(svd_data['cumulative_energy'], thresh)
                        print(f"  - k@{int(thresh*100)}%能量: {k_thresh}")
        
        if len(svd_results) < len(RATIOS) * len(SEEDS):
            print("[警告] 部分模型加载失败，跳过此迭代")
            continue
        
        # 2. 使用95%能量阈值确定k
        print(f"\n确定分析用的k值（基于95%能量阈值）:")
        k_values = []
        for key, svd_data in svd_results.items():
            k_95 = find_k_for_energy_threshold(svd_data['cumulative_energy'], 0.95)
            k_values.append(k_95)
            print(f"  {key}: k@95% = {k_95}")
        
        # 使用所有模型的最小k@95%
        k_analysis = min(k_values)
        print(f"\n选定分析k值: {k_analysis} (所有模型的min(k@95%))")
        
        # 3. 计算跨种子（同配比）的指标
        print(f"\n{'='*50}")
        print(f"跨种子比较（同配比）:")
        print(f"{'='*50}")
        
        cross_seed_results = []
        for ratio in RATIOS:
            seed_pairs = list(combinations(SEEDS, 2))
            print(f"\n配比 {ratio}% - 比较 {len(seed_pairs)} 对种子:")
            
            for s1, s2 in seed_pairs:
                key1 = f"r{ratio}_s{s1}"
                key2 = f"r{ratio}_s{s2}"
                
                if key1 in svd_results and key2 in svd_results:
                    # 子空间重合度
                    v_overlap = compute_subspace_overlap(svd_results[key1], svd_results[key2], k_analysis, 'V')
                    u_overlap = compute_subspace_overlap(svd_results[key1], svd_results[key2], k_analysis, 'U')
                    
                    # 能谱差异
                    spectrum_diff = compute_spectrum_difference(svd_results[key1], svd_results[key2], k_analysis)
                    
                    # 能量加权相似度
                    v_weighted = compute_energy_weighted_similarity(svd_results[key1], svd_results[key2], k_analysis, 'V')
                    u_weighted = compute_energy_weighted_similarity(svd_results[key1], svd_results[key2], k_analysis, 'U')
                    
                    result = {
                        'type': 'cross_seed',
                        'ratio': ratio,
                        'seed1': s1,
                        'seed2': s2,
                        'iteration': iteration,
                        'k': k_analysis,
                        'v_overlap': v_overlap,
                        'u_overlap': u_overlap,
                        'js_divergence': spectrum_diff['js_divergence'],
                        'l2_distance': spectrum_diff['l2_distance'],
                        'v_weighted': v_weighted,
                        'u_weighted': u_weighted
                    }
                    cross_seed_results.append(result)
                    all_results.append(result)
                    
                    print(f"  Seed {s1} vs {s2}:")
                    print(f"    - V空间重合度: {v_overlap:.4f}")
                    print(f"    - U空间重合度: {u_overlap:.4f}")
                    print(f"    - 能谱JS散度: {spectrum_diff['js_divergence']:.4f}")
                    print(f"    - V能量加权: {v_weighted:.4f}")
        
        # 4. 计算跨配比（同种子）的指标
        print(f"\n{'='*50}")
        print(f"跨配比比较（同种子）:")
        print(f"{'='*50}")
        
        cross_ratio_results = []
        for seed in SEEDS:
            print(f"\n种子 {seed} - 比较 0% vs 20%:")
            
            key1 = f"r0_s{seed}"
            key2 = f"r20_s{seed}"
            
            if key1 in svd_results and key2 in svd_results:
                # 子空间重合度
                v_overlap = compute_subspace_overlap(svd_results[key1], svd_results[key2], k_analysis, 'V')
                u_overlap = compute_subspace_overlap(svd_results[key1], svd_results[key2], k_analysis, 'U')
                
                # 能谱差异
                spectrum_diff = compute_spectrum_difference(svd_results[key1], svd_results[key2], k_analysis)
                
                # 能量加权相似度
                v_weighted = compute_energy_weighted_similarity(svd_results[key1], svd_results[key2], k_analysis, 'V')
                u_weighted = compute_energy_weighted_similarity(svd_results[key1], svd_results[key2], k_analysis, 'U')
                
                result = {
                    'type': 'cross_ratio',
                    'seed': seed,
                    'ratio1': 0,
                    'ratio2': 20,
                    'iteration': iteration,
                    'k': k_analysis,
                    'v_overlap': v_overlap,
                    'u_overlap': u_overlap,
                    'js_divergence': spectrum_diff['js_divergence'],
                    'l2_distance': spectrum_diff['l2_distance'],
                    'v_weighted': v_weighted,
                    'u_weighted': u_weighted
                }
                cross_ratio_results.append(result)
                all_results.append(result)
                
                print(f"    - V空间重合度: {v_overlap:.4f}")
                print(f"    - U空间重合度: {u_overlap:.4f}")
                print(f"    - 能谱JS散度: {spectrum_diff['js_divergence']:.4f}")
                print(f"    - V能量加权: {v_weighted:.4f}")
        
        # 5. 统计分析
        print(f"\n{'='*50}")
        print(f"统计汇总 (迭代 {iteration}):")
        print(f"{'='*50}")
        
        # 跨种子统计
        if cross_seed_results:
            df_cs = pd.DataFrame(cross_seed_results)
            print("\n跨种子（同配比）统计:")
            for metric in ['v_overlap', 'u_overlap', 'js_divergence', 'v_weighted']:
                mean_val = df_cs[metric].mean()
                std_val = df_cs[metric].std()
                print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 跨配比统计
        if cross_ratio_results:
            df_cr = pd.DataFrame(cross_ratio_results)
            print("\n跨配比（同种子）统计:")
            for metric in ['v_overlap', 'u_overlap', 'js_divergence', 'v_weighted']:
                mean_val = df_cr[metric].mean()
                std_val = df_cr[metric].std()
                print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 6. 最终判定
    print(f"\n{'='*80}")
    print(f"             无菌环境判定结果")
    print(f"{'='*80}")
    
    df_all = pd.DataFrame(all_results)
    
    # 分别计算跨种子和跨配比的统计
    df_cross_seed = df_all[df_all['type'] == 'cross_seed']
    df_cross_ratio = df_all[df_all['type'] == 'cross_ratio']
    
    print("\n关键指标对比:")
    print("-"*60)
    print(f"{'指标':<20} | {'跨种子(同配比)':<20} | {'跨配比(同种子)':<20}")
    print("-"*60)
    
    for metric in ['v_overlap', 'js_divergence', 'v_weighted']:
        cs_mean = df_cross_seed[metric].mean()
        cs_std = df_cross_seed[metric].std()
        cr_mean = df_cross_ratio[metric].mean()
        cr_std = df_cross_ratio[metric].std()
        
        metric_display = {
            'v_overlap': 'V空间重合度',
            'js_divergence': '能谱JS散度',
            'v_weighted': 'V能量加权相似度'
        }[metric]
        
        print(f"{metric_display:<20} | {cs_mean:.3f}±{cs_std:.3f}        | {cr_mean:.3f}±{cr_std:.3f}")
    
    # 判定逻辑
    print("\n" + "="*60)
    print("判定结论:")
    print("="*60)
    
    # V空间重合度判定
    v_cs_mean = df_cross_seed['v_overlap'].mean()
    v_cr_mean = df_cross_ratio['v_overlap'].mean()
    v_ratio = v_cr_mean / v_cs_mean if v_cs_mean > 0 else 0
    
    # 能谱差异判定
    js_cs_mean = df_cross_seed['js_divergence'].mean()
    js_cr_mean = df_cross_ratio['js_divergence'].mean()
    js_ratio = js_cr_mean / js_cs_mean if js_cs_mean > 0 else 0
    
    # 能量加权判定
    vw_cs_mean = df_cross_seed['v_weighted'].mean()
    vw_cr_mean = df_cross_ratio['v_weighted'].mean()
    vw_ratio = vw_cr_mean / vw_cs_mean if vw_cs_mean > 0 else 0
    
    clean_checks = []
    
    # 检查1: V空间几何
    if v_ratio > 1.1:  # 跨配比重合度比跨种子高10%以上
        clean_checks.append("✓ V空间几何: 跨配比差异小于跨种子差异（比率{:.2f}）".format(v_ratio))
    else:
        clean_checks.append("✗ V空间几何: 跨种子差异过大（比率{:.2f}）".format(v_ratio))
    
    # 检查2: 能谱差异
    if js_ratio > 1.5:  # 跨配比JS散度比跨种子大50%以上
        clean_checks.append("✓ 能谱差异: 跨配比差异显著大于跨种子（比率{:.2f}）".format(js_ratio))
    else:
        clean_checks.append("✗ 能谱差异: 差异区分度不足（比率{:.2f}）".format(js_ratio))
    
    # 检查3: 能量加权相似度
    if vw_ratio > 0.9 and vw_ratio < 1.2:  # 比较稳定
        clean_checks.append("✓ 能量加权: 稳定性良好（比率{:.2f}）".format(vw_ratio))
    else:
        clean_checks.append("△ 能量加权: 稳定性一般（比率{:.2f}）".format(vw_ratio))
    
    for check in clean_checks:
        print(f"  {check}")
    
    # 总体判定
    passed_checks = sum(1 for c in clean_checks if c.startswith("✓"))
    
    print("\n" + "="*60)
    if passed_checks >= 2:
        print("🎉 判定: 实验环境基本满足'无菌'条件")
        print("   种子效应不会干扰mix差异的结论")
    else:
        print("⚠️  判定: 实验环境未完全达到'无菌'标准")
        print("   建议增加更多种子或调整实验设计")
    print("="*60)
    
    # 7. 生成可视化
    generate_visualizations(df_all)
    
    return df_all

def generate_visualizations(df):
    """生成可视化图表"""
    
    print("\n生成可视化图表...")
    
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 准备数据
    df_cs = df[df['type'] == 'cross_seed']
    df_cr = df[df['type'] == 'cross_ratio']
    
    metrics = ['v_overlap', 'u_overlap', 'js_divergence', 'v_weighted', 'u_weighted', 'l2_distance']
    titles = ['V-Space Overlap', 'U-Space Overlap', 'Spectrum JS Divergence', 
              'V-Space Energy Weighted', 'U-Space Energy Weighted', 'Spectrum L2 Distance']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # 准备箱线图数据
        data_to_plot = []
        labels = []
        
        if not df_cs.empty:
            data_to_plot.append(df_cs[metric].values)
            labels.append('Cross-Seed\n(Same Mix)')
        
        if not df_cr.empty:
            data_to_plot.append(df_cr[metric].values)
            labels.append('Cross-Mix\n(Same Seed)')
        
        # 绘制箱线图
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加均值线
        for i, data in enumerate(data_to_plot):
            ax.axhline(y=np.mean(data), xmin=(i+0.75)/(len(data_to_plot)+1), 
                      xmax=(i+1.25)/(len(data_to_plot)+1), 
                      color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Clean Environment Analysis - Metric Comparisons', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('clean_environment_analysis.png', dpi=300, bbox_inches='tight')
    print("图表已保存: clean_environment_analysis.png")
    plt.show()
    
    # 生成热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 跨种子相似度矩阵
    if not df_cs.empty:
        pivot_cs = df_cs.pivot_table(values='v_overlap', index='seed1', columns='seed2', aggfunc='mean')
        sns.heatmap(pivot_cs, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0], vmin=0, vmax=1)
        axes[0].set_title('Cross-Seed V-Space Overlap (Same Mix)')
    
    # 跨配比对比
    if not df_cr.empty:
        cr_matrix = df_cr[['v_overlap', 'js_divergence', 'v_weighted']].mean().values.reshape(1, -1)
        sns.heatmap(cr_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1], 
                   xticklabels=['V-Overlap', 'JS-Div', 'V-Weighted'],
                   yticklabels=['0% vs 20%'])
        axes[1].set_title('Cross-Mix Metrics (Same Seed)')
    
    plt.tight_layout()
    plt.savefig('clean_environment_heatmap.png', dpi=300)
    print("热力图已保存: clean_environment_heatmap.png")
    plt.show()

# ============ 主程序 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="无菌环境(Clean Environment)综合分析工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--iterations', 
        nargs='+', 
        type=int,
        default=CHECK_ITERATIONS,
        help='要检查的迭代次数列表 (默认: {})'.format(CHECK_ITERATIONS)
    )
    
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=SEEDS,
        help='要分析的随机种子列表 (默认: {})'.format(SEEDS)
    )
    
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='是否保存详细结果到CSV文件'
    )
    
    args = parser.parse_args()
    
    # 更新全局配置
    if args.iterations:
        CHECK_ITERATIONS = args.iterations
    if args.seeds:
        SEEDS = args.seeds
    
    print("配置信息:")
    print(f"  - 模型维度: {D_MODEL}")
    print(f"  - 混合比例: {RATIOS}%")
    print(f"  - 随机种子: {SEEDS}")
    print(f"  - 检查迭代: {CHECK_ITERATIONS}")
    print(f"  - 能量阈值: {[f'{int(t*100)}%' for t in ENERGY_THRESHOLDS]}")
    print()
    
    # 运行分析
    results_df = analyze_clean_environment()
    
    # 保存结果
    if args.save_csv and results_df is not None:
        csv_filename = 'clean_environment_results.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\n详细结果已保存到: {csv_filename}")
    
    print("\n分析完成！")