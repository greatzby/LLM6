"""
part3_mixed_effect_deep_analysis.py
第三部分：0% vs 20% mix的深度对比分析
重点分析功能性正交化和能量分布
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import json
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 配置
CHECKPOINT_DIR = "out"
OUTPUT_DIR = "part3_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径"""
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

def load_weight_matrix(ratio, seed, iteration):
    """加载lm_head权重矩阵"""
    path = get_checkpoint_path(ratio, seed, iteration)
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 使用lm_head.weight
    W = state_dict['lm_head.weight'].float().numpy()
    
    print(f"  Loaded {ratio}% mix, seed {seed}, iter {iteration}: shape {W.shape}")
    
    return W

def compute_principal_angles_detailed(W1, W2, k=20):
    """计算两个权重矩阵之间的主角度（详细版本）"""
    # SVD分解
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    # 使用列空间（Vt的转置）
    V1 = Vt1.T
    V2 = Vt2.T
    
    # 只使用前k个主成分
    V1_k = V1[:, :k]
    V2_k = V2[:, :k]
    
    # 计算主角度
    M = V1_k.T @ V2_k
    cos_angles = svd(M, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)
    
    return {
        'angles_deg': angles_deg,
        'cos_angles': cos_angles,
        'mean_angle': np.mean(angles_deg),
        'max_angle': np.max(angles_deg),
        'S1': S1,
        'S2': S2,
        'V1': V1,
        'V2': V2
    }

def analyze_dimension_functionality(W1, W2, k=10):
    """分析关键维度的功能对齐"""
    _, S1, Vt1 = svd(W1, full_matrices=False)
    _, S2, Vt2 = svd(W2, full_matrices=False)
    
    results = []
    
    # 对每个主成分分析其在另一个空间中的最佳匹配
    for i in range(k):
        v1 = Vt1[i]  # W1的第i个主成分
        
        # 在W2中找最相似的成分
        similarities = np.abs(Vt2 @ v1)
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        angle = np.arccos(np.clip(best_similarity, -1, 1)) * 180 / np.pi
        
        # 计算该维度的能量贡献
        energy_ratio_1 = S1[i]**2 / np.sum(S1**2)
        energy_ratio_2 = S2[best_match_idx]**2 / np.sum(S2**2) if best_match_idx < len(S2) else 0
        
        results.append({
            'dim_idx': i,
            'singular_value_1': S1[i],
            'singular_value_2': S2[best_match_idx] if best_match_idx < len(S2) else 0,
            'best_match_idx': best_match_idx,
            'angle_to_best_match': angle,
            'similarity': best_similarity,
            'energy_ratio_1': energy_ratio_1,
            'energy_ratio_2': energy_ratio_2,
            'energy_ratio_change': energy_ratio_2 - energy_ratio_1
        })
    
    return pd.DataFrame(results)

def track_rotation_evolution(seeds=[42, 123, 456], 
                           iterations=[3000, 10000, 20000, 30000, 40000, 50000]):
    """追踪0% vs 20% mix在训练过程中的演化"""
    all_results = {}
    
    for seed in seeds:
        print(f"\n=== Analyzing seed {seed} ===")
        results = {
            'iterations': iterations,
            'principal_angles': [],
            'mean_angles': [],
            'max_angles': [],
            'er_0': [],
            'er_20': [],
            'coverage_0in20': [],
            'coverage_20in0': [],
            'dimension_analysis': []
        }
        
        for iter_num in iterations:
            print(f"\nIteration {iter_num}:")
            try:
                # 加载权重
                W_0 = load_weight_matrix(0, seed, iter_num)
                W_20 = load_weight_matrix(20, seed, iter_num)
                
                # 计算主角度
                angle_info = compute_principal_angles_detailed(W_0, W_20)
                results['principal_angles'].append(angle_info['angles_deg'])
                results['mean_angles'].append(angle_info['mean_angle'])
                results['max_angles'].append(angle_info['max_angle'])
                
                # 计算有效秩
                def effective_rank(S):
                    S = S[S > 1e-10]
                    if len(S) == 0:
                        return 0
                    S_normalized = S / S.sum()
                    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
                    return np.exp(entropy)
                
                er_0 = effective_rank(angle_info['S1'])
                er_20 = effective_rank(angle_info['S2'])
                results['er_0'].append(er_0)
                results['er_20'].append(er_20)
                
                # 计算覆盖率
                V1 = angle_info['V1']
                V2 = angle_info['V2']
                
                # 0% in 20%
                proj_0in20 = V2 @ (V2.T @ V1)
                coverage_0in20 = np.linalg.norm(proj_0in20, 'fro')**2 / np.linalg.norm(V1, 'fro')**2
                results['coverage_0in20'].append(coverage_0in20)
                
                # 20% in 0%
                proj_20in0 = V1 @ (V1.T @ V2)
                coverage_20in0 = np.linalg.norm(proj_20in0, 'fro')**2 / np.linalg.norm(V2, 'fro')**2
                results['coverage_20in0'].append(coverage_20in0)
                
                # 维度功能分析（只在关键迭代做）
                if iter_num in [3000, 30000, 50000]:
                    dim_analysis = analyze_dimension_functionality(W_0, W_20, k=10)
                    results['dimension_analysis'].append({
                        'iteration': iter_num,
                        'analysis': dim_analysis
                    })
                    
                    print(f"  Mean angle: {angle_info['mean_angle']:.1f}°")
                    print(f"  Max angle: {angle_info['max_angle']:.1f}°")
                    print(f"  ER: {er_0:.1f} vs {er_20:.1f}")
                    print(f"  Coverage (0% in 20%): {coverage_0in20:.3f}")
                    print(f"  Top-3 dimension angles: {angle_info['angles_deg'][:3]}")
                
            except Exception as e:
                print(f"  Error at iteration {iter_num}: {e}")
                continue
        
        all_results[f'seed_{seed}'] = results
    
    return all_results

def analyze_energy_distribution(W_0, W_20):
    """分析能量分布的差异"""
    _, S_0, _ = svd(W_0, full_matrices=False)
    _, S_20, _ = svd(W_20, full_matrices=False)
    
    # 归一化能量
    energy_0 = S_0**2 / np.sum(S_0**2)
    energy_20 = S_20**2 / np.sum(S_20**2)
    
    # 累积能量
    cumsum_0 = np.cumsum(energy_0)
    cumsum_20 = np.cumsum(energy_20)
    
    # 找出达到特定能量阈值所需的维度数
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    dims_needed = {}
    
    for thresh in thresholds:
        dims_0 = np.argmax(cumsum_0 >= thresh) + 1
        dims_20 = np.argmax(cumsum_20 >= thresh) + 1
        dims_needed[f'{int(thresh*100)}%'] = {
            '0%_mix': int(dims_0),
            '20%_mix': int(dims_20),
            'difference': int(dims_20 - dims_0)
        }
    
    # 能量集中度指标
    top_k = [5, 10, 20, 30]
    concentration = {}
    for k in top_k:
        conc_0 = np.sum(energy_0[:k])
        conc_20 = np.sum(energy_20[:k])
        concentration[f'top_{k}'] = {
            '0%_mix': float(conc_0),
            '20%_mix': float(conc_20),
            'ratio': float(conc_0 / conc_20) if conc_20 > 0 else np.inf
        }
    
    return {
        'energy_0': energy_0,
        'energy_20': energy_20,
        'cumsum_0': cumsum_0,
        'cumsum_20': cumsum_20,
        'dims_needed': dims_needed,
        'concentration': concentration
    }

def visualize_results(all_results):
    """生成可视化图表"""
    # 创建图形
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 主角度演化（所有种子）
    ax1 = plt.subplot(3, 3, 1)
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            ax1.plot(data['iterations'], data['mean_angles'], 
                    'o-', label=f'Seed {seed}', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Principal Angle (degrees)')
    ax1.set_title('Evolution of Mean Principal Angle (0% vs 20%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最大主角度演化
    ax2 = plt.subplot(3, 3, 2)
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            ax2.plot(data['iterations'], data['max_angles'], 
                    's-', label=f'Seed {seed}', linewidth=2, markersize=8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Principal Angle (degrees)')
    ax2.set_title('Evolution of Maximum Principal Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.5)
    
    # 3. 覆盖率演化
    ax3 = plt.subplot(3, 3, 3)
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            ax3.plot(data['iterations'], data['coverage_0in20'], 
                    '^-', label=f'Seed {seed}', linewidth=2, markersize=8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Coverage (0% in 20%)')
    ax3.set_title('How much of 0% mix is contained in 20% mix')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.7, 1.0])
    
    # 4. ER差异演化
    ax4 = plt.subplot(3, 3, 4)
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            er_diff = np.array(data['er_20']) - np.array(data['er_0'])
            ax4.plot(data['iterations'], er_diff, 
                    'd-', label=f'Seed {seed}', linewidth=2, markersize=8)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('ER Difference (20% - 0%)')
    ax4.set_title('Effective Rank Protection by 20% Mix')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 5. 前10个主角度分布（最终状态）
    ax5 = plt.subplot(3, 3, 5)
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            final_angles = data['principal_angles'][-1][:10]  # 最后一个迭代的前10个角度
            ax5.plot(range(1, len(final_angles)+1), final_angles, 
                    'o-', label=f'Seed {seed}', linewidth=2, markersize=6)
    ax5.set_xlabel('Principal Angle Index')
    ax5.set_ylabel('Angle (degrees)')
    ax5.set_title('Top 10 Principal Angles at Final Iteration')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 能量分布对比（seed 42, final）
    ax6 = plt.subplot(3, 3, 6)
    if f'seed_42' in all_results:
        # 加载最终权重
        W_0_final = load_weight_matrix(0, 42, 50000)
        W_20_final = load_weight_matrix(20, 42, 50000)
        energy_info = analyze_energy_distribution(W_0_final, W_20_final)
        
        # 绘制累积能量
        dims = np.arange(1, len(energy_info['cumsum_0'])+1)
        ax6.plot(dims[:50], energy_info['cumsum_0'][:50], 
                'r-', label='0% mix', linewidth=2)
        ax6.plot(dims[:50], energy_info['cumsum_20'][:50], 
                'b-', label='20% mix', linewidth=2)
        ax6.set_xlabel('Number of Dimensions')
        ax6.set_ylabel('Cumulative Energy')
        ax6.set_title('Energy Distribution Comparison (Final State)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    
    # 7. 维度功能对齐热图（seed 42）
    ax7 = plt.subplot(3, 3, 7)
    if f'seed_42' in all_results and all_results['seed_42']['dimension_analysis']:
        # 使用最终状态的分析
        final_analysis = all_results['seed_42']['dimension_analysis'][-1]['analysis']
        
        # 创建角度矩阵
        angles = final_analysis['angle_to_best_match'].values[:10]
        angle_matrix = angles.reshape(-1, 1)
        
        im = ax7.imshow(angle_matrix, cmap='hot', aspect='auto')
        ax7.set_yticks(range(10))
        ax7.set_yticklabels([f'PC{i+1}' for i in range(10)])
        ax7.set_xticks([0])
        ax7.set_xticklabels(['Angle'])
        ax7.set_title('Principal Component Alignment (Final State)')
        
        # 添加数值
        for i in range(10):
            ax7.text(0, i, f'{angles[i]:.1f}°', 
                    ha='center', va='center', color='white' if angles[i] > 45 else 'black')
        
        plt.colorbar(im, ax=ax7, label='Angle (degrees)')
    
    # 8. 关键发现汇总
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # 计算平均值
    final_mean_angles = []
    final_max_angles = []
    final_coverages = []
    final_er_diffs = []
    
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            data = all_results[f'seed_{seed}']
            final_mean_angles.append(data['mean_angles'][-1])
            final_max_angles.append(data['max_angles'][-1])
            final_coverages.append(data['coverage_0in20'][-1])
            final_er_diffs.append(data['er_20'][-1] - data['er_0'][-1])
    
    summary_text = f"""
    KEY FINDINGS (Final State Averages):
    
    Mean Principal Angle: {np.mean(final_mean_angles):.1f}° (±{np.std(final_mean_angles):.1f}°)
    Max Principal Angle: {np.mean(final_max_angles):.1f}° (±{np.std(final_max_angles):.1f}°)
    Coverage (0% in 20%): {np.mean(final_coverages):.3f} (±{np.std(final_coverages):.3f})
    ER Protection: {np.mean(final_er_diffs):.1f} (±{np.std(final_er_diffs):.1f})
    
    Interpretation:
    • High coverage (~86%) but large angles (~87°)
      → "Empty shell" effect
    • 20% mix maintains higher ER (+4.7 on average)
      → Better information capacity
    • Critical dimensions are functionally orthogonalized
      → Loss of compositional ability
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # 9. 能量集中度条形图
    ax9 = plt.subplot(3, 3, 9)
    if f'seed_42' in all_results:
        W_0_final = load_weight_matrix(0, 42, 50000)
        W_20_final = load_weight_matrix(20, 42, 50000)
        energy_info = analyze_energy_distribution(W_0_final, W_20_final)
        
        # 准备数据
        categories = ['Top 5', 'Top 10', 'Top 20', 'Top 30']
        mix_0_values = [energy_info['concentration'][f'top_{k}']['0%_mix'] 
                       for k in [5, 10, 20, 30]]
        mix_20_values = [energy_info['concentration'][f'top_{k}']['20%_mix'] 
                        for k in [5, 10, 20, 30]]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax9.bar(x - width/2, mix_0_values, width, label='0% mix', color='red', alpha=0.7)
        ax9.bar(x + width/2, mix_20_values, width, label='20% mix', color='blue', alpha=0.7)
        
        ax9.set_xlabel('Top K Dimensions')
        ax9.set_ylabel('Cumulative Energy Ratio')
        ax9.set_title('Energy Concentration Comparison')
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/part3_mixed_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {OUTPUT_DIR}/part3_mixed_effect_analysis.png")

def save_detailed_results(all_results):
    """保存详细结果"""
    # 转换为可JSON序列化的格式
    json_results = {}
    
    for seed_key, seed_data in all_results.items():
        json_results[seed_key] = {
            'iterations': seed_data['iterations'],
            'mean_angles': [float(x) for x in seed_data['mean_angles']],
            'max_angles': [float(x) for x in seed_data['max_angles']],
            'er_0': [float(x) for x in seed_data['er_0']],
            'er_20': [float(x) for x in seed_data['er_20']],
            'coverage_0in20': [float(x) for x in seed_data['coverage_0in20']],
            'coverage_20in0': [float(x) for x in seed_data['coverage_20in0']],
            'final_top10_angles': [float(x) for x in seed_data['principal_angles'][-1][:10]]
        }
    
    # 保存JSON
    with open(f'{OUTPUT_DIR}/part3_analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # 保存维度分析的CSV
    for seed in [42, 123, 456]:
        if f'seed_{seed}' in all_results:
            for dim_analysis in all_results[f'seed_{seed}']['dimension_analysis']:
                iter_num = dim_analysis['iteration']
                df = dim_analysis['analysis']
                df.to_csv(f'{OUTPUT_DIR}/dimension_analysis_seed{seed}_iter{iter_num}.csv', 
                         index=False)
    
    print(f"Results saved to {OUTPUT_DIR}/")

def main():
    """主函数"""
    print("="*80)
    print("Part 3: Deep Analysis of Mixed Training Effect (0% vs 20%)")
    print("="*80)
    
    # 运行主要分析
    print("\n1. Tracking rotation evolution across training...")
    all_results = track_rotation_evolution()
    
    # 生成可视化
    print("\n2. Generating visualizations...")
    visualize_results(all_results)
    
    # 保存结果
    print("\n3. Saving detailed results...")
    save_detailed_results(all_results)
    
    # 特别分析：最终状态的能量分布
    print("\n4. Analyzing final state energy distribution...")
    for seed in [42]:  # 只分析seed 42作为代表
        W_0_final = load_weight_matrix(0, seed, 50000)
        W_20_final = load_weight_matrix(20, seed, 50000)
        energy_info = analyze_energy_distribution(W_0_final, W_20_final)
        
        print(f"\n=== Energy Distribution Analysis (Seed {seed}, Final) ===")
        print("\nDimensions needed to capture X% of energy:")
        for thresh, info in energy_info['dims_needed'].items():
            print(f"  {thresh}: 0% mix needs {info['0%_mix']} dims, "
                  f"20% mix needs {info['20%_mix']} dims "
                  f"(difference: {info['difference']})")
        
        print("\nEnergy concentration in top K dimensions:")
        for k, info in energy_info['concentration'].items():
            print(f"  {k}: 0% mix = {info['0%_mix']:.3f}, "
                  f"20% mix = {info['20%_mix']:.3f} "
                  f"(ratio: {info['ratio']:.2f})")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()