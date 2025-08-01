"""
visualize_all_92_dimensions.py
可视化所有92个维度的变化，深入理解模型差异
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os

CHECKPOINT_DIR = "out_d92"
OUTPUT_DIR = "all_92_dims_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_checkpoint_path(ratio, seed, iteration):
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    return checkpoint_path

def load_and_analyze_all_dimensions(seed=42, iteration=50000):
    """加载并分析所有92个维度"""
    # 加载权重
    path_0 = get_checkpoint_path(0, seed, iteration)
    path_20 = get_checkpoint_path(20, seed, iteration)
    
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    W0 = state_0['lm_head.weight'].float().numpy()
    W20 = state_20['lm_head.weight'].float().numpy()
    
    # SVD分解
    U0, S0, V0 = np.linalg.svd(W0, full_matrices=False)
    U20, S20, V20 = np.linalg.svd(W20, full_matrices=False)
    
    # 分析所有92个维度
    dimension_analysis = []
    
    for i in range(92):
        # 1. 角度
        cos_angle = np.abs(np.dot(V0[i, :], V20[i, :]))
        angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        # 2. 能量变化
        energy_change = S20[i] - S0[i]
        relative_change = energy_change / (S0[i] + 1e-10)
        
        # 3. V向量的模式变化
        v_diff = V20[i, :] - V0[i, :]
        v_diff_norm = np.linalg.norm(v_diff)
        
        # 4. 该维度在哪些token上最活跃（前5个）
        top_tokens_0 = np.argsort(np.abs(V0[i, :]))[-5:][::-1]
        top_tokens_20 = np.argsort(np.abs(V20[i, :]))[-5:][::-1]
        
        dimension_analysis.append({
            'dim': i,
            'angle_deg': angle_deg,
            'singular_value_0': S0[i],
            'singular_value_20': S20[i],
            'energy_change': energy_change,
            'relative_change': relative_change,
            'v_diff_norm': v_diff_norm,
            'top_tokens_0': top_tokens_0.tolist(),
            'top_tokens_20': top_tokens_20.tolist(),
            'is_critical': angle_deg > 45 and relative_change > 0.1  # 关键维度
        })
    
    return dimension_analysis, V0, V20, S0, S20

def create_comprehensive_visualization(dim_analysis, V0, V20, S0, S20):
    """创建全面的可视化"""
    # 创建大型仪表板
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 所有92个维度的角度变化（条形图）
    ax1 = plt.subplot(4, 2, 1)
    angles = [d['angle_deg'] for d in dim_analysis]
    colors = ['red' if d['is_critical'] else 'blue' for d in dim_analysis]
    bars = ax1.bar(range(92), angles, color=colors, alpha=0.6)
    ax1.axhline(45, color='red', linestyle='--', label='45° threshold')
    ax1.set_xlabel('Dimension Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Principal Angles for All 92 Dimensions')
    ax1.legend()
    
    # 添加标注最大的几个
    top_angles_idx = np.argsort(angles)[-5:]
    for idx in top_angles_idx:
        ax1.text(idx, angles[idx]+1, f'{idx}', ha='center', fontsize=8)
    
    # 2. 能量（奇异值）变化
    ax2 = plt.subplot(4, 2, 2)
    relative_changes = [d['relative_change'] for d in dim_analysis]
    bars = ax2.bar(range(92), relative_changes, color=colors, alpha=0.6)
    ax2.axhline(0.1, color='orange', linestyle='--', label='10% threshold')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Dimension Index')
    ax2.set_ylabel('Relative Energy Change')
    ax2.set_title('Energy Change for All 92 Dimensions')
    ax2.legend()
    
    # 3. 2D散点图：角度 vs 能量变化
    ax3 = plt.subplot(4, 2, 3)
    scatter = ax3.scatter(angles, relative_changes, 
                         c=range(92), cmap='viridis', s=50, alpha=0.6)
    
    # 标注关键维度
    for d in dim_analysis:
        if d['is_critical']:
            ax3.annotate(f"{d['dim']}", 
                        (d['angle_deg'], d['relative_change']),
                        fontsize=9, color='red')
    
    ax3.axvline(45, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(0.1, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Relative Energy Change')
    ax3.set_title('Angle vs Energy Change (Critical Dims in Red)')
    
    # 4. 奇异值谱
    ax4 = plt.subplot(4, 2, 4)
    ax4.semilogy(S0, 'b-', label='0% model', linewidth=2)
    ax4.semilogy(S20, 'r-', label='20% model', linewidth=2)
    ax4.set_xlabel('Dimension Index')
    ax4.set_ylabel('Singular Value (log scale)')
    ax4.set_title('Singular Value Spectrum')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 热力图：V矩阵的差异
    ax5 = plt.subplot(4, 2, 5)
    V_diff = V20 - V0
    # 只显示前30个维度和前30个token位置
    im = ax5.imshow(V_diff[:30, :30], cmap='RdBu_r', aspect='auto')
    ax5.set_xlabel('Token Position')
    ax5.set_ylabel('Dimension')
    ax5.set_title('V Matrix Difference (first 30x30)')
    plt.colorbar(im, ax=ax5)
    
    # 6. 维度分组统计
    ax6 = plt.subplot(4, 2, 6)
    # 将维度分组
    groups = {
        'Stable (<20°)': sum(1 for d in dim_analysis if d['angle_deg'] < 20),
        'Mild (20-45°)': sum(1 for d in dim_analysis if 20 <= d['angle_deg'] < 45),
        'Critical (>45°)': sum(1 for d in dim_analysis if d['angle_deg'] >= 45)
    }
    ax6.pie(groups.values(), labels=groups.keys(), autopct='%1.1f%%', startangle=90)
    ax6.set_title('Dimension Distribution by Angle')
    
    # 7. 关键维度的详细信息
    ax7 = plt.subplot(4, 1, 4)
    ax7.axis('off')
    
    critical_dims = [d for d in dim_analysis if d['is_critical']]
    info_text = f"CRITICAL DIMENSIONS ANALYSIS\n"
    info_text += f"{'='*50}\n"
    info_text += f"Total critical dimensions: {len(critical_dims)}\n"
    info_text += f"Dimensions: {[d['dim'] for d in critical_dims]}\n\n"
    
    if critical_dims:
        info_text += "Top 5 Critical Dimensions:\n"
        for d in sorted(critical_dims, key=lambda x: x['angle_deg'], reverse=True)[:5]:
            info_text += f"\nDim {d['dim']:2d}: "
            info_text += f"Angle={d['angle_deg']:.1f}°, "
            info_text += f"Energy change={d['relative_change']:.2%}"
    
    ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/all_92_dimensions_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建第二个图：维度轨迹
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 角度分布直方图
    ax = axes[0, 0]
    ax.hist(angles, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(angles), color='red', linestyle='--', 
              label=f'Mean: {np.mean(angles):.1f}°')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dimension Angles')
    ax.legend()
    
    # 能量变化分布
    ax = axes[0, 1]
    ax.hist(relative_changes, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(0, color='black', linestyle='-')
    ax.axvline(np.mean(relative_changes), color='red', linestyle='--',
              label=f'Mean: {np.mean(relative_changes):.2%}')
    ax.set_xlabel('Relative Energy Change')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Energy Changes')
    ax.legend()
    
    # 累积分布
    ax = axes[1, 0]
    sorted_angles = sorted(angles)
    cumulative = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles) * 100
    ax.plot(sorted_angles, cumulative, 'b-', linewidth=2)
    ax.axvline(20, color='green', linestyle=':', label='20°')
    ax.axvline(45, color='red', linestyle=':', label='45°')
    ax.axhline(80, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Cumulative Percentage')
    ax.set_title('Cumulative Distribution of Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 维度相关性
    ax = axes[1, 1]
    # 计算一些有趣的相关性
    dims = list(range(92))
    ax.scatter(dims, [d['singular_value_0'] for d in dim_analysis], 
              alpha=0.5, label='0% model', s=30)
    ax.scatter(dims, [d['singular_value_20'] for d in dim_analysis], 
              alpha=0.5, label='20% model', s=30)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values by Dimension')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dimension_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_detailed_report(dim_analysis):
    """保存详细报告"""
    # 保存为JSON
    with open(f'{OUTPUT_DIR}/all_dimensions_report.json', 'w') as f:
        json.dump(dim_analysis, f, indent=2)
    
    # 创建Markdown报告
    with open(f'{OUTPUT_DIR}/dimension_analysis_report.md', 'w') as f:
        f.write("# 92-Dimensional Analysis Report\n\n")
        
        # 统计摘要
        angles = [d['angle_deg'] for d in dim_analysis]
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total dimensions: 92\n")
        f.write(f"- Mean angle: {np.mean(angles):.1f}°\n")
        f.write(f"- Max angle: {np.max(angles):.1f}°\n")
        f.write(f"- Dimensions with angle > 45°: {sum(a > 45 for a in angles)}\n")
        f.write(f"- Critical dimensions (angle>45° AND energy↑>10%): {sum(d['is_critical'] for d in dim_analysis)}\n\n")
        
        # 关键维度详情
        f.write("## Critical Dimensions\n\n")
        critical = sorted([d for d in dim_analysis if d['is_critical']], 
                         key=lambda x: x['angle_deg'], reverse=True)
        
        for d in critical:
            f.write(f"### Dimension {d['dim']}\n")
            f.write(f"- Angle: {d['angle_deg']:.1f}°\n")
            f.write(f"- Energy change: {d['relative_change']:.2%}\n")
            f.write(f"- Singular values: {d['singular_value_0']:.3f} → {d['singular_value_20']:.3f}\n\n")

def main():
    print("="*80)
    print("Analyzing All 92 Dimensions")
    print("="*80)
    
    # 分析种子42的数据（你可以改成循环分析所有种子）
    dim_analysis, V0, V20, S0, S20 = load_and_analyze_all_dimensions(seed=42)
    
    # 打印快速摘要
    angles = [d['angle_deg'] for d in dim_analysis]
    critical_dims = [d['dim'] for d in dim_analysis if d['is_critical']]
    
    print(f"\nQuick Summary:")
    print(f"- Angle range: [{min(angles):.1f}°, {max(angles):.1f}°]")
    print(f"- Mean angle: {np.mean(angles):.1f}°")
    print(f"- Dimensions > 45°: {sum(a > 45 for a in angles)}")
    print(f"- Critical dimensions: {critical_dims}")
    
    # 创建可视化
    print("\nCreating visualizations...")
    create_comprehensive_visualization(dim_analysis, V0, V20, S0, S20)
    
    # 保存报告
    print("Saving detailed report...")
    save_detailed_report(dim_analysis)
    
    print(f"\nAnalysis complete! Check {OUTPUT_DIR}/ for results")

if __name__ == "__main__":
    main()