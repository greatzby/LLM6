# recompute_with_real_success.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc
import pickle

# 真实的成功率数据
real_success_rates = {
    'original': {
        5000: 0.86, 10000: 0.74, 15000: 0.68, 20000: 0.72, 25000: 0.70,
        30000: 0.60, 35000: 0.40, 40000: 0.42, 45000: 0.42, 50000: 0.32
    },
    '5% mixed': {
        5000: 0.80, 10000: 0.76, 15000: 0.62, 20000: 0.70, 25000: 0.72,
        30000: 0.82, 35000: 0.84, 40000: 0.82, 45000: 0.84, 50000: 0.86
    },
    '10% mixed': {
        5000: 0.96, 10000: 0.86, 15000: 0.88, 20000: 0.82, 25000: 0.88,
        30000: 0.90, 35000: 0.90, 40000: 0.90, 45000: 0.90, 50000: 0.90
    }
}

def recompute_correlations():
    """使用真实成功率重新计算相关性"""
    
    # 加载之前的BRC结果
    with open('brc_analysis_results_fixed.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # 更新成功率
    for model_name in real_success_rates:
        success_list = []
        for iter_num in results[model_name]['iters']:
            success_list.append(real_success_rates[model_name][iter_num])
        results[model_name]['success_rate'] = success_list
    
    # 计算BRC*
    for name in results:
        results[name]['BRC_star'] = []
        for σ, div in zip(results[name]['sigma_bias'], results[name]['direction_diversity']):
            BRC_star = σ * (1 - div)
            results[name]['BRC_star'].append(BRC_star)
    
    # 收集所有数据点
    all_brc_star = []
    all_success = []
    
    for name in results:
        all_brc_star.extend(results[name]['BRC_star'])
        all_success.extend(results[name]['success_rate'])
    
    # 计算相关性
    r_pearson, p_pearson = pearsonr(all_brc_star, all_success)
    r_spearman, p_spearman = spearmanr(all_brc_star, all_success)
    
    print("="*60)
    print("Correlation Analysis with Real Success Rates")
    print("="*60)
    print(f"BRC* vs Success Rate:")
    print(f"  Pearson r = {r_pearson:.3f} (p = {p_pearson:.3e})")
    print(f"  Spearman r = {r_spearman:.3f} (p = {p_spearman:.3e})")
    
    # 绘制完整分析图
    plot_complete_analysis(results)
    
    return results

def plot_complete_analysis(results):
    """绘制完整的分析图表"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 15))
    
    colors = {'original': '#e74c3c', '5% mixed': '#f39c12', '10% mixed': '#27ae60'}
    
    # 1. 成功率演化（真实数据）
    ax1 = plt.subplot(3, 3, 1)
    for name in ['original', '5% mixed', '10% mixed']:
        iters_k = [x/1000 for x in results[name]['iters']]
        ax1.plot(iters_k, results[name]['success_rate'], 
                marker='o', label=name, linewidth=3, 
                markersize=10, color=colors[name])
    ax1.set_xlabel('Iteration (k)', fontsize=14)
    ax1.set_ylabel('S1→S3 Success Rate', fontsize=14)
    ax1.set_title('Real Success Rate Evolution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 2. BRC*演化
    ax2 = plt.subplot(3, 3, 2)
    for name in ['original', '5% mixed', '10% mixed']:
        iters_k = [x/1000 for x in results[name]['iters']]
        ax2.plot(iters_k, results[name]['BRC_star'], 
                marker='s', label=name, linewidth=3, 
                markersize=10, color=colors[name])
    ax2.set_xlabel('Iteration (k)', fontsize=14)
    ax2.set_ylabel('BRC* Index', fontsize=14)
    ax2.set_title('BRC* Evolution', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. BRC* vs Success散点图
    ax3 = plt.subplot(3, 3, 3)
    for name in ['original', '5% mixed', '10% mixed']:
        ax3.scatter(results[name]['BRC_star'], results[name]['success_rate'],
                   label=name, color=colors[name], s=150, alpha=0.8, 
                   edgecolors='black', linewidth=2)
    
    # 添加趋势线
    all_brc = []
    all_success = []
    for name in results:
        all_brc.extend(results[name]['BRC_star'])
        all_success.extend(results[name]['success_rate'])
    
    z = np.polyfit(all_brc, all_success, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_brc), max(all_brc), 100)
    ax3.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2, label='Linear fit')
    
    ax3.set_xlabel('BRC* Index', fontsize=14)
    ax3.set_ylabel('Success Rate', fontsize=14)
    ax3.set_title('BRC* vs Success Rate Correlation', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 打印更详细的统计
    print("\n" + "="*60)
    print("Detailed Statistics by Model")
    print("="*60)
    
    for name in ['original', '5% mixed', '10% mixed']:
        brc_values = results[name]['BRC_star']
        success_values = results[name]['success_rate']
        
        print(f"\n{name}:")
        print(f"  BRC* range: [{min(brc_values):.3f}, {max(brc_values):.3f}]")
        print(f"  Success range: [{min(success_values):.2f}, {max(success_values):.2f}]")
        print(f"  Final BRC*: {brc_values[-1]:.3f}")
        print(f"  Final Success: {success_values[-1]:.2f}")
    
    plt.tight_layout()
    plt.savefig('brc_real_success_analysis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    results = recompute_correlations()