# compute_brc_star.py - 最小修改版本
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pickle

def analyze_brc_star():
    """使用新的BRC*公式重新分析"""
    
    # 加载之前保存的结果
    with open('brc_analysis_results_fixed.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # 计算新的BRC*
    for name in results:
        if 'sigma_bias' in results[name] and 'direction_diversity' in results[name]:
            results[name]['BRC_star'] = []
            for σ, div in zip(results[name]['sigma_bias'], results[name]['direction_diversity']):
                BRC_star = σ * (1 - div)
                results[name]['BRC_star'].append(BRC_star)
    
    # 打印新的表格
    print("\n" + "="*100)
    print("BRC* Analysis (σ_bias × (1 - direction_diversity))")
    print("="*100)
    
    from tabulate import tabulate
    table_data = []
    
    for name in ['original', '5% mixed', '10% mixed']:
        for i, iter_k in enumerate(results[name]['iters']):
            table_data.append([
                name,
                f"{iter_k/1000:.0f}k",
                f"{results[name]['sigma_bias'][i]:.3f}",
                f"{results[name]['direction_diversity'][i]:.3f}",
                f"{results[name]['BRC_star'][i]:.3f}",
                f"{results[name]['success_rate'][i]:.2f}"
            ])
    
    headers = ["Model", "Iter", "σ_bias", "Dir_Div", "BRC*", "Success"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 相关性分析
    all_brc_star = []
    all_success = []
    
    for name in results:
        all_brc_star.extend(results[name]['BRC_star'])
        all_success.extend(results[name]['success_rate'])
    
    pearson_r, pearson_p = pearsonr(all_brc_star, all_success)
    spearman_r, spearman_p = spearmanr(all_brc_star, all_success)
    
    print(f"\nBRC* vs Success Rate Correlation:")
    print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.3e})")
    print(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.3e})")
    
    # 绘制关键图表
    plot_brc_star_analysis(results)
    
    return results

def plot_brc_star_analysis(results):
    """绘制BRC*分析的核心图表"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'original': '#e74c3c', '5% mixed': '#f39c12', '10% mixed': '#27ae60'}
    
    # 1. BRC* 演化
    ax1 = axes[0, 0]
    for name in ['original', '5% mixed', '10% mixed']:
        iters_k = [x/1000 for x in results[name]['iters']]
        ax1.plot(iters_k, results[name]['BRC_star'], 
                marker='o', label=name, linewidth=2.5, 
                markersize=8, color=colors[name])
    
    ax1.set_xlabel('Iteration (k)', fontsize=12)
    ax1.set_ylabel('BRC* = σ × (1 - dir_div)', fontsize=12)
    ax1.set_title('BRC* Index Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. BRC* vs 成功率散点图
    ax2 = axes[0, 1]
    for name in ['original', '5% mixed', '10% mixed']:
        ax2.scatter(results[name]['BRC_star'], results[name]['success_rate'],
                   label=name, color=colors[name], s=100, alpha=0.7, edgecolors='black')
    
    # 添加趋势线
    all_brc = []
    all_success = []
    for name in results:
        all_brc.extend(results[name]['BRC_star'])
        all_success.extend(results[name]['success_rate'])
    
    z = np.polyfit(all_brc, all_success, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_brc), max(all_brc), 100)
    ax2.plot(x_trend, p(x_trend), "k--", alpha=0.5, label=f'Linear fit')
    
    ax2.set_xlabel('BRC* Index', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title('BRC* vs Success Rate', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 方向多样性演化
    ax3 = axes[1, 0]
    for name in ['original', '5% mixed', '10% mixed']:
        iters_k = [x/1000 for x in results[name]['iters']]
        ax3.plot(iters_k, results[name]['direction_diversity'],
                marker='^', label=name, linewidth=2.5,
                markersize=8, color=colors[name])
    
    ax3.set_xlabel('Iteration (k)', fontsize=12)
    ax3.set_ylabel('Direction Diversity', fontsize=12)
    ax3.set_title('S2 Direction Diversity (↑ = more dispersed)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终指标对比条形图
    ax4 = axes[1, 1]
    model_names = ['original', '5% mixed', '10% mixed']
    final_brc_star = [results[name]['BRC_star'][-1] for name in model_names]
    final_success = [results[name]['success_rate'][-1] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_brc_star, width, 
                     label='BRC*', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, final_success, width, 
                     label='Success Rate', color='#e74c3c', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Final Metrics at 50k iterations', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend(fontsize=11)
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('brc_star_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_components(results):
    """单独分析各个组件与成功率的关系"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_sigma = []
    all_diversity = []
    all_success = []
    
    for name in results:
        all_sigma.extend(results[name]['sigma_bias'])
        all_diversity.extend(results[name]['direction_diversity'])
        all_success.extend(results[name]['success_rate'])
    
    # 1. σ_bias vs Success
    ax1 = axes[0]
    ax1.scatter(all_sigma, all_success, alpha=0.6, color='#3498db')
    r1, p1 = pearsonr(all_sigma, all_success)
    ax1.set_xlabel('σ_bias')
    ax1.set_ylabel('Success Rate')
    ax1.set_title(f'σ_bias vs Success (r={r1:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Direction Diversity vs Success
    ax2 = axes[1]
    ax2.scatter(all_diversity, all_success, alpha=0.6, color='#e74c3c')
    r2, p2 = pearsonr(all_diversity, all_success)
    ax2.set_xlabel('Direction Diversity')
    ax2.set_ylabel('Success Rate')
    ax2.set_title(f'Dir Diversity vs Success (r={r2:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. (1-Dir Diversity) vs Success
    ax3 = axes[2]
    ax3.scatter(1-np.array(all_diversity), all_success, alpha=0.6, color='#27ae60')
    r3, p3 = pearsonr(1-np.array(all_diversity), all_success)
    ax3.set_xlabel('1 - Direction Diversity')
    ax3.set_ylabel('Success Rate')
    ax3.set_title(f'(1-Dir Div) vs Success (r={r3:.3f})')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('component_analysis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 运行新的分析
    results = analyze_brc_star()
    
    # 额外分析：各组件单独的相关性
    plot_individual_components(results)