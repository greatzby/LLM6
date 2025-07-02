# analyze_mixed_models_comparison.py
import os
import sys
from analyze_composition_degradation import CompositionAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

def compare_three_models():
    """对比分析原始、5%混合、10%混合三个模型"""
    
    # 定义三个模型的路径
    models = {
        'Original': {
            'checkpoint_dir': 'out/composition_20250702_063926',
            'data_dir': 'data/simple_graph/composition_90'
        },
        '5% Mixed': {
            'checkpoint_dir': 'out/composition_20250703_004537',  # 根据你的实际路径修改
            'data_dir': 'data/simple_graph/composition_90_mixed_5'
        },
        '10% Mixed': {
            'checkpoint_dir': 'out/composition_20250703_011304',  # 根据你的实际路径修改
            'data_dir': 'data/simple_graph/composition_90_mixed_10'
        }
    }
    
    # 分析每个模型
    all_model_results = {}
    
    for model_name, paths in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name} Model")
        print(f"{'='*60}")
        
        try:
            analyzer = CompositionAnalyzer(
                checkpoint_dir=paths['checkpoint_dir'],
                data_dir=paths['data_dir']
            )
            
            # 分析所有checkpoints
            results = analyzer.analyze_all_checkpoints()
            all_model_results[model_name] = results
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue
    
    # 创建对比图
    create_comparison_plots(all_model_results)
    
    # 保存对比报告
    save_comparison_report(all_model_results)

def create_comparison_plots(all_model_results):
    """创建对比图表"""
    
    # 准备数据
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. S1->S3准确率对比
    ax1 = axes[0, 0]
    
    for model_name, results in all_model_results.items():
        iterations = sorted(results.keys())
        s1s3_accs = [results[it]['S1->S3']['accuracy'] for it in iterations]
        
        style = {'Original': 'r-o', '5% Mixed': 'b-s', '10% Mixed': 'g-^'}
        ax1.plot(iterations, s1s3_accs, style[model_name], 
                label=model_name, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('S1->S3 Accuracy')
    ax1.set_title('Composition Ability Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 2. 稳定性分析（标准差）
    ax2 = axes[0, 1]
    
    stability_data = []
    model_names = []
    
    for model_name, results in all_model_results.items():
        iterations = sorted(results.keys())
        s1s3_accs = [results[it]['S1->S3']['accuracy'] for it in iterations]
        
        # 计算后半段的标准差（作为稳定性指标）
        late_stage_accs = s1s3_accs[len(s1s3_accs)//2:]
        stability = np.std(late_stage_accs)
        
        stability_data.append(stability)
        model_names.append(model_name)
    
    bars = ax2.bar(model_names, stability_data, color=['red', 'blue', 'green'])
    ax2.set_ylabel('Standard Deviation (Lower is Better)')
    ax2.set_title('S1->S3 Performance Stability')
    
    # 添加数值标签
    for bar, val in zip(bars, stability_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. 失败模式对比（最后一个checkpoint）
    ax3 = axes[1, 0]
    
    error_types = ['invalid_edge', 'wrong_target', 'too_short', 'no_s2', 'other']
    bar_width = 0.25
    x = np.arange(len(error_types))
    
    for i, (model_name, results) in enumerate(all_model_results.items()):
        last_iter = max(results.keys())
        failures = results[last_iter]['S1->S3']['failures']
        
        error_counts = {et: 0 for et in error_types}
        for failure in failures:
            error = failure['error']
            if 'invalid_edge' in error:
                error_counts['invalid_edge'] += 1
            elif 'wrong_target' in error:
                error_counts['wrong_target'] += 1
            elif 'too_short' in error:
                error_counts['too_short'] += 1
            elif 'no_s2' in error:
                error_counts['no_s2'] += 1
            else:
                error_counts['other'] += 1
        
        counts = [error_counts[et] for et in error_types]
        ax3.bar(x + i*bar_width, counts, bar_width, 
               label=model_name, alpha=0.8)
    
    ax3.set_xlabel('Error Type')
    ax3.set_ylabel('Count')
    ax3.set_title('Failure Mode Distribution (Final Checkpoint)')
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(error_types, rotation=45)
    ax3.legend()
    
    # 4. 性能演化热图
    ax4 = axes[1, 1]
    
    # 创建性能矩阵
    model_names_list = list(all_model_results.keys())
    iterations = sorted(next(iter(all_model_results.values())).keys())
    
    perf_matrix = []
    for model_name in model_names_list:
        if model_name in all_model_results:
            results = all_model_results[model_name]
            row = [results[it]['S1->S3']['accuracy'] for it in iterations]
            perf_matrix.append(row)
    
    im = ax4.imshow(perf_matrix, aspect='auto', cmap='RdYlGn')
    ax4.set_yticks(range(len(model_names_list)))
    ax4.set_yticklabels(model_names_list)
    ax4.set_xticks(range(len(iterations)))
    ax4.set_xticklabels([f'{it//1000}k' for it in iterations])
    ax4.set_xlabel('Iteration')
    ax4.set_title('S1->S3 Performance Heatmap')
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('mixed_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_comparison_report(all_model_results):
    """保存对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'mixed_models_comparison_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MIXED TRAINING COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 总结表格
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<15} {'Initial S1->S3':<15} {'Peak S1->S3':<15} {'Final S1->S3':<15} {'Stability':<15}\n")
        f.write("-"*80 + "\n")
        
        for model_name, results in all_model_results.items():
            iterations = sorted(results.keys())
            s1s3_accs = [results[it]['S1->S3']['accuracy'] for it in iterations]
            
            initial = s1s3_accs[0]
            peak = max(s1s3_accs)
            final = s1s3_accs[-1]
            
            # 稳定性：后半段的标准差
            late_stage = s1s3_accs[len(s1s3_accs)//2:]
            stability = np.std(late_stage)
            
            f.write(f"{model_name:<15} {initial:<15.2%} {peak:<15.2%} {final:<15.2%} {stability:<15.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED ANALYSIS:\n")
        f.write("="*80 + "\n")
        
        # 详细分析每个模型
        for model_name, results in all_model_results.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")
            f.write("-"*40 + "\n")
            
            iterations = sorted(results.keys())
            
            # 找出关键点
            s1s3_accs = {it: results[it]['S1->S3']['accuracy'] for it in iterations}
            best_iter = max(s1s3_accs, key=s1s3_accs.get)
            worst_iter = min(s1s3_accs, key=s1s3_accs.get)
            
            f.write(f"Best Performance: {s1s3_accs[best_iter]:.2%} at iteration {best_iter}\n")
            f.write(f"Worst Performance: {s1s3_accs[worst_iter]:.2%} at iteration {worst_iter}\n")
            f.write(f"Performance Range: {s1s3_accs[best_iter] - s1s3_accs[worst_iter]:.2%}\n")
            
            # 分析最后的失败模式
            last_iter = iterations[-1]
            failures = results[last_iter]['S1->S3']['failures']
            
            if failures:
                f.write(f"\nFailure Analysis at iteration {last_iter}:\n")
                error_counts = defaultdict(int)
                for failure in failures:
                    error_type = failure['error'].split('_')[0]
                    error_counts[error_type] += 1
                
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {error}: {count}\n")
        
        print(f"\nComparison report saved to: {report_file}")

if __name__ == "__main__":
    compare_three_models()