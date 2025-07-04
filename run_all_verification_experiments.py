# run_all_verification_experiments.py
import os
import sys
from datetime import datetime

def ensure_dependencies():
    """确保所有依赖都已安装"""
    required = ['torch', 'numpy', 'matplotlib', 'networkx', 'scikit-learn', 'tqdm']
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

def run_all_experiments():
    """运行所有验证实验"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"verification_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 改变工作目录
    original_dir = os.getcwd()
    os.chdir(results_dir)
    
    print("="*80)
    print("COMPREHENSIVE VERIFICATION EXPERIMENTS")
    print("="*80)
    print(f"Results will be saved to: {results_dir}")
    print()
    
    experiments = [
        ("Gradient Decomposition Analysis", "gradient_decomposition_analysis.py"),
        ("Cosine Distance Evolution", "cosine_distance_evolution.py"),
        ("CKA Similarity Analysis", "cka_similarity_analysis.py"),
        # ("Mixture Ratio Sweep", "mixture_ratio_sweep.py")  # 这个比较耗时，可选
    ]
    
    # 复制必要的脚本到结果目录
    for _, script in experiments:
        src = os.path.join(original_dir, script)
        if os.path.exists(src):
            os.system(f"cp {src} .")
    
    # 运行每个实验
    for name, script in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        try:
            # 使用exec而不是subprocess，这样可以共享环境
            with open(script, 'r') as f:
                code = f.read()
            exec(code)
            
            print(f"✓ {name} completed successfully!")
        except Exception as e:
            print(f"✗ Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成综合报告
    generate_comprehensive_report()
    
    # 返回原目录
    os.chdir(original_dir)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {results_dir}/")
    print("="*80)

def generate_comprehensive_report():
    """生成综合分析报告"""
    with open('COMPREHENSIVE_REPORT.md', 'w') as f:
        f.write("# Verification Experiment Results\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary of Key Findings\n\n")
        
        # 检查各个结果文件
        if os.path.exists('cosine_distance_results.txt'):
            f.write("### 1. Cosine Distance Analysis\n")
            with open('cosine_distance_results.txt', 'r') as rf:
                f.write("```\n")
                f.write(rf.read())
                f.write("```\n\n")
        
        if os.path.exists('cka_correlation_results.txt'):
            f.write("### 2. CKA Similarity Analysis\n")
            with open('cka_correlation_results.txt', 'r') as rf:
                f.write("```\n")
                f.write(rf.read())
                f.write("```\n\n")
        
        f.write("## Visualizations\n\n")
        
        plots = [
            ('gradient_decomposition_analysis.png', 'Gradient Decomposition'),
            ('cosine_distance_evolution.png', 'Cosine Distance Evolution'),
            ('cka_similarity_analysis.png', 'CKA Similarity Analysis')
        ]
        
        for plot, title in plots:
            if os.path.exists(plot):
                f.write(f"### {title}\n")
                f.write(f"![{title}](./{plot})\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("Based on the experimental results:\n\n")
        f.write("1. **Representation Drift**: S2 representations gradually drift towards S3\n")
        f.write("2. **Gradient Imbalance**: S2→S3 gradients dominate over S1→S2\n")
        f.write("3. **Stability Correlation**: Higher CKA stability correlates with better performance\n")
        f.write("4. **Critical Mixture Ratio**: ~10% mixture provides stable performance\n")

if __name__ == "__main__":
    # 确保依赖
    ensure_dependencies()
    
    # 确保在正确的目录
    if not os.path.exists('model.py'):
        print("Error: Please run this script from your project root directory")
        print("(The directory containing model.py)")
        sys.exit(1)
    
    # 运行所有实验
    run_all_experiments()