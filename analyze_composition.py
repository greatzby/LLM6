import glob
import re
import pandas as pd
from collections import defaultdict

def parse_logs_improved():
    """解析日志并生成更详细的数据"""
    
    success_pattern = re.compile(r"S1->S3: (\d+\.\d+)%")
    iter_pattern = re.compile(r"Iteration (\d+)")
    
    data = []
    
    for log_file in glob.glob("logs/sweep_training/train_mix*_seed*.log"):
        match = re.search(r"train_mix(\d+)_seed(\d+)\.log", log_file)
        if not match:
            continue
            
        ratio = int(match.group(1))
        seed = int(match.group(2))
        
        with open(log_file, 'r', encoding='utf-8') as f:
            current_iter = 0
            for line in f:
                # 获取迭代次数
                iter_match = iter_pattern.search(line)
                if iter_match:
                    current_iter = int(iter_match.group(1))
                
                # 获取成功率
                if "S1->S3:" in line:
                    m = success_pattern.search(line)
                    if m:
                        success_rate = float(m.group(1)) / 100
                        data.append({
                            'ratio': ratio,
                            'seed': seed,
                            'iter': current_iter,
                            'success': success_rate
                        })
    
    return pd.DataFrame(data)

def analyze_convergence(df):
    """分析收敛后的性能"""
    
    # 1. 过滤掉启动期
    df_converged = df[df['iter'] >= 5000].copy()
    
    print("=== 收敛后的组合成功率统计 ===")
    print("（仅统计 iter >= 5000 的数据）\n")
    
    # 2. 按混合比例分组统计
    stats = df_converged.groupby('ratio')['success'].agg([
        'min', 'max', 'mean', 'std', 'count'
    ]).round(3)
    
    # 3. 计算达到80%阈值的比例
    threshold_stats = df_converged.groupby('ratio').apply(
        lambda x: (x['success'] >= 0.8).sum() / len(x) * 100
    ).round(1)
    
    stats['above_80%'] = threshold_stats
    
    print(stats)
    
    # 4. 最后10次checkpoint的平均值
    print("\n=== 最后10次checkpoint的平均成功率 ===")
    last_10_stats = df.groupby(['ratio', 'seed']).apply(
        lambda x: x.nlargest(10, 'iter')['success'].mean()
    ).groupby('ratio').agg(['mean', 'std']).round(3)
    
    print(last_10_stats)
    
    return df_converged

# 主程序
if __name__ == "__main__":
    # 1. 解析日志
    print("正在解析日志文件...")
    df = parse_logs_improved()
    df.to_csv("success_log_detailed.csv", index=False)
    print(f"已保存 {len(df)} 条记录到 success_log_detailed.csv")
    
    # 2. 分析收敛性能
    df_converged = analyze_convergence(df)
    
    # 3. 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 成功率随迭代的变化
        for ratio in sorted(df['ratio'].unique()):
            data = df[df['ratio'] == ratio]
            avg = data.groupby('iter')['success'].mean()
            ax1.plot(avg.index, avg.values, label=f'mix{ratio}%', alpha=0.8)
        
        ax1.axvline(x=5000, color='red', linestyle='--', alpha=0.5, label='iter=5000')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Success Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 收敛后的分布
        df_converged.boxplot(column='success', by='ratio', ax=ax2)
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_xlabel('Mixing Ratio (%)')
        ax2.set_ylabel('Success Rate (iter >= 5000)')
        ax2.set_title('Success Rate Distribution After Convergence')
        
        plt.tight_layout()
        plt.savefig('composition_analysis.png', dpi=150)
        print("\n已保存图表到 composition_analysis.png")
        
    except ImportError:
        print("\n（未安装matplotlib，跳过可视化）")