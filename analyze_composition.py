import glob
import re
from collections import defaultdict

def analyze_composition_logs():
    """分析组合训练日志，统计各混合比例的成功率分布"""
    
    # 用于匹配S1->S3成功率的正则表达式
    success_pattern = re.compile(r"S1->S3: (\d+\.\d+)%")
    
    # 存储数据
    data = defaultdict(list)
    
    # 读取所有日志文件
    log_files = glob.glob("logs/sweep_training/train_mix*_seed*.log")
    
    if not log_files:
        print("未找到日志文件！请确保在正确的目录下运行脚本。")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    for log_file in log_files:
        # 从文件名提取混合比例
        match = re.search(r"train_mix(\d+)_seed(\d+)\.log", log_file)
        if not match:
            continue
            
        ratio = int(match.group(1))
        seed = int(match.group(2))
        
        print(f"正在处理: mix{ratio}% seed{seed}")
        
        # 读取文件内容
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 查找包含S1->S3成功率的行
                if "S1->S3:" in line:
                    m = success_pattern.search(line)
                    if m:
                        success_rate = float(m.group(1)) / 100
                        data[ratio].append(success_rate)
    
    # 输出统计结果
    print("\n=== 组合成功率统计 ===")
    print("混合比例  最小值  最大值  平均值  样本数")
    print("-" * 45)
    
    for ratio in sorted(data.keys()):
        values = data[ratio]
        if values:
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values) / len(values)
            count = len(values)
            print(f"mix{ratio:2d}%    {min_val:.2f}    {max_val:.2f}    {mean_val:.2f}    {count:4d}")
    
    # 更详细的分析
    print("\n=== 详细分析 ===")
    for ratio in sorted(data.keys()):
        values = data[ratio]
        if values:
            # 计算达到80%阈值的比例
            above_threshold = sum(1 for v in values if v >= 0.80)
            threshold_rate = above_threshold / len(values) * 100
            print(f"\nmix{ratio}%:")
            print(f"  达到80%阈值的比例: {threshold_rate:.1f}%")
            print(f"  最后10次的平均值: {sum(values[-10:]) / len(values[-10:]):.2f}")

if __name__ == "__main__":
    analyze_composition_logs()