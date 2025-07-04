# run_sweep_training.py
#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime
import argparse

def run_training(ratio, seed, gpu_id=0):
    """运行单个训练任务"""
    
    # 确定数据目录
    if ratio == 0:
        data_dir = "data/simple_graph/composition_90"  # 特殊处理0%
    else:
        data_dir = f"data/simple_graph/composition_90_mixed_{ratio}"
    
    # 日志目录
    log_dir = "logs/sweep_training"
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件
    log_file = os.path.join(log_dir, f"train_mix{ratio}_seed{seed}.log")
    
    # 构建命令
    cmd = [
        "python", "train_composition_sweep.py",
        "--data_dir", data_dir,
        "--mixing_ratio", str(ratio),
        "--seed", str(seed),
        "--max_iters", "50000",
        "--test_interval", "1000",
        "--checkpoint_interval", "1000",
        "--device", f"cuda:{gpu_id}",
        "--log_file", log_file
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting: Mixing {ratio}%, Seed {seed}")
    print(f"Data: {data_dir}")
    print(f"Log: {log_file}")
    print(f"GPU: cuda:{gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # 运行
    start_time = time.time()
    
    try:
        # 使用subprocess运行，将输出同时写到文件和终端
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     universal_newlines=True, bufsize=1)
            
            for line in process.stdout:
                print(line, end='')  # 打印到终端
                f.write(line)  # 写入日志文件
                f.flush()
            
            process.wait()
            
        elapsed = time.time() - start_time
        print(f"\n✓ Completed: Mix {ratio}%, Seed {seed} (Time: {elapsed/60:.1f} min)")
        
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed: Mix {ratio}%, Seed {seed} - {str(e)}")
        return False, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dry_run', action='store_true', help='Just print commands without running')
    parser.add_argument('--start_from', type=int, default=None, 
                       help='Start from specific mixing ratio (useful for resuming)')
    parser.add_argument('--only_ratio', type=int, default=None,
                       help='Only run specific mixing ratio')
    parser.add_argument('--only_seed', type=int, default=None,
                       help='Only run specific seed')
    args = parser.parse_args()
    
    # 配置
    MIXING_RATIOS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    RANDOM_SEEDS = [42, 123, 456]
    
    # 过滤
    if args.only_ratio is not None:
        MIXING_RATIOS = [args.only_ratio]
    elif args.start_from is not None:
        MIXING_RATIOS = [r for r in MIXING_RATIOS if r >= args.start_from]
    
    if args.only_seed is not None:
        RANDOM_SEEDS = [args.only_seed]
    
    # 总任务数
    total_tasks = len(MIXING_RATIOS) * len(RANDOM_SEEDS)
    
    print("="*80)
    print("Composition Training Sweep")
    print("="*80)
    print(f"Mixing Ratios: {MIXING_RATIOS}")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    print(f"Total Tasks: {total_tasks}")
    print(f"GPU: cuda:{args.gpu}")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN MODE - Not executing]")
        for ratio in MIXING_RATIOS:
            for seed in RANDOM_SEEDS:
                if ratio == 0:
                    data_dir = "data/simple_graph/composition_90"
                else:
                    data_dir = f"data/simple_graph/composition_90_mixed_{ratio}"
                print(f"Would run: Mix {ratio}%, Seed {seed} from {data_dir}")
        return
    
    # 确认
    response = input(f"\nProceed with {total_tasks} training runs? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # 开始时间
    sweep_start = time.time()
    
    # 记录结果
    results = []
    completed = 0
    failed = 0
    
    # 运行所有任务
    for ratio in MIXING_RATIOS:
        for seed in RANDOM_SEEDS:
            completed += 1
            print(f"\n{'#'*80}")
            print(f"Task {completed}/{total_tasks}: Mixing {ratio}%, Seed {seed}")
            print(f"{'#'*80}")
            
            success, elapsed = run_training(ratio, seed, args.gpu)
            
            results.append({
                'ratio': ratio,
                'seed': seed,
                'success': success,
                'time_min': elapsed / 60
            })
            
            if not success:
                failed += 1
    
    # 总结
    total_time = time.time() - sweep_start
    
    print("\n" + "="*80)
    print("SWEEP COMPLETED")
    print("="*80)
    print(f"Total Time: {total_time/3600:.1f} hours")
    print(f"Successful: {completed - failed}/{total_tasks}")
    print(f"Failed: {failed}/{total_tasks}")
    
    # 打印结果表格
    print("\nDetailed Results:")
    print("-"*60)
    print(f"{'Ratio':>6} {'Seed':>6} {'Status':>10} {'Time (min)':>12}")
    print("-"*60)
    
    for res in results:
        status = "SUCCESS" if res['success'] else "FAILED"
        print(f"{res['ratio']:>6} {res['seed']:>6} {status:>10} {res['time_min']:>12.1f}")
    
    # 保存结果摘要
    summary_file = f"logs/sweep_training/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Sweep Summary\n")
        f.write(f"="*60 + "\n")
        f.write(f"Total Time: {total_time/3600:.1f} hours\n")
        f.write(f"Tasks: {completed}/{total_tasks}\n")
        f.write(f"Failed: {failed}\n\n")
        
        f.write("Results:\n")
        for res in results:
            f.write(f"Mix {res['ratio']}%, Seed {res['seed']}: "
                   f"{'SUCCESS' if res['success'] else 'FAILED'} "
                   f"({res['time_min']:.1f} min)\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()