#!/usr/bin/env python3
"""
解析训练日志，提取组合成功率
"""
import re
import csv
import glob
from pathlib import Path
from tqdm import tqdm

def parse_logs():
    log_pattern = "logs/sweep_training/train_mix*_seed*.log"
    iter_pattern = re.compile(r"Iteration (\d+)\s*\|\s*Mix (\d+)%\s*\|\s*Seed (\d+)")
    success_pattern = re.compile(r"S1->S3:\s*(\d+\.\d+)%\s*\((\d+)/50\)")
    
    rows = []
    
    for log_file in tqdm(sorted(glob.glob(log_pattern)), desc="Parsing logs"):
        filename = Path(log_file).name
        file_match = re.search(r"train_mix(\d+)_seed(\d+)\.log", filename)
        if not file_match:
            continue
            
        file_ratio = int(file_match.group(1))
        file_seed = int(file_match.group(2))
        
        with open(log_file, 'r', encoding='utf-8') as f:
            current_iter = None
            for line in f:
                iter_match = iter_pattern.search(line)
                if iter_match:
                    current_iter = int(iter_match.group(1))
                    current_ratio = int(iter_match.group(2))
                    current_seed = int(iter_match.group(3))
                    continue
                
                if "S1->S3:" in line and current_iter is not None:
                    success_match = success_pattern.search(line)
                    if success_match:
                        success_pct = float(success_match.group(1))
                        success_count = int(success_match.group(2))
                        
                        rows.append({
                            'ratio': current_ratio,
                            'seed': current_seed,
                            'iter': current_iter,
                            'success': success_pct / 100.0,
                            'success_count': success_count
                        })
    
    with open("success_log.csv", "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"✓ Saved success_log.csv ({len(rows)} rows)")
    return rows

if __name__ == "__main__":
    parse_logs()