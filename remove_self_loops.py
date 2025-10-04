#!/usr/bin/env python3
"""
remove_self_loops.py
从ALPINE strict数据集中删除所有自边（S1->S1, S2->S2, S3->S3）
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

def load_stage_info(stage_info_path):
    """加载stage信息"""
    with open(stage_info_path, 'rb') as f:
        stage_info = pickle.load(f)
    S1, S2, S3 = stage_info['stages']
    return set(S1), set(S2), set(S3)

def process_text_file(input_path, output_path, S1, S2, S3):
    """处理文本格式的训练文件，删除自边"""
    
    print(f"\nProcessing text file: {input_path}")
    
    # 统计信息
    stats = {
        'total': 0,
        'kept': 0,
        'removed': 0,
        'removed_s1s1': 0,
        'removed_s2s2': 0,
        'removed_s3s3': 0,
        'kept_s1s2': 0,
        'kept_s2s3': 0,
        'kept_s1s3': 0
    }
    
    kept_lines = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Filtering self-loops"):
        stats['total'] += 1
        
        # 解析每行数据
        parts = line.strip().split()
        if len(parts) < 4:  # 至少需要source, target, path_start, path_end
            continue
            
        source = int(parts[0])
        target = int(parts[1])
        
        # 检查是否是自边
        is_self_loop = False
        if source in S1 and target in S1:
            stats['removed_s1s1'] += 1
            is_self_loop = True
        elif source in S2 and target in S2:
            stats['removed_s2s2'] += 1
            is_self_loop = True
        elif source in S3 and target in S3:
            stats['removed_s3s3'] += 1
            is_self_loop = True
        
        if is_self_loop:
            stats['removed'] += 1
        else:
            kept_lines.append(line)
            stats['kept'] += 1
            
            # 统计保留的类型
            if source in S1 and target in S2:
                stats['kept_s1s2'] += 1
            elif source in S2 and target in S3:
                stats['kept_s2s3'] += 1
            elif source in S1 and target in S3:
                stats['kept_s1s3'] += 1
    
    # 写入新文件
    with open(output_path, 'w') as f:
        for line in kept_lines:
            f.write(line)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("FILTERING STATISTICS")
    print("="*60)
    print(f"Total samples: {stats['total']}")
    print(f"Removed: {stats['removed']} ({stats['removed']/stats['total']*100:.1f}%)")
    print(f"  - S1->S1: {stats['removed_s1s1']}")
    print(f"  - S2->S2: {stats['removed_s2s2']}")
    print(f"  - S3->S3: {stats['removed_s3s3']}")
    print(f"\nKept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  - S1->S2: {stats['kept_s1s2']} ({stats['kept_s1s2']/stats['kept']*100:.1f}%)")
    print(f"  - S2->S3: {stats['kept_s2s3']} ({stats['kept_s2s3']/stats['kept']*100:.1f}%)")
    print(f"  - S1->S3: {stats['kept_s1s3']} ({stats['kept_s1s3']/stats['kept']*100:.1f}%)")
    print("="*60)
    
    return stats

def process_binary_file(input_path, output_path, S1, S2, S3):
    """处理二进制格式的训练文件，删除自边"""
    
    print(f"\nProcessing binary file: {input_path}")
    
    # 加载二进制数据
    data = np.fromfile(input_path, dtype=np.uint16)
    
    # 统计信息
    stats = {
        'total': 0,
        'kept': 0,
        'removed': 0,
        'removed_s1s1': 0,
        'removed_s2s2': 0,
        'removed_s3s3': 0,
        'kept_s1s2': 0,
        'kept_s2s3': 0,
        'kept_s1s3': 0
    }
    
    kept_samples = []
    
    # 解析每个样本（假设每个样本以0,1结束）
    i = 0
    while i < len(data):
        # 找到样本的结束位置
        sample_start = i
        
        # 读取source和target
        if i + 1 >= len(data):
            break
            
        source = data[i] - 2  # 减2是因为token偏移
        target = data[i + 1] - 2
        
        # 找到样本结束（0,1序列）
        sample_end = i + 2
        while sample_end < len(data) - 1:
            if data[sample_end] == 0 and data[sample_end + 1] == 1:
                sample_end += 2
                break
            sample_end += 1
        
        stats['total'] += 1
        
        # 检查是否是自边
        is_self_loop = False
        if source in S1 and target in S1:
            stats['removed_s1s1'] += 1
            is_self_loop = True
        elif source in S2 and target in S2:
            stats['removed_s2s2'] += 1
            is_self_loop = True
        elif source in S3 and target in S3:
            stats['removed_s3s3'] += 1
            is_self_loop = True
        
        if is_self_loop:
            stats['removed'] += 1
        else:
            # 保留这个样本
            kept_samples.append(data[sample_start:sample_end])
            stats['kept'] += 1
            
            # 统计保留的类型
            if source in S1 and target in S2:
                stats['kept_s1s2'] += 1
            elif source in S2 and target in S3:
                stats['kept_s2s3'] += 1
            elif source in S1 and target in S3:
                stats['kept_s1s3'] += 1
        
        i = sample_end
    
    # 合并所有保留的样本并写入
    if kept_samples:
        output_data = np.concatenate(kept_samples)
        output_data.tofile(output_path)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("FILTERING STATISTICS (Binary)")
    print("="*60)
    print(f"Total samples: {stats['total']}")
    print(f"Removed: {stats['removed']} ({stats['removed']/stats['total']*100:.1f}%)")
    print(f"  - S1->S1: {stats['removed_s1s1']}")
    print(f"  - S2->S2: {stats['removed_s2s2']}")
    print(f"  - S3->S3: {stats['removed_s3s3']}")
    print(f"\nKept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  - S1->S2: {stats['kept_s1s2']} ({stats['kept_s1s2']/stats['kept']*100:.1f}%)")
    print(f"  - S2->S3: {stats['kept_s2s3']} ({stats['kept_s2s3']/stats['kept']*100:.1f}%)")
    print(f"  - S1->S3: {stats['kept_s1s3']} ({stats['kept_s1s3']/stats['kept']*100:.1f}%)")
    print("="*60)
    
    return stats

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔧 REMOVING SELF-LOOPS FROM ALPINE STRICT DATASET")
    print("="*80)
    
    # 设置路径
    base_dir = 'data/simple_graph/composition_90_alpine_strict'
    stage_info_path = os.path.join(base_dir, 'stage_info.pkl')
    
    # 输出目录
    output_dir = 'data/simple_graph/composition_90_alpine_strict_no_selfloops'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nInput directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # 加载stage信息
    print(f"\nLoading stage info from: {stage_info_path}")
    S1, S2, S3 = load_stage_info(stage_info_path)
    print(f"  S1: {len(S1)} nodes")
    print(f"  S2: {len(S2)} nodes")
    print(f"  S3: {len(S3)} nodes")
    
    # 处理train_20.txt
    txt_input = os.path.join(base_dir, 'train_20.txt')
    txt_output = os.path.join(output_dir, 'train_20.txt')
    
    if os.path.exists(txt_input):
        process_text_file(txt_input, txt_output, S1, S2, S3)
    else:
        print(f"⚠️ {txt_input} not found")
    
    # 处理train_20.bin
    bin_input = os.path.join(base_dir, 'train_20.bin')
    bin_output = os.path.join(output_dir, 'train_20.bin')
    
    if os.path.exists(bin_input):
        process_binary_file(bin_input, bin_output, S1, S2, S3)
    else:
        print(f"⚠️ {bin_input} not found")
    
    # 复制其他必要文件
    import shutil
    
    files_to_copy = ['test.txt', 'test.bin', 'composition_graph.graphml', 'stage_info.pkl']
    for filename in files_to_copy:
        src = os.path.join(base_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ Copied {filename}")
    
    print("\n" + "="*80)
    print("✅ DATASET PROCESSING COMPLETE!")
    print(f"📁 New dataset saved to: {output_dir}")
    print("="*80)
    
    # 创建README说明
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# ALPINE Strict Dataset - No Self-Loops\n\n")
        f.write("This dataset is derived from `composition_90_alpine_strict` with all self-loops removed.\n\n")
        f.write("## Removed patterns:\n")
        f.write("- S1 → S1 (self-loops within S1)\n")
        f.write("- S2 → S2 (self-loops within S2)\n")
        f.write("- S3 → S3 (self-loops within S3)\n\n")
        f.write("## Kept patterns:\n")
        f.write("- S1 → S2 (forward edges)\n")
        f.write("- S2 → S3 (forward edges)\n")
        f.write("- S1 → S3 (skip connections)\n\n")
        f.write("This modification allows cleaner analysis of weight gaps between different stage transitions.\n")
    
    print(f"📝 README created at: {readme_path}")

if __name__ == "__main__":
    main()