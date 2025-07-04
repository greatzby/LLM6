# prepare_mixed_data_batch_safe.py
import os
import shutil
import numpy as np
import pickle
import glob
import re

def prepare_mixed_dataset_original_logic(mixed_dir, mixed_ratio):
    """使用原始逻辑准备混合数据集 - 100%兼容"""
    print(f"Preparing {mixed_dir} (ratio={mixed_ratio}%)...")
    
    # 获取原始目录 - 使用固定路径而不是replace
    original_dir = 'data/simple_graph/composition_90'
    
    # 检查目录存在
    if not os.path.exists(mixed_dir):
        print(f"  Error: Directory {mixed_dir} does not exist!")
        return False
    
    # 读取训练数据
    train_file = os.path.join(mixed_dir, 'train_10.txt')
    if not os.path.exists(train_file):
        print(f"  Error: {train_file} not found!")
        return False
        
    with open(train_file, 'r') as f:
        lines = f.readlines()
    
    print(f"  Found {len(lines)} lines in train_10.txt")
    
    # 加载编码器
    meta_file = os.path.join(original_dir, 'meta.pkl')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    block_size = meta['block_size']
    
    # 复制meta.pkl到目标目录（如果不存在）
    target_meta = os.path.join(mixed_dir, 'meta.pkl')
    if not os.path.exists(target_meta):
        shutil.copy(meta_file, target_meta)
        print(f"  Copied meta.pkl to {mixed_dir}")
    
    # ===== 核心逻辑：与原始代码完全一致 =====
    train_ids = []
    for line in lines:
        if line.strip():
            # 完全复制原始代码的这一行
            tokens = [stoi[t] for t in line.strip().split() if t in stoi]
            tokens.append(1)  # EOS
            # Padding - 完全复制原始代码
            tokens.extend([0] * (block_size + 1 - len(tokens)))
            train_ids.extend(tokens)
    
    # 保存 - 完全复制原始代码
    train_ids = np.array(train_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(mixed_dir, 'train_10.bin'))
    # ===== 核心逻辑结束 =====
    
    print(f"  Created train_10.bin: {len(train_ids)} tokens")
    print(f"  Done!")
    return True

def verify_compatibility(test_dir='data/simple_graph/composition_90_mixed_1'):
    """验证新代码与原始代码的兼容性"""
    print("\n" + "="*70)
    print("Compatibility Verification")
    print("="*70)
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found, skipping verification")
        return
    
    # 备份现有文件（如果存在）
    bin_file = os.path.join(test_dir, 'train_10.bin')
    backup_file = os.path.join(test_dir, 'train_10.bin.backup')
    
    if os.path.exists(bin_file):
        shutil.copy(bin_file, backup_file)
        print(f"Backed up existing file to {backup_file}")
    
    # 运行新代码
    prepare_mixed_dataset_original_logic(test_dir, 1)
    
    # 如果有备份，比较文件
    if os.path.exists(backup_file):
        with open(bin_file, 'rb') as f1, open(backup_file, 'rb') as f2:
            new_data = f1.read()
            old_data = f2.read()
            
        if new_data == old_data:
            print("✓ Verification PASSED: Files are identical!")
        else:
            print("✗ Verification FAILED: Files differ!")
            print(f"  New file size: {len(new_data)} bytes")
            print(f"  Old file size: {len(old_data)} bytes")
        
        # 清理备份
        os.remove(backup_file)

def find_all_mixed_dirs(base_path='data/simple_graph', skip_ratios=[5, 10]):
    """查找所有混合数据集目录（跳过指定的比例）"""
    pattern = os.path.join(base_path, 'composition_90_mixed_*')
    mixed_dirs = glob.glob(pattern)
    
    dir_info = []
    for dir_path in mixed_dirs:
        match = re.search(r'composition_90_mixed_(\d+)$', dir_path)
        if match:
            ratio = int(match.group(1))
            if ratio not in skip_ratios:
                dir_info.append((ratio, dir_path))
    
    dir_info.sort(key=lambda x: x[0])
    return dir_info

def main():
    """主函数：批量处理所有混合数据集（跳过5%和10%）"""
    print("="*70)
    print("Batch Processing Mixed Datasets (100% Compatible Version)")
    print("★ Using EXACT same logic as original code")
    print("★ Skipping 5% and 10% (already processed)")
    print("="*70)
    
    # 可选：运行兼容性验证
    # verify_compatibility()
    
    # 查找所有混合数据集（跳过5%和10%）
    mixed_datasets = find_all_mixed_dirs(skip_ratios=[5, 10])
    
    if not mixed_datasets:
        print("No new mixed datasets found!")
        return
    
    print(f"\nFound {len(mixed_datasets)} new mixed datasets to process:")
    for ratio, dir_path in mixed_datasets:
        # 检查是否已有train_10.bin
        existing = os.path.exists(os.path.join(dir_path, 'train_10.bin'))
        status = " [EXISTS]" if existing else ""
        print(f"  - {ratio}%: {dir_path}{status}")
    
    print("\nSkipping:")
    print("  - 5%: data/simple_graph/composition_90_mixed_5")
    print("  - 10%: data/simple_graph/composition_90_mixed_10")
    
    # 用户确认
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print("\n" + "="*70)
    print("Processing...")
    print("="*70 + "\n")
    
    # 处理每个数据集
    success_count = 0
    failed_dirs = []
    
    for ratio, dir_path in mixed_datasets:
        success = prepare_mixed_dataset_original_logic(dir_path, ratio)
        if success:
            success_count += 1
        else:
            failed_dirs.append((ratio, dir_path))
        print()
    
    # 汇总结果
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Processed: {success_count}/{len(mixed_datasets)}")
    
    if failed_dirs:
        print("\nFailed:")
        for ratio, dir_path in failed_dirs:
            print(f"  - {ratio}%: {dir_path}")

if __name__ == "__main__":
    # 安全起见，添加用户确认
    print("This script will create train_10.bin files for all mixed datasets")
    print("except 5% and 10% which are already processed.")
    print()
    main()