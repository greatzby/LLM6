#!/usr/bin/env python3
# error_analysis_correct.py
"""
正确版本：使用与训练时相同的测试数据
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import pandas as pd
from pathlib import Path
import networkx as nx
import pickle
from tqdm import tqdm
from collections import defaultdict
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_data_correctly(data_dir='data/simple_graph/composition_90'):
    """正确加载测试数据 - 与训练时完全一致"""
    # 加载stage信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    S1, S2, S3 = stages
    
    # 加载test.txt - 这是关键！
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    # 只提取S1->S3的测试样例
    s1_s3_tests = []
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if source in S1 and target in S3:
                true_path = [int(p) for p in parts[2:]]
                s1_s3_tests.append({
                    'source': source,
                    'target': target,
                    'true_path': true_path
                })
    
    print(f"Loaded {len(s1_s3_tests)} S1->S3 test cases from test.txt")
    return s1_s3_tests, stages

def evaluate_checkpoint_correctly(checkpoint_path, test_cases, stages):
    """使用正确的评估逻辑 - 与训练时完全一致"""
    # 加载模型
    try:
        from model import GPT, GPTConfig
    except:
        from modeling_modified import GPT, GPTConfig
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        # 从你的训练代码默认值
        model_args = {
            'n_layer': 1,
            'n_head': 1,
            'n_embd': 120,  # 注意这里是120，不是32！
            'block_size': 12,
            'vocab_size': 100,
            'bias': False,
            'dropout': 0.0
        }
    
    # 创建模型
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    
    # 加载权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 加载元信息
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    S1, S2, S3 = stages
    
    # 评估 - 使用与训练时完全相同的逻辑
    correct = 0
    total = len(test_cases)
    
    errors = defaultdict(int)
    error_examples = defaultdict(list)
    
    for test_case in test_cases:
        source = test_case['source']
        target = test_case['target']
        true_path = test_case['true_path']
        
        # Token级编码 - 与训练代码完全一致
        prompt = f"{source} {target} {source}"
        prompt_tokens = prompt.split()
        
        prompt_ids = []
        for token in prompt_tokens:
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # 生成 - 使用相同的参数
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
        
        # 解码
        all_numbers = []
        for i, tid in enumerate(y[0].tolist()):
            if tid == 1:  # EOS
                break
            if tid in itos:
                try:
                    all_numbers.append(int(itos[tid]))
                except:
                    pass
        
        # 路径从第3个位置开始
        if len(all_numbers) >= 3:
            generated_path = all_numbers[2:]
        else:
            generated_path = []
        
        # 验证
        success = False
        if len(generated_path) >= 2:
            if generated_path[0] == source and generated_path[-1] == target:
                # 检查是否经过S2
                has_s2 = any(node in S2 for node in generated_path[1:-1])
                if has_s2:
                    # 验证路径有效性
                    path_valid = all(
                        G.has_edge(str(generated_path[i]), str(generated_path[i+1]))
                        for i in range(len(generated_path)-1)
                    )
                    if path_valid:
                        success = True
        
        if success:
            correct += 1
        else:
            # 分析错误类型
            if len(generated_path) < 2:
                errors['too_short'] += 1
            elif generated_path[0] != source:
                errors['wrong_start'] += 1
            elif len(generated_path) == 2 and generated_path[1] == target:
                errors['skip_s2'] += 1
                if len(error_examples['skip_s2']) < 3:
                    error_examples['skip_s2'].append({
                        'source': source,
                        'target': target,
                        'generated': generated_path,
                        'true_path': true_path
                    })
            else:
                errors['other'] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': dict(errors),
        'error_examples': dict(error_examples)
    }

def main():
    print("="*70)
    print("CORRECT ERROR ANALYSIS")
    print("="*70)
    
    # 加载正确的测试数据
    test_cases, stages = load_test_data_correctly()
    
    # 查找checkpoints
    checkpoint_pattern = "out/composition_mix*/ckpt_*_iter50000.pt"
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    print(f"\nFound {len(checkpoints)} checkpoints")
    
    results = []
    
    # 评估每个checkpoint
    for ckpt_path in tqdm(checkpoints[:20], desc="Evaluating"):  # 限制数量
        # 提取信息
        filename = Path(ckpt_path).name
        match = re.search(r'ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt', filename)
        if not match:
            continue
            
        info = {
            'ratio': int(match.group(1)),
            'seed': int(match.group(2)),
            'iteration': int(match.group(3))
        }
        
        # 评估
        eval_results = evaluate_checkpoint_correctly(ckpt_path, test_cases, stages)
        eval_results.update(info)
        results.append(eval_results)
        
        # 打印进度
        if len(results) % 5 == 0:
            print(f"  Latest: {info['ratio']}% - {eval_results['accuracy']:.1%}")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 按ratio分组
    grouped = df.groupby('ratio')['accuracy'].agg(['mean', 'std'])
    
    plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                 marker='o', linewidth=2, markersize=8, capsize=5)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    plt.fill_between([2.5, 5.5], 0, 1, alpha=0.2, color='yellow', label='Critical region')
    
    plt.xlabel('Mixture Ratio (%)')
    plt.ylabel('S1→S3 Accuracy')
    plt.title('Correct Evaluation Results')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('correct_evaluation.png', dpi=300)
    
    # 保存结果
    df.to_csv('correct_evaluation_results.csv', index=False)
    
    # 打印总结
    print("\n" + "="*70)
    print("CORRECT RESULTS")
    print("="*70)
    print("\nAccuracy by Ratio:")
    for ratio, row in grouped.iterrows():
        print(f"   {ratio:2d}%: {row['mean']:.1%} ± {row['std']:.1%}")

if __name__ == "__main__":
    main()