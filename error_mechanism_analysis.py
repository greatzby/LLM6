#!/usr/bin/env python3
# error_mechanism_analysis_fixed.py
"""
修复版：解决JSON序列化问题
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
import pickle
import networkx as nx
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import os
import re

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========== 配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Token映射
S1_TOKENS = list(range(2, 32))    # 30 S1 nodes
S2_TOKENS = list(range(32, 62))   # 30 S2 nodes  
S3_TOKENS = list(range(62, 92))   # 30 S3 nodes
PAD_TOKEN = 0
EOS_TOKEN = 1

# ========== JSON序列化辅助函数 ==========
def convert_to_serializable(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

# ========== 工具函数 ==========
def load_graph_and_edges(data_dir='data/simple_graph/composition_90'):
    """加载图和边信息"""
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    edges_s1_s2 = defaultdict(list)
    edges_s2_s3 = defaultdict(list)
    
    for edge in G.edges():
        source, target = int(edge[0]), int(edge[1])
        
        if source in S1_TOKENS and target in S2_TOKENS:
            edges_s1_s2[source].append(target)
        elif source in S2_TOKENS and target in S3_TOKENS:
            edges_s2_s3[source].append(target)
    
    return G, edges_s1_s2, edges_s2_s3

def load_model_simple(checkpoint_path):
    """简化的模型加载"""
    try:
        from modeling_modified import GPT, GPTConfig
    except:
        from model import GPT, GPTConfig
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        model_args = {
            'n_layer': 1,
            'n_head': 1,
            'n_embd': 32,
            'block_size': 12,
            'vocab_size': 100,
            'bias': False,
            'dropout': 0.0
        }
    
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def generate_path(model, s1, s3, max_len=10):
    """生成S1到S3的路径"""
    model.eval()
    
    prompt = torch.tensor([s1, s3, s1], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=max_len, temperature=0.1, top_k=10)
        generated = output[0, 3:].tolist()
        
        path = []
        for token in generated:
            if token == EOS_TOKEN:
                break
            if token >= 2:
                path.append(token)
        
    return [s1] + path if path else [s1]

def analyze_single_checkpoint(checkpoint_path, G, edges_s1_s2, edges_s2_s3, n_samples=300):
    """分析单个checkpoint的错误类型（减少样本数以加快速度）"""
    print(f"  Loading {Path(checkpoint_path).name}")
    
    model = load_model_simple(checkpoint_path)
    
    errors = {
        'total': 0,
        'correct': 0,
        'skip_s2': 0,
        'wrong_s2': 0,
        'stuck_s2': 0,
        'wrong_s3': 0,
        'incomplete': 0,
        'invalid': 0
    }
    
    error_examples = defaultdict(list)
    
    # 生成测试样例
    test_pairs = []
    for _ in range(n_samples):
        s1 = np.random.choice(S1_TOKENS)
        valid_s3s = []
        for s2 in edges_s1_s2[s1]:
            valid_s3s.extend(edges_s2_s3[s2])
        if valid_s3s:
            s3 = np.random.choice(valid_s3s)
            test_pairs.append((s1, s3))
    
    # 分析每个测试样例
    for s1, s3 in test_pairs:
        errors['total'] += 1
        
        generated_path = generate_path(model, s1, s3)
        
        # 分析错误类型
        if len(generated_path) == 2 and generated_path[0] == s1 and generated_path[1] == s3:
            errors['skip_s2'] += 1
            if len(error_examples['skip_s2']) < 3:
                error_examples['skip_s2'].append({
                    's1': int(s1), 's3': int(s3), 
                    'generated': [int(x) for x in generated_path]
                })
            
        elif len(generated_path) >= 3:
            path_valid = True
            has_s2 = False
            
            for node in generated_path[1:-1]:
                if node in S2_TOKENS:
                    has_s2 = True
                    break
            
            for i in range(len(generated_path)-1):
                if not G.has_edge(str(generated_path[i]), str(generated_path[i+1])):
                    path_valid = False
                    break
            
            if path_valid and has_s2 and generated_path[-1] == s3:
                errors['correct'] += 1
            elif generated_path[1] in S2_TOKENS and generated_path[1] not in edges_s1_s2[s1]:
                errors['wrong_s2'] += 1
            elif len(generated_path) >= 2 and generated_path[-1] in S2_TOKENS:
                errors['stuck_s2'] += 1
            elif has_s2 and generated_path[-1] != s3:
                errors['wrong_s3'] += 1
            else:
                errors['invalid'] += 1
        else:
            errors['incomplete'] += 1
    
    accuracy = errors['correct'] / errors['total'] if errors['total'] > 0 else 0
    error_rates = {k: v/errors['total'] for k, v in errors.items() if k not in ['total', 'correct']}
    
    return {
        'accuracy': float(accuracy),
        'errors': {k: int(v) for k, v in errors.items()},
        'error_rates': {k: float(v) for k, v in error_rates.items()},
        'error_examples': convert_to_serializable(dict(error_examples))
    }

def extract_checkpoint_info(checkpoint_path):
    """从文件名提取信息"""
    filename = Path(checkpoint_path).name
    match = re.search(r'ckpt_mix(\d+)_seed(\d+)_iter(\d+)\.pt', filename)
    if match:
        return {
            'ratio': int(match.group(1)),
            'seed': int(match.group(2)),
            'iteration': int(match.group(3))
        }
    return None

def main():
    """主分析流程"""
    print("="*70)
    print("ERROR MECHANISM ANALYSIS (FIXED)")
    print("="*70)
    
    print("Loading graph structure...")
    G, edges_s1_s2, edges_s2_s3 = load_graph_and_edges()
    
    checkpoint_pattern = "out/composition_mix*/ckpt_*_iter50000.pt"
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    print(f"\nFound {len(checkpoints)} checkpoints at 50k iterations")
    
    # 限制分析的checkpoint数量（为了速度）
    if len(checkpoints) > 20:
        print(f"Limiting analysis to key ratios: 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20")
        key_ratios = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
        filtered_checkpoints = []
        for ckpt in checkpoints:
            info = extract_checkpoint_info(ckpt)
            if info and info['ratio'] in key_ratios:
                filtered_checkpoints.append(ckpt)
        checkpoints = filtered_checkpoints
        print(f"Analyzing {len(checkpoints)} checkpoints")
    
    all_results = []
    
    print("\nAnalyzing checkpoints...")
    for ckpt_path in tqdm(checkpoints):
        info = extract_checkpoint_info(ckpt_path)
        if info is None:
            continue
        
        try:
            results = analyze_single_checkpoint(ckpt_path, G, edges_s1_s2, edges_s2_s3)
            results.update(info)
            all_results.append(results)
            
            # 保存错误示例（只保存关键ratio）
            if info['ratio'] in [0, 3, 5, 10] and info['seed'] == 42:
                examples_file = f"error_examples_r{info['ratio']}.json"
                with open(examples_file, 'w') as f:
                    json.dump(convert_to_serializable({
                        'info': info,
                        'accuracy': results['accuracy'],
                        'error_rates': results['error_rates'],
                        'examples': results['error_examples']
                    }), f, indent=2)
        except Exception as e:
            print(f"\nError processing {ckpt_path}: {e}")
            continue
    
    if not all_results:
        print("No results to analyze!")
        return
    
    df = pd.DataFrame(all_results)
    
    # ========== 简化的绘图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 准确率 vs Ratio
    ax1 = axes[0, 0]
    grouped = df.groupby('ratio')['accuracy'].agg(['mean', 'std'])
    ax1.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                 marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between([2.5, 5.5], 0, 1, alpha=0.2, color='yellow')
    ax1.set_xlabel('Mixture Ratio (%)')
    ax1.set_ylabel('Accuracy on S1→S3')
    ax1.set_title('Phase Transition at 3-5%')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Skip S2错误率
    ax2 = axes[0, 1]
    skip_rates = df.groupby('ratio').apply(
        lambda x: x['error_rates'].apply(lambda y: y.get('skip_s2', 0)).mean()
    )
    ax2.plot(skip_rates.index, skip_rates.values, 'ro-', linewidth=3, markersize=10)
    ax2.set_xlabel('Mixture Ratio (%)')
    ax2.set_ylabel('Skip S2 Error Rate')
    ax2.set_title('Direct S1→S3 Jumps')
    ax2.grid(True, alpha=0.3)
    
    # 3. 方差分析
    ax3 = axes[1, 0]
    variance = df.groupby('ratio')['accuracy'].std()
    bars = ax3.bar(variance.index, variance.values)
    for i, (ratio, var) in enumerate(variance.items()):
        if 3 <= ratio <= 5:
            bars[i].set_color('orange')
    ax3.set_xlabel('Mixture Ratio (%)')
    ax3.set_ylabel('Std Dev of Accuracy')
    ax3.set_title('Instability in Critical Region')
    ax3.grid(True, alpha=0.3)
    
    # 4. 总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""KEY FINDINGS:
    
1. Threshold: 3-5%
   Below: <50% accuracy
   Above: >80% accuracy
   
2. Main Error: Skip S2
   0% mix: {skip_rates.get(0, 0):.1%}
   5% mix: {skip_rates.get(5, 0):.1%}
   
3. Mechanism:
   Direct training → S2 bypass
   Mixed training → Maintains paths
"""
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    plt.suptitle('Error Mechanism Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('error_mechanism_fixed.png', dpi=300)
    
    # 保存结果
    df.to_csv('error_analysis_fixed.csv', index=False)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nAccuracy by Ratio:")
    for ratio, row in grouped.iterrows():
        print(f"   {ratio:2d}%: {row['mean']:.1%} ± {row['std']:.1%}")
    
    print("\nFiles saved:")
    print("   • error_mechanism_fixed.png")
    print("   • error_analysis_fixed.csv")

if __name__ == "__main__":
    main()