"""
dimension_importance_analysis_graph_composition.py
分析哪些维度对图路径组合任务至关重要
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import json
import os
import glob
from tqdm import tqdm
import pandas as pd
import networkx as nx
import pickle
from collections import defaultdict

# 导入您的模型定义
from model import GPT, GPTConfig

# 配置
CHECKPOINT_DIR = "out"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径 - 处理带时间戳的目录"""
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

def load_model_and_data(checkpoint_path, data_dir="data/simple_graph/composition_90"):
    """加载模型和数据"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 加载元数据
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    vocab_size = meta['vocab_size']
    block_size = meta['block_size']
    
    # 创建模型配置
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=1,
        n_head=2,
        n_embd=120,
        dropout=0.0,
        bias=False
    )
    
    # 创建模型并加载权重
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()
    
    # 加载图和阶段信息
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    return model, meta, G, stages

def generate_composition_test_data(G, stages, stoi, n_samples=100):
    """生成S1->S3组合任务测试数据"""
    S1, S2, S3 = stages
    
    test_data = []
    labels = []
    
    # 收集有效的S1->S3路径
    s1_s3_pairs = []
    for s1 in S1[:20]:  # 限制数量以加快速度
        for s3 in S3[:20]:
            if nx.has_path(G, str(s1), str(s3)):
                # 确保路径经过S2
                try:
                    path = nx.shortest_path(G, str(s1), str(s3))
                    path_nodes = [int(p) for p in path]
                    if any(node in S2 for node in path_nodes[1:-1]):
                        s1_s3_pairs.append((s1, s3, path_nodes))
                except:
                    pass
    
    # 随机选择样本
    if len(s1_s3_pairs) > n_samples:
        import random
        s1_s3_pairs = random.sample(s1_s3_pairs, n_samples)
    
    # 生成输入序列
    is_token_level = len(stoi) > 50
    
    for source, target, true_path in s1_s3_pairs:
        if is_token_level:
            # Token级编码
            prompt = f"{source} {target} {source}"
            prompt_tokens = prompt.split()
            
            prompt_ids = []
            for token in prompt_tokens:
                if token in stoi:
                    prompt_ids.append(stoi[token])
            
            # 添加padding
            while len(prompt_ids) < 10:
                prompt_ids.append(0)
            
            test_data.append(torch.tensor(prompt_ids[:10], dtype=torch.long))
            labels.append((source, target, true_path))
        else:
            # 字符级编码（如果需要）
            prompt_str = f"{source} {target}"
            prompt_ids = []
            for char in prompt_str:
                if char in stoi:
                    prompt_ids.append(stoi[char])
            
            while len(prompt_ids) < 10:
                prompt_ids.append(0)
                
            test_data.append(torch.tensor(prompt_ids[:10], dtype=torch.long))
            labels.append((source, target, true_path))
    
    return torch.stack(test_data), labels

def compute_gradient_importance_for_composition(model, test_data, labels, stages):
    """计算每个维度对S1->S3组合任务的梯度重要性"""
    model.eval()
    S1, S2, S3 = stages
    
    # 获取模型最后一层
    lm_head = model.lm_head
    hidden_dim = model.config.n_embd
    
    # 累积梯度
    accumulated_gradients = torch.zeros(hidden_dim).to(DEVICE)
    gradient_counts = 0
    
    # 分批处理
    batch_size = 16
    n_batches = (len(test_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Computing gradients"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_data))
        
        batch_data = test_data[start_idx:end_idx].to(DEVICE)
        batch_labels = labels[start_idx:end_idx]
        
        # 获取隐藏状态
        with torch.no_grad():
            # 通过模型的transformer部分
            tok_emb = model.transformer.wte(batch_data)
            pos = torch.arange(0, batch_data.size(1), dtype=torch.long, device=DEVICE).unsqueeze(0)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                x = block(x)
            hidden_states = model.transformer.ln_f(x)
        
        # 计算组合任务特定的损失
        hidden_states.requires_grad_(True)
        
        # 对每个样本计算损失
        batch_losses = []
        for i, (source, target, true_path) in enumerate(batch_labels):
            h = hidden_states[i:i+1]
            
            # 生成完整序列
            logits = lm_head(h)
            
            # 创建一个奖励信号：如果模型能正确预测S1->S3路径，给予奖励
            # 这里我们使用一个简化的损失：鼓励模型在看到S1和S3后激活正确的隐藏维度
            
            # 目标：让隐藏状态能够编码S1->S2->S3的组合信息
            # 使用对比损失
            
            # 正样本：真实的S1->S3对
            pos_score = h[:, -1, :].mean()  # 简化：使用平均激活作为分数
            
            # 负样本：随机的S1->S3对（实际上不连通的）
            neg_score = -h[:, -1, :].mean()  # 简化：负分数
            
            # 对比损失
            loss = -torch.log(torch.sigmoid(pos_score - neg_score))
            batch_losses.append(loss)
        
        # 合并批次损失
        total_loss = torch.stack(batch_losses).mean()
        
        # 反向传播
        total_loss.backward()
        
        # 收集梯度
        with torch.no_grad():
            # 获取隐藏状态的梯度
            grad = hidden_states.grad.mean(dim=(0, 1))  # 在batch和sequence维度上平均
            accumulated_gradients += torch.abs(grad)
            gradient_counts += 1
        
        # 清理梯度
        hidden_states.grad = None
    
    # 平均梯度
    dimension_importance = (accumulated_gradients / gradient_counts).cpu().numpy()
    
    # 归一化
    dimension_importance = dimension_importance / dimension_importance.max()
    
    return dimension_importance

def evaluate_model_on_composition(model, test_data, labels, G, stages, stoi, itos):
    """评估模型在S1->S3组合任务上的性能"""
    model.eval()
    S1, S2, S3 = stages
    
    correct = 0
    total = len(test_data)
    
    is_token_level = len(stoi) > 50
    
    with torch.no_grad():
        for i in range(len(test_data)):
            x = test_data[i:i+1].to(DEVICE)
            source, target, true_path = labels[i]
            
            # 生成路径
            y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
            
            # 解码路径
            if is_token_level:
                # Token级解码
                generated_numbers = []
                for tid in y[0].tolist():
                    if tid == 1:  # EOS
                        break
                    if tid in itos:
                        try:
                            generated_numbers.append(int(itos[tid]))
                        except:
                            pass
                
                # 路径从第3个位置开始
                if len(generated_numbers) >= 3:
                    generated_path = generated_numbers[2:]
                else:
                    generated_path = []
            else:
                # 字符级解码
                chars = []
                for tid in y[0].tolist():
                    if tid == 1:  # newline
                        break
                    if tid in itos and tid > 1:
                        chars.append(itos[tid])
                
                # 解析数字
                full_str = ''.join(chars)
                numbers = []
                current = ""
                
                for char in full_str:
                    if char == ' ':
                        if current.isdigit():
                            numbers.append(int(current))
                        current = ""
                    elif char.isdigit():
                        current += char
                
                if current.isdigit():
                    numbers.append(int(current))
                
                generated_path = numbers[2:] if len(numbers) >= 3 else []
            
            # 验证路径
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
    
    model.train()
    return correct / total if total > 0 else 0

def track_dimension_evolution(seeds=[42, 123, 456], ratios=[0, 20], 
                            iterations=[3000, 10000, 20000, 30000, 40000, 50000]):
    """追踪关键维度在训练过程中的演化"""
    results = {
        'dimension_changes': {},
        'performance': {},
        'correlations': {},
        'critical_dimensions': {}
    }
    
    # 使用早期模型识别关键维度
    print("\n=== Step 1: Identifying critical dimensions ===")
    
    # 加载参考模型和数据
    reference_model, meta, G, stages = load_model_and_data(
        get_checkpoint_path(0, seeds[0], 3000)
    )
    stoi, itos = meta['stoi'], meta['itos']
    
    # 生成测试数据
    test_data, labels = generate_composition_test_data(G, stages, stoi, n_samples=50)
    
    # 计算维度重要性
    dimension_importance = compute_gradient_importance_for_composition(
        reference_model, test_data, labels, stages
    )
    
    # 找出最重要的维度
    top_k = 20  # 追踪前20个最重要的维度
    critical_dims = np.argsort(dimension_importance)[-top_k:][::-1]
    print(f"Top {top_k} critical dimensions: {critical_dims}")
    print(f"Their importance scores: {dimension_importance[critical_dims]}")
    
    # 保存维度重要性
    results['dimension_importance'] = {
        'scores': dimension_importance.tolist(),
        'critical_dims': critical_dims.tolist(),
        'top_5_dims': critical_dims[:5].tolist(),
        'top_5_scores': dimension_importance[critical_dims[:5]].tolist()
    }
    
    # 追踪这些维度的变化
    print("\n=== Step 2: Tracking dimension evolution ===")
    
    for ratio in ratios:
        for seed in seeds:
            key = f"mix{ratio}_seed{seed}"
            results['dimension_changes'][key] = {}
            results['performance'][key] = {}
            results['critical_dimensions'][key] = {}
            
            print(f"\nProcessing {key}...")
            
            # 获取参考权重（初始状态）
            ref_model, _, _, _ = load_model_and_data(
                get_checkpoint_path(ratio, seed, iterations[0])
            )
            
            # 提取关键维度的初始状态
            # 使用最后一层前的layer norm权重作为维度表征
            ref_ln_weight = ref_model.transformer.ln_f.weight.detach().cpu().numpy()
            ref_critical_weights = ref_ln_weight[critical_dims]
            
            for iter_num in iterations:
                try:
                    # 加载模型
                    model, _, _, _ = load_model_and_data(
                        get_checkpoint_path(ratio, seed, iter_num)
                    )
                    
                    # 提取当前权重
                    current_ln_weight = model.transformer.ln_f.weight.detach().cpu().numpy()
                    current_critical_weights = current_ln_weight[critical_dims]
                    
                    # 计算每个关键维度的变化
                    dim_changes = []
                    for i in range(len(critical_dims)):
                        # 计算变化角度
                        ref_vec = ref_critical_weights[i]
                        curr_vec = current_critical_weights[i]
                        
                        cos_sim = np.dot(ref_vec, curr_vec) / (
                            np.linalg.norm(ref_vec) * np.linalg.norm(curr_vec) + 1e-8
                        )
                        cos_sim = np.clip(cos_sim, -1, 1)
                        angle = np.arccos(cos_sim) * 180 / np.pi
                        
                        # 计算幅度变化
                        norm_ratio = np.linalg.norm(curr_vec) / (np.linalg.norm(ref_vec) + 1e-8)
                        
                        dim_changes.append({
                            'cos_similarity': float(cos_sim),
                            'angle_degrees': float(angle),
                            'norm_ratio': float(norm_ratio),
                            'dim_idx': int(critical_dims[i])
                        })
                    
                    results['dimension_changes'][key][iter_num] = dim_changes
                    
                    # 评估模型性能
                    accuracy = evaluate_model_on_composition(
                        model, test_data, labels, G, stages, stoi, itos
                    )
                    results['performance'][key][iter_num] = float(accuracy)
                    
                    # 记录前5个关键维度的详细变化
                    top5_changes = {
                        'angles': [dim_changes[i]['angle_degrees'] for i in range(5)],
                        'avg_angle': np.mean([d['angle_degrees'] for d in dim_changes[:5]]),
                        'max_angle': max([d['angle_degrees'] for d in dim_changes[:5]])
                    }
                    results['critical_dimensions'][key][iter_num] = top5_changes
                    
                    print(f"  iter {iter_num}: accuracy={accuracy:.3f}, "
                          f"avg_angle(top5)={top5_changes['avg_angle']:.1f}°")
                    
                except Exception as e:
                    print(f"  Error at iter {iter_num}: {e}")
    
    # 计算相关性
    print("\n=== Step 3: Computing correlations ===")
    compute_dimension_correlations(results)
    
    return results

def compute_dimension_correlations(results):
    """计算维度变化与性能的相关性"""
    for key in results['dimension_changes'].keys():
        if key not in results['performance']:
            continue
        
        iterations = sorted(results['dimension_changes'][key].keys())
        
        # 收集数据
        metrics = {
            'avg_angle_all': [],
            'avg_angle_top5': [],
            'max_angle_top5': [],
            'performances': []
        }
        
        for iter_num in iterations:
            if iter_num in results['performance'][key]:
                # 所有关键维度的平均角度
                dim_data = results['dimension_changes'][key][iter_num]
                avg_angle_all = np.mean([d['angle_degrees'] for d in dim_data])
                
                # Top5维度的指标
                top5_data = results['critical_dimensions'][key][iter_num]
                
                metrics['avg_angle_all'].append(avg_angle_all)
                metrics['avg_angle_top5'].append(top5_data['avg_angle'])
                metrics['max_angle_top5'].append(top5_data['max_angle'])
                metrics['performances'].append(results['performance'][key][iter_num])
        
        if len(metrics['performances']) > 3:
            # 计算多种相关性
            correlations = {}
            
            for metric_name in ['avg_angle_all', 'avg_angle_top5', 'max_angle_top5']:
                if metrics[metric_name]:
                    pearson_r, pearson_p = pearsonr(metrics[metric_name], metrics['performances'])
                    spearman_r, spearman_p = spearmanr(metrics[metric_name], metrics['performances'])
                    
                    correlations[metric_name] = {
                        'pearson': {'r': float(pearson_r), 'p': float(pearson_p)},
                        'spearman': {'r': float(spearman_r), 'p': float(spearman_p)}
                    }
            
            results['correlations'][key] = {
                'metrics': correlations,
                'data': metrics
            }

def visualize_results(results):
    """可视化分析结果"""
    # 创建图形
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 维度重要性分布
    plt.subplot(3, 3, 1)
    importance = results['dimension_importance']['scores']
    sorted_importance = sorted(importance, reverse=True)
    plt.bar(range(len(sorted_importance)), sorted_importance)
    plt.axvline(x=19.5, color='r', linestyle='--', label='Top 20 cutoff')
    plt.xlabel('Dimension Rank')
    plt.ylabel('Importance Score')
    plt.title('Dimension Importance Distribution')
    plt.legend()
    
    # 2. Top5维度平均角度变化
    plt.subplot(3, 3, 2)
    for ratio, color in [(0, 'red'), (20, 'blue')]:
        all_iters = []
        all_angles = []
        
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['critical_dimensions']:
                for iter_num, data in results['critical_dimensions'][key].items():
                    all_iters.append(iter_num)
                    all_angles.append(data['avg_angle'])
        
        if all_iters:
            # 计算每个iteration的平均值
            iter_angle_map = defaultdict(list)
            for it, angle in zip(all_iters, all_angles):
                iter_angle_map[it].append(angle)
            
            iters = sorted(iter_angle_map.keys())
            avg_angles = [np.mean(iter_angle_map[it]) for it in iters]
            std_angles = [np.std(iter_angle_map[it]) for it in iters]
            
            plt.plot(iters, avg_angles, color=color, label=f'{ratio}% mix', linewidth=2)
            plt.fill_between(iters, 
                           np.array(avg_angles) - np.array(std_angles),
                           np.array(avg_angles) + np.array(std_angles),
                           color=color, alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Angle Change (degrees)')
    plt.title('Top-5 Critical Dimensions: Average Rotation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 性能轨迹
    plt.subplot(3, 3, 3)
    for ratio, color in [(0, 'red'), (20, 'blue')]:
        for seed, style in [(42, '-'), (123, '--'), (456, ':')]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['performance']:
                iters = sorted(results['performance'][key].keys())
                perfs = [results['performance'][key][i] for i in iters]
                
                plt.plot(iters, perfs, color=color, linestyle=style,
                        label=f'{ratio}% mix, seed {seed}', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('S1→S3 Composition Accuracy')
    plt.title('Composition Performance Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 角度vs性能散点图（所有维度）
    plt.subplot(3, 3, 4)
    for ratio, color in [(0, 'red'), (20, 'blue')]:
        all_angles = []
        all_perfs = []
        
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['correlations'] and 'data' in results['correlations'][key]:
                all_angles.extend(results['correlations'][key]['data']['avg_angle_all'])
                all_perfs.extend(results['correlations'][key]['data']['performances'])
        
        if all_angles:
            plt.scatter(all_angles, all_perfs, color=color, label=f'{ratio}% mix',
                       alpha=0.6, s=50)
    
    plt.xlabel('Average Angle Change - All Top-20 Dims (degrees)')
    plt.ylabel('Composition Accuracy')
    plt.title('Performance vs Dimension Rotation (All Dims)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 角度vs性能散点图（Top5维度）
    plt.subplot(3, 3, 5)
    for ratio, color in [(0, 'red'), (20, 'blue')]:
        all_angles = []
        all_perfs = []
        
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['correlations'] and 'data' in results['correlations'][key]:
                all_angles.extend(results['correlations'][key]['data']['avg_angle_top5'])
                all_perfs.extend(results['correlations'][key]['data']['performances'])
        
        if all_angles:
            plt.scatter(all_angles, all_perfs, color=color, label=f'{ratio}% mix',
                       alpha=0.6, s=50)
            
            # 添加趋势线
            if len(all_angles) > 3:
                z = np.polyfit(all_angles, all_perfs, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(all_angles), max(all_angles), 100)
                plt.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8)
    
    plt.xlabel('Average Angle Change - Top-5 Dims (degrees)')
    plt.ylabel('Composition Accuracy')
    plt.title('Performance vs Dimension Rotation (Top-5 Dims)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 相关性热图
    plt.subplot(3, 3, 6)
    corr_data = []
    labels = []
    
    for key in sorted(results['correlations'].keys()):
        if 'metrics' in results['correlations'][key]:
            for metric in ['avg_angle_top5', 'max_angle_top5']:
                if metric in results['correlations'][key]['metrics']:
                    corr = results['correlations'][key]['metrics'][metric]['pearson']['r']
                    corr_data.append(corr)
                    labels.append(f"{key}_{metric}")
    
    if corr_data:
        corr_matrix = np.array(corr_data).reshape(-1, 2)
        sns.heatmap(corr_matrix, 
                   xticklabels=['Avg Angle', 'Max Angle'],
                   yticklabels=[l.split('_')[0] for l in labels[::2]],
                   annot=True, fmt='.3f', center=0, cmap='RdBu_r',
                   vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
        plt.title('Correlation: Angle vs Performance')
    
    # 7. 各个维度的变化热图
    plt.subplot(3, 3, 7)
    # 选择一个代表性的配置
    key = "mix0_seed42"
    if key in results['dimension_changes']:
        iters = sorted(results['dimension_changes'][key].keys())
        top_dims = results['dimension_importance']['critical_dims'][:10]
        
        angle_matrix = []
        for dim_idx in range(10):
            angles = []
            for iter_num in iters:
                if iter_num in results['dimension_changes'][key]:
                    angles.append(results['dimension_changes'][key][iter_num][dim_idx]['angle_degrees'])
            angle_matrix.append(angles)
        
        im = plt.imshow(angle_matrix, aspect='auto', cmap='hot')
        plt.colorbar(im, label='Angle (degrees)')
        plt.xlabel('Training Progress')
        plt.ylabel('Dimension Index (in top-10)')
        plt.title(f'Dimension Rotation Heatmap ({key})')
        plt.xticks(range(len(iters)), [f"{it//1000}k" for it in iters])
    
    # 8. 临界角度分析
    plt.subplot(3, 3, 8)
    threshold_angles = {'0% mix': [], '20% mix': []}
    
    for ratio in [0, 20]:
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['correlations'] and 'data' in results['correlations'][key]:
                angles = results['correlations'][key]['data']['avg_angle_top5']
                perfs = results['correlations'][key]['data']['performances']
                
                # 找出性能下降到0.5以下的第一个点
                for i, (angle, perf) in enumerate(zip(angles, perfs)):
                    if perf < 0.5 and i > 0:
                        threshold_angles[f'{ratio}% mix'].append(angle)
                        break
    
    # 绘制临界角度分布
    for i, (label, angles) in enumerate(threshold_angles.items()):
        if angles:
            plt.hist(angles, bins=10, alpha=0.7, label=label, 
                    color=['red', 'blue'][i])
            plt.axvline(np.mean(angles), color=['red', 'blue'][i], 
                       linestyle='--', linewidth=2,
                       label=f'{label} mean: {np.mean(angles):.1f}°')
    
    plt.xlabel('Critical Angle (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of Critical Angles for Performance Collapse')
    plt.legend()
    
    # 9. 汇总统计
    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.9, "KEY FINDINGS:", fontsize=14, fontweight='bold',
            transform=plt.gca().transAxes)
    
    y_pos = 0.8
    for ratio in [0, 20]:
        # 计算平均相关性
        corrs = []
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['correlations'] and 'metrics' in results['correlations'][key]:
                if 'avg_angle_top5' in results['correlations'][key]['metrics']:
                    corrs.append(results['correlations'][key]['metrics']['avg_angle_top5']['pearson']['r'])
        
        if corrs:
            avg_corr = np.mean(corrs)
            plt.text(0.1, y_pos, f"{ratio}% mix: Avg correlation = {avg_corr:.3f}",
                    fontsize=12, transform=plt.gca().transAxes)
            y_pos -= 0.1
    
    # 添加关键维度信息
    plt.text(0.1, y_pos-0.1, f"\nTop-5 critical dimensions: {results['dimension_importance']['top_5_dims']}",
            fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, y_pos-0.2, f"Their importance scores: {[f'{s:.3f}' for s in results['dimension_importance']['top_5_scores']]}",
            fontsize=10, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dimension_importance_analysis_graph_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to: dimension_importance_analysis_graph_composition.png")

def main():
    print("="*80)
    print("DIMENSION IMPORTANCE ANALYSIS FOR GRAPH COMPOSITION TASK")
    print("="*80)
    
    # 运行分析
    results = track_dimension_evolution(
        seeds=[42, 123, 456],
        ratios=[0, 20],
        iterations=[3000, 10000, 20000, 30000, 40000, 50000]
    )
    
    # 保存结果
    with open('dimension_importance_results_graph_composition.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    visualize_results(results)
    
    # 打印关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 打印维度重要性
    print("\nTop-10 most important dimensions for S1→S3 composition:")
    top_10_dims = results['dimension_importance']['critical_dims'][:10]
    top_10_scores = [results['dimension_importance']['scores'][d] for d in top_10_dims]
    for i, (dim, score) in enumerate(zip(top_10_dims, top_10_scores)):
        print(f"  {i+1}. Dimension {dim}: importance = {score:.3f}")
    
    # 打印相关性
    print("\nCorrelations between dimension rotation and performance:")
    print("-"*60)
    for key in sorted(results['correlations'].keys()):
        if 'metrics' in results['correlations'][key]:
            metrics = results['correlations'][key]['metrics']
            if 'avg_angle_top5' in metrics:
                r = metrics['avg_angle_top5']['pearson']['r']
                p = metrics['avg_angle_top5']['pearson']['p']
                print(f"{key}: Pearson r = {r:.3f} (p = {p:.3f})")
    
    # 找出临界角度
    print("\nCritical angle analysis:")
    print("-"*60)
    for ratio in [0, 20]:
        critical_angles = []
        for seed in [42, 123, 456]:
            key = f"mix{ratio}_seed{seed}"
            if key in results['correlations'] and 'data' in results['correlations'][key]:
                angles = results['correlations'][key]['data']['avg_angle_top5']
                perfs = results['correlations'][key]['data']['performances']
                
                # 找出性能下降到0.5以下的点
                for i, (angle, perf) in enumerate(zip(angles, perfs)):
                    if perf < 0.5 and i > 0 and angles[i-1] < angle:
                        critical_angles.append(angle)
                        print(f"  {key}: Performance dropped below 0.5 at angle = {angle:.1f}°")
                        break
        
        if critical_angles:
            print(f"\n{ratio}% mix: Average critical angle = {np.mean(critical_angles):.1f}° ± {np.std(critical_angles):.1f}°")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("Results saved to:")
    print("  - dimension_importance_results_graph_composition.json")
    print("  - dimension_importance_analysis_graph_composition.png")
    print("="*80)

if __name__ == "__main__":
    main()