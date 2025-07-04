# compute_brc_evolution.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
import pickle
from model import GPTConfig, GPT
from tabulate import tabulate

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """从checkpoint加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model

def get_deep_node_embeddings(model, nodes, device='cuda'):
    """获取节点的深层嵌入"""
    embeddings = []
    
    with torch.no_grad():
        for node in nodes:
            x = torch.tensor([node], dtype=torch.long, device=device).unsqueeze(0)
            
            # 前向传播
            tok_emb = model.transformer.wte(x)
            pos_emb = model.transformer.wpe(torch.arange(x.size(1), device=device))
            x = model.transformer.drop(tok_emb + pos_emb)
            
            for block in model.transformer.h:
                x = block(x)
            
            x = model.transformer.ln_f(x)
            
            # 取第一个位置的嵌入
            node_emb = x[0, 0, :].cpu().numpy()
            embeddings.append(node_emb)
    
    return np.array(embeddings)

def compute_node_bias(node_emb, S1_embeddings, S3_embeddings):
    """计算单个节点的偏向值"""
    S1_centroid = S1_embeddings.mean(axis=0)
    S3_centroid = S3_embeddings.mean(axis=0)
    
    dist_to_S1 = np.linalg.norm(node_emb - S1_centroid)
    dist_to_S3 = np.linalg.norm(node_emb - S3_centroid)
    
    # 偏向值：正值表示偏向S3
    bias = (dist_to_S1 - dist_to_S3) / (dist_to_S1 + dist_to_S3)
    
    return bias

def compute_brc_metrics(model, stages, device='cuda'):
    """计算BRC相关指标"""
    S1, S2, S3 = stages
    
    # 获取各阶段嵌入
    S1_embeddings = get_deep_node_embeddings(model, S1, device)
    S2_embeddings = get_deep_node_embeddings(model, S2, device)
    S3_embeddings = get_deep_node_embeddings(model, S3, device)
    
    # 1. 计算bias分布
    biases = []
    for i, node in enumerate(S2):
        emb = S2_embeddings[i]
        bias = compute_node_bias(emb, S1_embeddings, S3_embeddings)
        biases.append(bias)
    
    sigma_bias = np.std(biases)
    mean_bias = np.mean(biases)
    
    # 2. 计算有效维度 (90%方差)
    pca = PCA()
    pca.fit(S2_embeddings)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ED_90 = np.argmax(cumsum_var >= 0.9) + 1
    
    # 3. 参与率
    eigenvalues = pca.explained_variance_
    PR = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
    
    # 4. S2内部距离标准差
    if len(S2) > 1:
        pairwise_dists = pdist(S2_embeddings)
        dist_std = np.std(pairwise_dists)
    else:
        dist_std = 0
    
    # 5. BRC指数
    BRC = sigma_bias / ED_90 if ED_90 > 0 else 0
    
    # 6. 条件数
    S2_centered = S2_embeddings - S2_embeddings.mean(axis=0)
    U, S, V = np.linalg.svd(S2_centered)
    condition_number = S[0] / S[-1] if S[-1] > 1e-10 else np.inf
    
    return {
        'sigma_bias': sigma_bias,
        'mean_bias': mean_bias,
        'ED_90': ED_90,
        'PR': PR,
        'dist_std': dist_std,
        'BRC': BRC,
        'condition_number': condition_number
    }

def analyze_all_checkpoints():
    """分析所有已训练模型的BRC演化"""
    
    # 加载阶段信息（使用统一的stage_info）
    with open('data/simple_graph/composition_90/stage_info.pkl', 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    print(f"Stages loaded: S1={len(stages[0])} nodes, S2={len(stages[1])} nodes, S3={len(stages[2])} nodes")
    
    models = {
        'original': 'out/composition_20250702_063926',
        '5% mixed': 'out/composition_20250703_004537', 
        '10% mixed': 'out/composition_20250703_011304'
    }
    
    checkpoints = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    # 收集所有结果用于表格
    table_data = []
    results = {}
    
    for name, out_dir in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {name}")
        print('='*60)
        
        model_results = {
            'iters': [], 'BRC': [], 'ED_90': [], 'sigma_bias': [], 
            'mean_bias': [], 'PR': [], 'condition': [], 'dist_std': []
        }
        
        for ckpt in checkpoints:
            ckpt_path = f"{out_dir}/ckpt_{ckpt}.pt"
            if not os.path.exists(ckpt_path):
                print(f"  Checkpoint {ckpt} not found, skipping...")
                continue
            
            print(f"\n  Processing iteration {ckpt}...")
            
            try:
                # 加载模型
                model = load_model_from_checkpoint(ckpt_path)
                
                # 计算BRC指标
                metrics = compute_brc_metrics(model, stages)
                
                # 保存结果
                model_results['iters'].append(ckpt)
                model_results['BRC'].append(metrics['BRC'])
                model_results['ED_90'].append(metrics['ED_90'])
                model_results['sigma_bias'].append(metrics['sigma_bias'])
                model_results['mean_bias'].append(metrics['mean_bias'])
                model_results['PR'].append(metrics['PR'])
                model_results['condition'].append(metrics['condition_number'])
                model_results['dist_std'].append(metrics['dist_std'])
                
                # 添加到表格数据
                table_data.append([
                    name, 
                    f"{ckpt/1000:.0f}k",
                    f"{metrics['BRC']:.3f}",
                    f"{metrics['ED_90']:.0f}",
                    f"{metrics['sigma_bias']:.3f}",
                    f"{metrics['mean_bias']:.3f}",
                    f"{metrics['PR']:.1f}",
                    f"{metrics['dist_std']:.3f}"
                ])
                
                # 清理GPU内存
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        results[name] = model_results
    
    # 打印表格
    print("\n" + "="*100)
    print("BRC Analysis Summary")
    print("="*100)
    
    headers = ["Model", "Iter", "BRC", "ED_90", "σ_bias", "Mean_bias", "PR", "Dist_std"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存结果
    with open('brc_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_brc_evolution(results):
    """绘制BRC核心指标"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'original': '#1f77b4', '5% mixed': '#ff7f0e', '10% mixed': '#2ca02c'}
    
    # 1. BRC演化
    ax1 = axes[0, 0]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax1.plot(iters_k, results[name]['BRC'], 
                    marker='o', label=name, linewidth=2.5, 
                    markersize=8, color=colors[name])
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='BRC=0.5 threshold')
    ax1.set_xlabel('Iteration (k)', fontsize=12)
    ax1.set_ylabel('BRC Index', fontsize=12)
    ax1.set_title('BRC Index Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 有效维度演化
    ax2 = axes[0, 1]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax2.plot(iters_k, results[name]['ED_90'],
                    marker='s', label=name, linewidth=2.5,
                    markersize=8, color=colors[name])
    ax2.set_xlabel('Iteration (k)', fontsize=12)
    ax2.set_ylabel('Effective Dimension (90% var)', fontsize=12)
    ax2.set_title('S2 Effective Dimension', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias标准差
    ax3 = axes[1, 0]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax3.plot(iters_k, results[name]['sigma_bias'],
                    marker='^', label=name, linewidth=2.5,
                    markersize=8, color=colors[name])
    ax3.set_xlabel('Iteration (k)', fontsize=12)
    ax3.set_ylabel('Bias Std Dev (σ)', fontsize=12)
    ax3.set_title('S2 Bias Distribution Width', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. BRC组成部分对比
    ax4 = axes[1, 1]
    # 最后一个checkpoint的对比
    model_names = []
    brc_values = []
    ed_values = []
    sigma_values = []
    
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['BRC']:
            model_names.append(name)
            brc_values.append(results[name]['BRC'][-1])
            ed_values.append(results[name]['ED_90'][-1])
            sigma_values.append(results[name]['sigma_bias'][-1])
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax4.bar(x - width, brc_values, width, label='BRC', color='#1f77b4')
    bars2 = ax4.bar(x, np.array(ed_values)/100, width, label='ED/100', color='#ff7f0e')
    bars3 = ax4.bar(x + width, sigma_values, width, label='σ_bias', color='#2ca02c')
    
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Final BRC Components (iter=50k)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('brc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("="*60)
    print("BRC Evolution Analysis (No Train Log Version)")
    print("="*60)
    
    # 分析所有checkpoints
    results = analyze_all_checkpoints()
    
    # 绘制结果
    plot_brc_evolution(results)
    
    print("\nAnalysis complete!")
    print("Results saved to:")
    print("- brc_analysis_results.pkl")
    print("- brc_analysis.png")

if __name__ == "__main__":
    main()