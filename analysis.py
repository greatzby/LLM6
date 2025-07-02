# analysis_final.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import networkx as nx
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from model import GPT, GPTConfig

# 创建输出目录
output_dir = f'analysis_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: {output_dir}/")

device = torch.device('cpu')
print(f"Using device: {device}")

def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model args
    model_args = checkpoint.get('model_args', {})
    
    # Create model
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    iteration = checkpoint.get('iter_num', 0)
    return model, iteration

def load_data_info(data_dir='data/simple_graph/composition_90'):
    """Load data information"""
    # Load stages
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    # Load meta
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Load graph
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    return stages, meta, G

@torch.no_grad()
def generate_path(model, source, target, stoi, itos, vocab_size, 
                  temperature=0.1, top_k=10, max_new_tokens=30):
    """Generate path using the model - matching training code"""
    model.eval()
    
    is_token_level = vocab_size > 50
    
    if is_token_level:
        # Token级编码 - 匹配训练代码
        prompt = f"{source} {target} {source}"
        prompt_tokens = prompt.split()
        
        prompt_ids = []
        for token in prompt_tokens:
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # 生成
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
        # 解码完整序列
        all_numbers = []
        for tid in y[0].tolist():
            if tid == 1:  # EOS
                break
            if tid in itos:
                try:
                    all_numbers.append(int(itos[tid]))
                except:
                    pass
        
        # 路径从第3个位置开始（跳过prompt的3个token）
        if len(all_numbers) >= 3:
            generated_path = all_numbers[2:]
        else:
            generated_path = []
    else:
        # Character-level encoding
        prompt_str = f"{source} {target}"
        prompt_ids = []
        for char in prompt_str:
            if char in stoi:
                prompt_ids.append(stoi[char])
        
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
        
        # Decode
        chars = []
        for tid in y[0].tolist():
            if tid == 1:  # newline
                break
            if tid in itos and tid > 1:
                chars.append(itos[tid])
        
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
    
    return generated_path

def analyze_model_performance(model, test_cases, stages, meta, G, model_name="model"):
    """Analyze model's compositional ability"""
    print(f"\nAnalyzing {model_name}...")
    
    S1, S2, S3 = stages
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    
    results = []
    
    for i, (source, target) in enumerate(test_cases[:50]):
        if i % 10 == 0:
            print(f"  Processing {i}/{min(50, len(test_cases))} test cases...")
        
        try:
            # Generate path
            generated_path = generate_path(model, source, target, stoi, itos, vocab_size)
            
            # Validate path
            uses_s2 = any(30 <= n <= 59 for n in generated_path)
            valid_path = len(generated_path) >= 2
            
            if valid_path and generated_path:
                starts_correctly = generated_path[0] == source
                reaches_target = generated_path[-1] == target
                
                # Check if path is valid in graph
                path_valid = True
                if len(generated_path) >= 2:
                    for j in range(len(generated_path)-1):
                        if not G.has_edge(str(generated_path[j]), str(generated_path[j+1])):
                            path_valid = False
                            break
                else:
                    path_valid = False
            else:
                starts_correctly = False
                reaches_target = False
                path_valid = False
            
            results.append({
                'source': source,
                'target': target,
                'path': generated_path,
                'uses_s2': uses_s2,
                'valid': valid_path,
                'reaches_target': reaches_target,
                'starts_correctly': starts_correctly,
                'path_length': len(generated_path)
            })
            
        except Exception as e:
            print(f"  Error for {source}->{target}: {e}")
            results.append({
                'source': source,
                'target': target,
                'path': [],
                'uses_s2': False,
                'valid': False,
                'reaches_target': False,
                'starts_correctly': False,
                'path_length': 0
            })
    
    # Calculate statistics
    valid_results = [r for r in results if r['valid'] and r['starts_correctly'] and r['reaches_target']]
    success_results = [r for r in valid_results if r['uses_s2']]
    success_rate = len(success_results) / len(results) if results else 0
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success rate pie chart
    success_count = len(success_results)
    fail_count = len(results) - success_count
    
    ax1.pie([success_count, fail_count], labels=['Success (via S2)', 'Failed/Direct'], 
            autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax1.set_title(f'{model_name}: S1→S3 Path Composition')
    
    # 2. Path length distribution
    valid_path_lengths = [r['path_length'] for r in valid_results]
    if valid_path_lengths:
        ax2.hist(valid_path_lengths, bins=range(2, min(8, max(valid_path_lengths)+2)), 
                 alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Path Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Valid Path Length Distribution')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No valid paths', ha='center', va='center')
        ax2.set_title('Path Length Distribution')
    
    # 3. Example paths
    ax3.axis('off')
    ax3.text(0.05, 0.95, 'Example Generated Paths:', fontsize=14, weight='bold', transform=ax3.transAxes)
    
    y_pos = 0.85
    shown = 0
    # Show some successful and failed examples
    for result in results[:15]:
        if shown >= 8:
            break
        if result['path']:
            if result['valid'] and result['reaches_target']:
                status = "✓" if result['uses_s2'] else "○"
                color = 'green' if result['uses_s2'] else 'orange'
            else:
                status = "✗"
                color = 'red'
            
            path_str = ' → '.join(map(str, result['path'][:5]))
            if len(result['path']) > 5:
                path_str += ' → ...'
            text = f"{status} {result['source']}→{result['target']}: {path_str}"
            ax3.text(0.05, y_pos - shown*0.1, text, fontsize=9, color=color, transform=ax3.transAxes)
            shown += 1
    
    # 4. Summary statistics
    ax4.axis('off')
    stats_text = [
        (0.8, f'Model: {model_name}', 16, 'bold'),
        (0.65, f'Success Rate (S1→S3 via S2): {success_rate:.1%}', 14, 'normal'),
        (0.55, f'Valid Paths: {len(valid_results)}/{len(results)}', 12, 'normal'),
        (0.45, f'Uses S2: {len(success_results)}/{len(valid_results) if valid_results else 0}', 12, 'normal'),
    ]
    
    if valid_path_lengths:
        stats_text.append((0.35, f'Avg Valid Path Length: {np.mean(valid_path_lengths):.1f}', 12, 'normal'))
    
    for y, text, size, weight in stats_text:
        ax4.text(0.1, y, text, fontsize=size, weight=weight, transform=ax4.transAxes)
    
    plt.suptitle(f'Compositional Analysis: {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'analysis_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return results, success_rate

def analyze_embeddings(model, stages, model_name="model"):
    """Analyze token embeddings"""
    print(f"Analyzing embeddings for {model_name}...")
    
    S1, S2, S3 = stages
    
    with torch.no_grad():
        embeddings = model.transformer.wte.weight.cpu().numpy()
    
    # Extract embeddings for S1, S2, S3 nodes
    s1_embeddings = embeddings[S1]
    s2_embeddings = embeddings[S2]
    s3_embeddings = embeddings[S3]
    
    all_embeddings = np.vstack([s1_embeddings, s2_embeddings, s3_embeddings])
    labels = ['S1']*30 + ['S2']*30 + ['S3']*30
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    plt.figure(figsize=(12, 8))
    
    colors = {'S1': 'blue', 'S2': 'green', 'S3': 'red'}
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[labels[i]], alpha=0.6, s=80, edgecolor='black', linewidth=0.5)
    
    # Add some labels
    for i in [0, 15, 29, 30, 45, 59, 60, 75, 89]:
        idx = i if i < 30 else (i-30+30) if i < 60 else (i-60+60)
        if idx < len(embeddings_2d):
            x, y = embeddings_2d[idx]
            plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    for tier, color in colors.items():
        plt.scatter([], [], c=color, label=tier, s=150, edgecolor='black', linewidth=1)
    
    plt.legend(loc='best', fontsize=12)
    plt.title(f'{model_name}: Token Embeddings (PCA)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'embeddings_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(embeddings_2d, labels)
    print(f"  Silhouette score: {silhouette:.3f}")
    
    return embeddings_2d, silhouette

def plot_training_progression(progression_results):
    """Plot training progression with better visualization"""
    if not progression_results:
        return
    
    plt.figure(figsize=(14, 6))
    
    # Extract data
    iterations = []
    success_rates = []
    names_ordered = ['5k', '15k', '25k', '35k', '50k']
    
    for name in names_ordered:
        if name in progression_results:
            iter_num = int(name[:-1]) * 1000
            iterations.append(iter_num)
            success_rates.append(progression_results[name]['success_rate'])
    
    if iterations and success_rates:
        # Main plot
        plt.subplot(1, 2, 1)
        plt.plot(iterations, success_rates, 'o-', color='darkblue', linewidth=3, markersize=12)
        
        # Add value labels
        for x, y in zip(iterations, success_rates):
            plt.text(x, y + 0.02, f'{y:.1%}', ha='center', fontsize=11, weight='bold')
        
        # Find peak
        peak_idx = success_rates.index(max(success_rates))
        peak_iter = iterations[peak_idx]
        peak_rate = success_rates[peak_idx]
        
        # Highlight peak
        plt.scatter([peak_iter], [peak_rate], color='green', s=200, zorder=5, 
                   marker='*', edgecolor='black', linewidth=2)
        
        plt.xlabel('Training Iterations', fontsize=14)
        plt.ylabel('S1→S3 Success Rate (via S2)', fontsize=14)
        plt.title('Compositional Ability During Training', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        # Detail plot
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        plt.text(0.1, 0.85, 'Training Analysis', fontsize=18, weight='bold')
        plt.text(0.1, 0.70, f'Peak Performance: {peak_rate:.1%}', fontsize=14, color='green')
        plt.text(0.1, 0.60, f'Peak at: {peak_iter:,} iterations', fontsize=12)
        plt.text(0.1, 0.50, f'Final Performance: {success_rates[-1]:.1%}', fontsize=14)
        plt.text(0.1, 0.40, f'Performance Drop: {peak_rate - success_rates[-1]:.1%}', fontsize=12, 
                color='red' if peak_rate - success_rates[-1] > 0.1 else 'black')
        
        # Interpretation
        if peak_rate > 0.7:
            plt.text(0.1, 0.25, '✓ Model demonstrated compositional ability', fontsize=12, color='green')
        elif peak_rate > 0.3:
            plt.text(0.1, 0.25, '○ Model showed partial compositional ability', fontsize=12, color='orange')
        else:
            plt.text(0.1, 0.25, '✗ Model failed to learn composition', fontsize=12, color='red')
        
        if peak_rate - success_rates[-1] > 0.3:
            plt.text(0.1, 0.15, '⚠ Severe degradation with overtraining', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progression.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Run complete analysis"""
    print("="*60)
    print("Compositional Generalization Analysis")
    print("="*60)
    
    # Load data information
    stages, meta, G = load_data_info()
    S1, S2, S3 = stages
    
    # Generate test cases (S1->S3)
    test_cases = []
    for s1 in S1[::3]:  # Every 3rd node
        for s3 in S3[::3]:
            if nx.has_path(G, str(s1), str(s3)):
                test_cases.append((s1, s3))
    
    print(f"\nGenerated {len(test_cases)} S1→S3 test cases")
    
    # Main model analysis
    checkpoint_configs = [
        ('original', 'out/composition_20250702_063926/ckpt_50000.pt'),
        ('mixed_5', 'out/composition_20250703_004537/ckpt_50000.pt'),
        ('mixed_10', 'out/composition_20250703_011304/ckpt_50000.pt')
    ]
    
    results_summary = {}
    
    for name, ckpt_path in checkpoint_configs:
        print(f"\n{'='*40}")
        print(f"Analyzing {name}")
        print(f"{'='*40}")
        
        model, iteration = load_checkpoint(ckpt_path)
        if model is not None:
            results, success_rate = analyze_model_performance(
                model, test_cases, stages, meta, G, f"{name}_iter{iteration}"
            )
            analyze_embeddings(model, stages, f"{name}_iter{iteration}")
            results_summary[name] = {
                'success_rate': success_rate,
                'iteration': iteration
            }
    
    # Training progression
    print("\n" + "="*40)
    print("Analyzing training progression...")
    print("="*40)
    
    progression_checkpoints = [
        ('5k', 'out/composition_20250702_063926/ckpt_5000.pt'),
        ('15k', 'out/composition_20250702_063926/ckpt_15000.pt'),
        ('25k', 'out/composition_20250702_063926/ckpt_25000.pt'),
        ('35k', 'out/composition_20250702_063926/ckpt_35000.pt'),
        ('50k', 'out/composition_20250702_063926/ckpt_50000.pt')
    ]
    
    progression_results = {}
    for name, ckpt_path in progression_checkpoints:
        model, iteration = load_checkpoint(ckpt_path)
        if model is not None:
            # Use fewer test cases for speed
            results, success_rate = analyze_model_performance(
                model, test_cases[:30], stages, meta, G, f"{name}_iter{iteration}"
            )
            analyze_embeddings(model, stages, f"{name}_iter{iteration}")
            progression_results[name] = {
                'success_rate': success_rate,
                'iteration': iteration
            }
    
    # Model comparison
    if results_summary:
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(2, 2, 1)
        names = list(results_summary.keys())
        rates = [results_summary[n]['success_rate'] for n in names]
        colors = ['red', 'orange', 'green']
        
        bars = plt.bar(names, rates, color=colors, edgecolor='black', linewidth=2)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('S1→S3 Compositional Success', fontsize=14)
        plt.ylim(0, 1)
        
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Summary text
        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.text(0.1, 0.8, 'Model Comparison', fontsize=16, weight='bold')
        
        for i, (name, data) in enumerate(results_summary.items()):
            y_pos = 0.6 - i*0.15
            rate = data['success_rate']
            plt.text(0.1, y_pos, f'{name}: {rate:.1%}', fontsize=12)
            
            if rate > 0.8:
                plt.text(0.5, y_pos, '✓ Excellent', fontsize=12, color='green')
            elif rate > 0.5:
                plt.text(0.5, y_pos, '○ Good', fontsize=12, color='orange')
            else:
                plt.text(0.5, y_pos, '✗ Poor', fontsize=12, color='red')
        
        # Key findings
        plt.subplot(2, 1, 2)
        plt.axis('off')
        plt.text(0.1, 0.9, 'Key Findings:', fontsize=16, weight='bold')
        
        findings = []
        
        if 'original' in results_summary and results_summary['original']['success_rate'] < 0.3:
            findings.append('• Original model fails at compositional generalization')
        
        if 'mixed_5' in results_summary and results_summary['mixed_5']['success_rate'] > 0.7:
            findings.append('• 5% mixed training dramatically improves composition')
        
        if 'mixed_10' in results_summary and results_summary['mixed_10']['success_rate'] > 0.8:
            findings.append('• 10% mixed training achieves excellent performance')
        
        for i, finding in enumerate(findings):
            plt.text(0.1, 0.7 - i*0.1, finding, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot training progression
    plot_training_progression(progression_results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}/")
    print("Generated files:")
    print("- analysis_[model_name].png")
    print("- embeddings_[model_name].png")
    print("- model_comparison.png")
    print("- training_progression.png")
    print("="*60)

if __name__ == "__main__":
    main()