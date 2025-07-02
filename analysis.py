# analysis_correct.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Import your custom model from root directory
from model import GPT, GPTConfig

# Force CPU to avoid CUDA issues
device = torch.device('cpu')
print(f"Using device: {device}")

def load_checkpoint(checkpoint_path):
    """Load model from checkpoint using your custom GPT implementation"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint
    model_args = checkpoint.get('model_args', {})
    config_args = checkpoint.get('config', {})
    
    # Create config with your parameters
    config = GPTConfig(
        vocab_size=92,
        n_layer=1,
        n_head=1,
        n_embd=120,
        block_size=32,
        bias=False,
        dropout=0.0
    )
    
    # Create model
    model = GPT(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    iteration = checkpoint.get('iter_num', 0)
    return model, iteration

def simple_tokenize(text):
    """Simple tokenizer for your vocabulary"""
    tokens = []
    for word in text.split():
        if word.isdigit() and int(word) < 90:
            tokens.append(int(word))
        elif word == '->':
            tokens.append(90)
        elif word == ':':
            tokens.append(91)
    return tokens

def simple_decode(token_ids):
    """Decode tokens back to text"""
    words = []
    for tid in token_ids:
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        if tid < 90:
            words.append(str(tid))
        elif tid == 90:
            words.append('->')
        elif tid == 91:
            words.append(':')
    return ' '.join(words)

def generate_path(model, prompt, max_new_tokens=10, temperature=0.1):
    """Generate path using your GPT model"""
    model.eval()
    
    # Tokenize prompt
    tokens = simple_tokenize(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate using your model's generate method
    with torch.no_grad():
        generated_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode only the new tokens
    new_tokens = generated_idx[0, len(tokens):].tolist()
    return simple_decode(new_tokens)

def analyze_model_performance(model, test_cases, model_name="model"):
    """Analyze model's compositional ability"""
    print(f"\nAnalyzing {model_name}...")
    
    results = []
    
    for i, (start, target) in enumerate(test_cases[:50]):
        if i % 10 == 0:
            print(f"  Processing {i}/{min(50, len(test_cases))} test cases...")
            
        prompt = f"{start} -> {target} :"
        
        try:
            generated = generate_path(model, prompt)
            
            # Parse the generated path
            numbers = []
            for token in generated.split():
                if token.isdigit():
                    numbers.append(int(token))
            
            # Check if path uses S2
            uses_s2 = any(30 <= n <= 59 for n in numbers)
            valid_path = len(numbers) >= 2
            reaches_target = numbers[-1] == target if numbers else False
            
            results.append({
                'start': start,
                'target': target,
                'generated': generated,
                'numbers': numbers,
                'uses_s2': uses_s2,
                'valid': valid_path,
                'reaches_target': reaches_target,
                'path_length': len(numbers)
            })
        except Exception as e:
            print(f"  Error for {start}->{target}: {e}")
            results.append({
                'start': start,
                'target': target,
                'generated': "ERROR",
                'numbers': [],
                'uses_s2': False,
                'valid': False,
                'reaches_target': False,
                'path_length': 0
            })
    
    # Calculate statistics
    valid_results = [r for r in results if r['valid']]
    success_rate = sum(r['uses_s2'] for r in valid_results) / len(valid_results) if valid_results else 0
    target_rate = sum(r['reaches_target'] for r in valid_results) / len(valid_results) if valid_results else 0
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success rate pie chart
    success_count = sum(r['uses_s2'] for r in results)
    fail_count = len(results) - success_count
    ax1.pie([success_count, fail_count], labels=['Uses S2', 'Direct/Invalid'], 
            autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax1.set_title(f'{model_name}: Path Composition Success')
    
    # 2. Path length distribution
    path_lengths = [r['path_length'] for r in results if r['path_length'] > 0]
    if path_lengths:
        ax2.hist(path_lengths, bins=range(1, min(8, max(path_lengths)+2)), 
                 alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Path Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Path Length Distribution')
        ax2.grid(True, alpha=0.3)
    
    # 3. Example paths
    ax3.axis('off')
    ax3.text(0.05, 0.95, 'Example Generated Paths:', fontsize=14, weight='bold', transform=ax3.transAxes)
    
    # Show first 8 examples
    y_pos = 0.85
    for i, result in enumerate(results[:8]):
        if result['valid']:
            status = "✓" if result['uses_s2'] else "✗"
            color = 'green' if result['uses_s2'] else 'red'
        else:
            status = "✗"
            color = 'gray'
        
        text = f"{status} {result['start']}→{result['target']}: {result['generated'][:30]}"
        ax3.text(0.05, y_pos - i*0.1, text, fontsize=10, color=color, transform=ax3.transAxes)
    
    # 4. Summary statistics
    ax4.axis('off')
    stats_text = [
        (0.8, f'Model: {model_name}', 16, 'bold'),
        (0.65, f'Success Rate (uses S2): {success_rate:.1%}', 14, 'normal'),
        (0.55, f'Reaches Target: {target_rate:.1%}', 12, 'normal'),
        (0.45, f'Valid Paths: {len(valid_results)}/{len(results)}', 12, 'normal'),
    ]
    
    if path_lengths:
        stats_text.append((0.35, f'Avg Path Length: {np.mean(path_lengths):.1f}', 12, 'normal'))
    
    stats_text.append((0.25, f'Total S2 Usage: {success_count}/{len(results)}', 12, 'normal'))
    
    for y, text, size, weight in stats_text:
        ax4.text(0.1, y, text, fontsize=size, weight=weight, transform=ax4.transAxes)
    
    plt.suptitle(f'Compositional Analysis: {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'analysis_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results, success_rate

def analyze_embeddings(model, model_name="model"):
    """Analyze token embeddings using PCA"""
    print(f"Analyzing embeddings for {model_name}...")
    
    # Get token embeddings
    with torch.no_grad():
        embeddings = model.transformer.wte.weight.cpu().numpy()
    
    # Extract embeddings for S1, S2, S3
    s1_embeddings = embeddings[0:30]    # S1: nodes 0-29
    s2_embeddings = embeddings[30:60]   # S2: nodes 30-59
    s3_embeddings = embeddings[60:90]   # S3: nodes 60-89
    
    all_embeddings = np.vstack([s1_embeddings, s2_embeddings, s3_embeddings])
    labels = ['S1']*30 + ['S2']*30 + ['S3']*30
    
    # PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot embeddings
    colors = {'S1': 'blue', 'S2': 'green', 'S3': 'red'}
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[labels[i]], alpha=0.6, s=80, edgecolor='black', linewidth=0.5)
    
    # Add some node labels
    for i in [0, 15, 29, 30, 45, 59, 60, 75, 89]:
        if i < len(embeddings_2d):
            x, y = embeddings_2d[i]
            plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    # Add legend
    for tier, color in colors.items():
        plt.scatter([], [], c=color, label=tier, s=150, edgecolor='black', linewidth=1)
    
    plt.legend(loc='best', fontsize=12)
    plt.title(f'{model_name}: Token Embeddings (PCA)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'embeddings_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate tier separation
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(embeddings_2d, labels)
    print(f"  Silhouette score: {silhouette:.3f}")
    
    return embeddings_2d, silhouette

def compare_checkpoints(checkpoint_paths, test_cases):
    """Compare multiple checkpoints"""
    results_summary = {}
    
    for name, ckpt_path in checkpoint_paths:
        model, iteration = load_checkpoint(ckpt_path)
        if model is not None:
            results, success_rate = analyze_model_performance(model, test_cases, f"{name}_iter{iteration}")
            analyze_embeddings(model, f"{name}_iter{iteration}")
            results_summary[name] = {
                'success_rate': success_rate,
                'iteration': iteration,
                'results': results
            }
    
    # Create comparison plot
    if results_summary:
        plt.figure(figsize=(10, 6))
        
        names = list(results_summary.keys())
        rates = [results_summary[n]['success_rate'] for n in names]
        colors = plt.cm.RdYlGn([r for r in rates])  # Red to green colormap
        
        bars = plt.bar(names, rates, color=colors, edgecolor='black', linewidth=1)
        plt.ylabel('Success Rate (Uses S2)', fontsize=12)
        plt.title('Compositional Generalization: Model Comparison', fontsize=16)
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=11)
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return results_summary

def main():
    """Run complete analysis"""
    print("="*60)
    print("Compositional Generalization Analysis")
    print("Using Custom GPT Implementation")
    print("="*60)
    
    # Generate test cases
    test_cases = []
    for s1 in range(0, 30, 3):  # S1: 0-29
        for s3 in range(60, 90, 3):  # S3: 60-89
            test_cases.append((s1, s3))
    
    print(f"\nGenerated {len(test_cases)} test cases")
    
    # Define checkpoint paths with your exact paths
    checkpoint_configs = [
        ('original', 'out/composition_20250702_063926/ckpt_50000.pt'),
        ('mixed_5', 'out/composition_20250703_004537/ckpt_50000.pt'),
        ('mixed_10', 'out/composition_20250703_011304/ckpt_50000.pt')
    ]
    
    # Run analysis
    results_summary = compare_checkpoints(checkpoint_configs, test_cases)
    
    # Also analyze training progression for original model
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
    
    progression_results = compare_checkpoints(progression_checkpoints, test_cases[:20])
    
    # Create training progression plot
    if progression_results:
        plt.figure(figsize=(10, 6))
        
        iterations = [5000, 15000, 25000, 35000, 50000]
        success_rates = []
        
        for name in ['5k', '15k', '25k', '35k', '50k']:
            if name in progression_results:
                success_rates.append(progression_results[name]['success_rate'])
        
        if success_rates:
            plt.plot(iterations[:len(success_rates)], success_rates, 'o-', 
                    color='darkblue', linewidth=2, markersize=10)
            
            # Add value labels
            for x, y in zip(iterations[:len(success_rates)], success_rates):
                plt.text(x, y + 0.02, f'{y:.1%}', ha='center', fontsize=10)
            
            plt.xlabel('Training Iterations', fontsize=12)
            plt.ylabel('S1→S3 Success Rate (Uses S2)', fontsize=12)
            plt.title('Compositional Ability During Training', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('training_progression.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Generated files:")
    print("- analysis_[model_name]_iter[iteration].png")
    print("- embeddings_[model_name]_iter[iteration].png")
    print("- model_comparison.png")
    print("- training_progression.png")
    print("="*60)

if __name__ == "__main__":
    main()