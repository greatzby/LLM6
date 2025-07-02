# analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    config = {
        "n_embd": 120,
        "n_layer": 1,
        "n_head": 1,
        "vocab_size": 92,
    }
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Resize embeddings to match our vocab size
    model.resize_token_embeddings(92)
    
    # Modify config
    model.config.n_embd = 120
    model.config.n_layer = 1
    model.config.n_head = 1
    model.config.n_positions = 512
    
    # Load checkpoint if exists
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    return model, tokenizer

def load_test_data():
    """Load or generate test S1->S3 paths"""
    test_paths = []
    
    # Generate some test cases
    for s1 in range(0, 30, 3):  # Sample S1 nodes
        for s3 in range(60, 90, 3):  # Sample S3 nodes
            # Create expected path through S2
            s2 = 30 + (s1 + s3) % 30  # Simple rule for S2 selection
            expected_path = f"{s1} {s2} {s3}"
            test_paths.append((s1, s3, expected_path))
    
    return test_paths[:20]  # Return first 20 for quick analysis

def generate_path(model, tokenizer, prompt, max_length=20):
    """Generate a path given a prompt"""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,  # Low temperature for more deterministic output
            do_sample=True
        )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()

def extract_attention_patterns(model, tokenizer, test_paths, model_name="original"):
    """Extract and visualize attention patterns for S1->S3 paths"""
    
    model.eval()
    attention_patterns = []
    
    print(f"Extracting attention patterns for {model_name}...")
    
    with torch.no_grad():
        for start, target, expected_path in test_paths:
            # Prepare input
            prompt = f"{start} -> {target} :"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Get model outputs with attention
            outputs = model(input_ids=inputs.input_ids, output_attentions=True)
            
            # Extract attention from the last layer, last token
            if outputs.attentions:
                attention = outputs.attentions[0][0, 0, -1, :].cpu().numpy()
            else:
                # Fallback if attention not available
                attention = np.ones(inputs.input_ids.shape[1]) / inputs.input_ids.shape[1]
            
            # Generate path
            generated_path = generate_path(model, tokenizer, prompt)
            
            # Extract numbers from generated path
            try:
                path_numbers = [int(x) for x in generated_path.split() if x.isdigit()]
                contains_s2 = any(30 <= n <= 59 for n in path_numbers)
            except:
                contains_s2 = False
            
            attention_patterns.append({
                'query': f"{start}->{target}",
                'attention': attention,
                'generated': generated_path,
                'expected': expected_path,
                'contains_s2': contains_s2
            })
    
    # Visualize attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Group by success/failure
    successful = [p for p in attention_patterns if p['contains_s2']]
    failed = [p for p in attention_patterns if not p['contains_s2']]
    
    # Plot average attention for successful paths
    if successful:
        avg_success_attn = np.mean([p['attention'] for p in successful], axis=0)
        axes[0, 0].bar(range(len(avg_success_attn)), avg_success_attn)
        axes[0, 0].set_title(f'{model_name}: Avg Attention (Successful Paths)')
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Attention Weight')
    
    # Plot average attention for failed paths
    if failed:
        avg_fail_attn = np.mean([p['attention'] for p in failed], axis=0)
        axes[0, 1].bar(range(len(avg_fail_attn)), avg_fail_attn)
        axes[0, 1].set_title(f'{model_name}: Avg Attention (Failed Paths)')
        axes[0, 1].set_xlabel('Token Position')
        axes[0, 1].set_ylabel('Attention Weight')
    else:
        axes[0, 1].text(0.5, 0.5, 'No failed paths', ha='center', va='center')
        axes[0, 1].set_title(f'{model_name}: Avg Attention (Failed Paths)')
    
    # Heatmap of individual attention patterns
    all_attns = [p['attention'] for p in attention_patterns[:10]]
    if all_attns:
        # Pad attention vectors to same length
        max_len = max(len(a) for a in all_attns)
        padded_attns = [np.pad(a, (0, max_len - len(a))) for a in all_attns]
        sns.heatmap(padded_attns, ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title(f'{model_name}: Attention Heatmap (First 10 queries)')
        axes[1, 0].set_xlabel('Token Position')
        axes[1, 0].set_ylabel('Query Index')
    
    # Summary statistics
    success_rate = len(successful) / len(attention_patterns) if attention_patterns else 0
    axes[1, 1].text(0.1, 0.8, f"Model: {model_name}", fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.6, f"Success rate: {success_rate:.1%}", fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Total queries: {len(attention_patterns)}", fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Contains S2: {len(successful)}/{len(attention_patterns)}", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'attention_analysis_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return attention_patterns

def analyze_representations(model, tokenizer, model_name="original", checkpoint_iter=50000):
    """Analyze how the model represents S1, S2, S3 nodes"""
    
    print(f"Analyzing representations for {model_name} at iteration {checkpoint_iter}...")
    
    model.eval()
    
    # Prepare node tokens
    s1_nodes = list(range(0, 30))
    s2_nodes = list(range(30, 60))
    s3_nodes = list(range(60, 90))
    
    embeddings = {
        'S1': [],
        'S2': [],
        'S3': []
    }
    
    with torch.no_grad():
        # Extract embeddings for each node
        embed_layer = model.transformer.wte if hasattr(model, 'transformer') else model.get_input_embeddings()
        
        for node in s1_nodes:
            # Convert node number to token id
            token_ids = tokenizer(str(node), return_tensors='pt').input_ids.to(device)
            hidden = embed_layer(token_ids)
            embeddings['S1'].append(hidden.mean(dim=1).squeeze().cpu().numpy())
            
        for node in s2_nodes:
            token_ids = tokenizer(str(node), return_tensors='pt').input_ids.to(device)
            hidden = embed_layer(token_ids)
            embeddings['S2'].append(hidden.mean(dim=1).squeeze().cpu().numpy())
            
        for node in s3_nodes:
            token_ids = tokenizer(str(node), return_tensors='pt').input_ids.to(device)
            hidden = embed_layer(token_ids)
            embeddings['S3'].append(hidden.mean(dim=1).squeeze().cpu().numpy())
    
    # Combine all embeddings
    all_embeddings = np.vstack([
        np.array(embeddings['S1']),
        np.array(embeddings['S2']),
        np.array(embeddings['S3'])
    ])
    
    labels = ['S1']*30 + ['S2']*30 + ['S3']*30
    
    # PCA visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    colors = {'S1': 'blue', 'S2': 'green', 'S3': 'red'}
    for i, (x, y) in enumerate(embeddings_2d):
        ax1.scatter(x, y, c=colors[labels[i]], alpha=0.6, s=50)
    
    ax1.set_title(f'{model_name} (iter {checkpoint_iter}): PCA of Node Embeddings')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    
    # Add legend
    for tier, color in colors.items():
        ax1.scatter([], [], c=color, label=tier, s=100)
    ax1.legend()
    
    # Calculate separation metrics
    silhouette = silhouette_score(embeddings_2d, labels)
    
    # Calculate centroid distances
    centroids = {}
    for tier in ['S1', 'S2', 'S3']:
        mask = [l == tier for l in labels]
        centroids[tier] = embeddings_2d[mask].mean(axis=0)
    
    s1_s2_dist = np.linalg.norm(centroids['S1'] - centroids['S2'])
    s2_s3_dist = np.linalg.norm(centroids['S2'] - centroids['S3'])
    s1_s3_dist = np.linalg.norm(centroids['S1'] - centroids['S3'])
    
    # Metrics text
    ax2.text(0.1, 0.8, f"Model: {model_name}", fontsize=16, weight='bold')
    ax2.text(0.1, 0.65, f"Silhouette Score: {silhouette:.3f}", fontsize=14)
    ax2.text(0.1, 0.5, f"S1-S2 distance: {s1_s2_dist:.3f}", fontsize=14)
    ax2.text(0.1, 0.4, f"S2-S3 distance: {s2_s3_dist:.3f}", fontsize=14)
    ax2.text(0.1, 0.3, f"S1-S3 distance: {s1_s3_dist:.3f}", fontsize=14)
    ax2.text(0.1, 0.15, f"Compositionality ratio: {s1_s3_dist/(s1_s2_dist+s2_s3_dist):.3f}", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'representation_analysis_{model_name}_iter{checkpoint_iter}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return embeddings, silhouette

def analyze_path_patterns(model, tokenizer, test_set, model_name="original"):
    """Analyze path generation patterns and diversity"""
    
    print(f"Analyzing path patterns for {model_name}...")
    
    model.eval()
    path_stats = {
        'unique_paths': set(),
        'path_lengths': [],
        'uses_s2': 0,
        'direct_jumps': 0,
        'common_patterns': {}
    }
    
    for start, target, _ in test_set[:50]:  # Analyze 50 test cases
        prompt = f"{start} -> {target} :"
        
        # Generate path
        generated = generate_path(model, tokenizer, prompt)
        
        # Analyze path
        try:
            path_nodes = [int(x) for x in generated.split() if x.isdigit()]
            if path_nodes:
                path_stats['path_lengths'].append(len(path_nodes))
                path_stats['unique_paths'].add(tuple(path_nodes))
                
                # Check if uses S2
                if any(30 <= n <= 59 for n in path_nodes):
                    path_stats['uses_s2'] += 1
                else:
                    path_stats['direct_jumps'] += 1
                
                # Track common S2 nodes
                s2_nodes = [n for n in path_nodes if 30 <= n <= 59]
                for node in s2_nodes:
                    path_stats['common_patterns'][node] = path_stats['common_patterns'].get(node, 0) + 1
        except:
            pass
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Path length distribution
    if path_stats['path_lengths']:
        axes[0, 0].hist(path_stats['path_lengths'], bins=range(2, max(path_stats['path_lengths'])+2), alpha=0.7)
        axes[0, 0].set_xlabel('Path Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{model_name}: Path Length Distribution')
    
    # S2 usage pie chart
    total = path_stats['uses_s2'] + path_stats['direct_jumps']
    if total > 0:
        axes[0, 1].pie([path_stats['uses_s2'], path_stats['direct_jumps']], 
                       labels=['Uses S2', 'Direct Jump'], autopct='%1.1f%%',
                       colors=['green', 'red'])
        axes[0, 1].set_title(f'{model_name}: S2 Usage')
    
    # Most common S2 nodes
    if path_stats['common_patterns']:
        top_s2 = sorted(path_stats['common_patterns'].items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, counts = zip(*top_s2)
        axes[1, 0].bar(range(len(nodes)), counts)
        axes[1, 0].set_xticks(range(len(nodes)))
        axes[1, 0].set_xticklabels([str(n) for n in nodes], rotation=45)
        axes[1, 0].set_xlabel('S2 Node')
        axes[1, 0].set_ylabel('Usage Count')
        axes[1, 0].set_title(f'{model_name}: Most Used S2 Nodes')
    
    # Summary stats
    axes[1, 1].text(0.1, 0.8, f"Model: {model_name}", fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.65, f"Unique paths: {len(path_stats['unique_paths'])}", fontsize=12)
    if path_stats['path_lengths']:
        axes[1, 1].text(0.1, 0.5, f"Avg path length: {np.mean(path_stats['path_lengths']):.2f}", fontsize=12)
    if total > 0:
        axes[1, 1].text(0.1, 0.35, f"S2 usage rate: {path_stats['uses_s2']/total:.1%}", fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Total paths analyzed: {total}", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'path_patterns_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return path_stats

def create_comparison_plot(all_results):
    """Create a comparison visualization across all models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(all_results.keys())
    
    # Success rates
    success_rates = []
    for model in models:
        attention_results = all_results[model]['attention']
        successful = [p for p in attention_results if p['contains_s2']]
        success_rates.append(len(successful) / len(attention_results))
    
    axes[0].bar(models, success_rates, color=['red', 'orange', 'green'])
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('S1→S3 Composition Success Rate')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center')
    
    # Path diversity
    unique_paths = []
    for model in models:
        path_results = all_results[model]['paths']
        unique_paths.append(len(path_results['unique_paths']))
    
    axes[1].bar(models, unique_paths, color=['red', 'orange', 'green'])
    axes[1].set_ylabel('Number of Unique Paths')
    axes[1].set_title('Path Generation Diversity')
    
    # Representation separation (if available)
    silhouette_scores = []
    for model in models:
        if 'silhouette' in all_results[model]:
            silhouette_scores.append(all_results[model]['silhouette'])
        else:
            silhouette_scores.append(0)
    
    axes[2].bar(models, silhouette_scores, color=['red', 'orange', 'green'])
    axes[2].set_ylabel('Silhouette Score')
    axes[2].set_title('S1/S2/S3 Representation Separation')
    
    plt.suptitle('Model Comparison: Compositional Generalization', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Run all analyses"""
    
    print("=" * 60)
    print("Compositional Generalization Analysis")
    print("=" * 60)
    
    # Define model paths - adjust these to your actual checkpoint paths
    model_configs = [
        ('original', 'out/composition_20250702_063926/ckpt_50000.pt'),
        ('mixed_5', 'out/composition_20250703_004537/ckpt_50000.pt'),
        ('mixed_10', 'out/composition_20250703_011304/ckpt_50000.pt')
    ]
    
    # If checkpoints don't exist, use a dummy model for demonstration
    print("\nNote: Using dummy models for demonstration. Replace with actual checkpoint paths.")
    
    # Load test data
    test_data = load_test_data()
    print(f"\nLoaded {len(test_data)} test cases")
    
    all_results = {}
    
    for model_name, checkpoint_path in model_configs:
        print(f"\n{'='*40}")
        print(f"Analyzing {model_name} model")
        print(f"{'='*40}")
        
        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path)
        except:
            print(f"Warning: Could not load {checkpoint_path}, using base GPT2 instead")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            model.resize_token_embeddings(92)
            model.to(device)
        
        # Run analyses
        attention_results = extract_attention_patterns(model, tokenizer, test_data, model_name)
        embeddings, silhouette = analyze_representations(model, tokenizer, model_name)
        path_results = analyze_path_patterns(model, tokenizer, test_data, model_name)
        
        all_results[model_name] = {
            'attention': attention_results,
            'embeddings': embeddings,
            'silhouette': silhouette,
            'paths': path_results
        }
    
    # Create comparison plot
    print("\nCreating comparison visualizations...")
    create_comparison_plot(all_results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Generated files:")
    print("- attention_analysis_[model_name].png")
    print("- representation_analysis_[model_name]_iter50000.png")
    print("- path_patterns_[model_name].png")
    print("- model_comparison.png")
    print("="*60)

if __name__ == "__main__":
    main()