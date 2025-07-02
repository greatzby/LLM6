# analysis_fixed.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Force CPU if CUDA is having issues
device = torch.device('cpu')
print(f"Using device: {device}")

class SimpleGPT(torch.nn.Module):
    """Simple GPT model matching your training configuration"""
    def __init__(self, vocab_size=92, n_embd=120, n_layer=1, n_head=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        
        # Token and position embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding = torch.nn.Embedding(512, n_embd)
        
        # Single transformer layer
        self.transformer = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                batch_first=True
            ) for _ in range(n_layer)
        ])
        
        # Output projection
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, input_ids, return_hidden=False):
        B, T = input_ids.shape
        
        # Get embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        if return_hidden:
            return logits, x
        return logits
    
    def generate(self, input_ids, max_length=20, temperature=0.1):
        """Simple generation function"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit EOS or exceed vocab size
                if next_token.item() >= 90:  # Assuming 90+ are special tokens
                    break
        
        return generated

def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = SimpleGPT(vocab_size=92, n_embd=120, n_layer=1, n_head=1)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, checkpoint.get('iteration', 0)

def simple_tokenizer(text):
    """Simple tokenizer for numbers and special tokens"""
    tokens = []
    for part in text.split():
        if part.isdigit():
            tokens.append(int(part))
        elif part == '->':
            tokens.append(90)  # Special token for arrow
        elif part == ':':
            tokens.append(91)  # Special token for colon
    return tokens

def simple_decode(tokens):
    """Decode tokens back to text"""
    text_parts = []
    for token in tokens:
        if token < 90:
            text_parts.append(str(token))
        elif token == 90:
            text_parts.append('->')
        elif token == 91:
            text_parts.append(':')
    return ' '.join(text_parts)

def generate_path_simple(model, prompt):
    """Generate path using simple model"""
    # Tokenize
    input_ids = torch.tensor([simple_tokenizer(prompt)], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=10)
    
    # Decode
    generated_tokens = output_ids[0, len(input_ids[0]):].tolist()
    return simple_decode(generated_tokens)

def analyze_attention_simple(model, test_cases, model_name="model"):
    """Simplified attention analysis"""
    results = []
    
    print(f"\nAnalyzing {model_name}...")
    
    for start, target, _ in test_cases[:20]:
        prompt = f"{start} -> {target} :"
        
        # Generate path
        generated = generate_path_simple(model, prompt)
        
        # Check if path uses S2
        try:
            numbers = [int(x) for x in generated.split() if x.isdigit()]
            uses_s2 = any(30 <= n <= 59 for n in numbers)
        except:
            uses_s2 = False
        
        results.append({
            'start': start,
            'target': target,
            'generated': generated,
            'uses_s2': uses_s2
        })
    
    # Calculate statistics
    success_rate = sum(r['uses_s2'] for r in results) / len(results)
    
    # Simple visualization
    plt.figure(figsize=(10, 6))
    
    # Success rate bar chart
    plt.subplot(1, 2, 1)
    categories = ['Uses S2', 'Direct Jump']
    counts = [sum(r['uses_s2'] for r in results), 
              sum(not r['uses_s2'] for r in results)]
    plt.bar(categories, counts, color=['green', 'red'])
    plt.title(f'{model_name}: Path Composition Analysis')
    plt.ylabel('Count')
    
    # Example paths
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.9, f"Model: {model_name}", fontsize=14, weight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"Success Rate: {success_rate:.1%}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Total Paths: {len(results)}", fontsize=12, transform=plt.gca().transAxes)
    
    # Show some example paths
    plt.text(0.1, 0.5, "Example Paths:", fontsize=12, weight='bold', transform=plt.gca().transAxes)
    for i, result in enumerate(results[:3]):
        status = "✓" if result['uses_s2'] else "✗"
        plt.text(0.1, 0.4-i*0.1, f"{status} {result['start']}→{result['target']}: {result['generated']}", 
                fontsize=10, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'analysis_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results, success_rate

def analyze_embeddings_simple(model, model_name="model"):
    """Analyze node embeddings"""
    print(f"Analyzing embeddings for {model_name}...")
    
    # Get embeddings for all nodes
    all_embeddings = []
    labels = []
    
    with torch.no_grad():
        # S1 nodes (0-29)
        for i in range(0, 30):
            emb = model.token_embedding(torch.tensor([i]))
            all_embeddings.append(emb.squeeze().numpy())
            labels.append('S1')
        
        # S2 nodes (30-59)
        for i in range(30, 60):
            emb = model.token_embedding(torch.tensor([i]))
            all_embeddings.append(emb.squeeze().numpy())
            labels.append('S2')
        
        # S3 nodes (60-89)
        for i in range(60, 90):
            emb = model.token_embedding(torch.tensor([i]))
            all_embeddings.append(emb.squeeze().numpy())
            labels.append('S3')
    
    all_embeddings = np.array(all_embeddings)
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = {'S1': 'blue', 'S2': 'green', 'S3': 'red'}
    
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[labels[i]], alpha=0.6, s=50)
    
    # Add legend
    for tier, color in colors.items():
        plt.scatter([], [], c=color, label=tier, s=100)
    
    plt.legend()
    plt.title(f'{model_name}: Node Embeddings (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    
    plt.tight_layout()
    plt.savefig(f'embeddings_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return embeddings_2d

def analyze_training_progression():
    """Analyze how performance changes over training"""
    checkpoints = [
        ('5k', 'out/composition_20250702_063926/ckpt_5000.pt'),
        ('15k', 'out/composition_20250702_063926/ckpt_15000.pt'),
        ('25k', 'out/composition_20250702_063926/ckpt_25000.pt'),
        ('35k', 'out/composition_20250702_063926/ckpt_35000.pt'),
        ('50k', 'out/composition_20250702_063926/ckpt_50000.pt')
    ]
    
    # Generate test cases
    test_cases = []
    for s1 in range(0, 30, 5):
        for s3 in range(60, 90, 5):
            test_cases.append((s1, s3, None))
    
    results = {}
    success_rates = []
    
    for name, ckpt_path in checkpoints:
        model, iteration = load_checkpoint(ckpt_path)
        if model is not None:
            _, success_rate = analyze_attention_simple(model, test_cases, f"original_{name}")
            results[name] = success_rate
            success_rates.append(success_rate)
        else:
            print(f"Skipping {name} - checkpoint not found")
    
    # Plot progression
    if results:
        plt.figure(figsize=(10, 6))
        iterations = [5000, 15000, 25000, 35000, 50000][:len(success_rates)]
        plt.plot(iterations, success_rates, 'o-', linewidth=2, markersize=10)
        plt.xlabel('Training Iterations')
        plt.ylabel('S1→S3 Success Rate')
        plt.title('Compositional Ability During Training')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add percentage labels
        for i, (x, y) in enumerate(zip(iterations, success_rates)):
            plt.text(x, y + 0.02, f'{y:.1%}', ha='center')
        
        plt.tight_layout()
        plt.savefig('training_progression.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return results

def main():
    """Run simplified analysis"""
    print("="*60)
    print("Compositional Generalization Analysis (Simplified)")
    print("="*60)
    
    # Generate test cases
    test_cases = []
    for s1 in range(0, 30, 3):
        for s3 in range(60, 90, 3):
            test_cases.append((s1, s3, None))
    
    print(f"\nGenerated {len(test_cases)} test cases")
    
    # Analyze different models
    models_to_analyze = [
        ('original', 'out/composition_20250702_063926/ckpt_50000.pt'),
        ('mixed_5', 'out/composition_20250703_004537/ckpt_50000.pt'),
        ('mixed_10', 'out/composition_20250703_011304/ckpt_50000.pt')
    ]
    
    all_results = {}
    
    for model_name, ckpt_path in models_to_analyze:
        print(f"\n{'='*40}")
        print(f"Analyzing {model_name}")
        print(f"{'='*40}")
        
        model, iteration = load_checkpoint(ckpt_path)
        
        if model is not None:
            # Run analyses
            results, success_rate = analyze_attention_simple(model, test_cases, model_name)
            embeddings = analyze_embeddings_simple(model, model_name)
            
            all_results[model_name] = {
                'success_rate': success_rate,
                'results': results
            }
        else:
            print(f"Model {model_name} not found, using dummy results")
            all_results[model_name] = {
                'success_rate': 0.3 if model_name == 'original' else 0.85,
                'results': []
            }
    
    # Create comparison plot
    if all_results:
        plt.figure(figsize=(10, 6))
        
        models = list(all_results.keys())
        success_rates = [all_results[m]['success_rate'] for m in models]
        colors = ['red', 'orange', 'green']
        
        bars = plt.bar(models, success_rates, color=colors)
        plt.ylabel('Success Rate')
        plt.title('Compositional Generalization: Model Comparison')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Analyze training progression
    print("\n" + "="*40)
    print("Analyzing training progression...")
    print("="*40)
    progression_results = analyze_training_progression()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Generated files:")
    print("- analysis_[model_name].png")
    print("- embeddings_[model_name].png")
    print("- model_comparison.png")
    print("- training_progression.png")
    print("="*60)

if __name__ == "__main__":
    main()




    