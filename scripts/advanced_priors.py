"""
============================================================================
ADVANCED PRIOR IDEAS: 3 Experiments to Bridge the Gap to SOTA
============================================================================

This script implements and tests 3 key ideas to improve generalization:

1. DYNAMIC SPECTRAL PRIOR
   - Problem: Fixed H=2.44 matches Wine but not Iris (H=1.09) or Digits (H=3.92)
   - Solution: Sample H ~ U(1.0, 4.0) during training for each batch
   - Target: Iris (+7%), global robustness

2. HYBRID-GEOMETRY MIXTURE PRIOR  
   - Problem: Smooth manifold prior fails on geometric/spatial data
   - Solution: Mix 40% Manifold + 40% DAG/Causal + 20% Tree/Step priors
   - Target: Vehicle (+14%), Segment, Digits

3. THERMODYNAMIC TEMPERATURE SCALING
   - Problem: Soft boundaries in prior, but some data has hard margins
   - Solution: Vary τ from 0.01 (hard) to 1.0 (soft) in label generation
   - Target: Vowel (+16%)

Usage:
    python scripts/advanced_priors.py --idea 1  # Dynamic Spectral
    python scripts/advanced_priors.py --idea 2  # Hybrid Mixture
    python scripts/advanced_priors.py --idea 3  # Temperature Scaling
    python scripts/advanced_priors.py --all     # All 3 experiments

Author: Spectral Prior Research
============================================================================
"""

import sys
import os
import torch
from torch import nn
import numpy as np
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# Evaluation imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# IDEA 1: DYNAMIC SPECTRAL PRIOR
# ============================================================================

class DynamicSpectralPrior:
    """
    Prior that samples H_target from U(1.0, 4.0) for each batch.
    This teaches the model In-Context Spectral Adaptation.
    """
    def __init__(self, n_classes=3, device='cpu', H_range=(1.0, 4.0)):
        self.n_classes = n_classes
        self.device = device
        self.H_range = H_range
        print(f"  DynamicSpectralPrior: H ~ U{H_range}")
    
    def get_batch(self, batch_size, seq_len, n_features, n_classes=None):
        if n_classes is None:
            n_classes = self.n_classes
        
        # Sample H for this batch
        H_target = np.random.uniform(self.H_range[0], self.H_range[1])
        
        # Map H to hidden_dim (controls spectral complexity)
        if H_target < 1.5:
            hidden_dim = 8   # Simple, concentrated spectrum
        elif H_target < 2.5:
            hidden_dim = 32  # Moderate complexity
        elif H_target < 3.5:
            hidden_dim = 64  # High complexity
        else:
            hidden_dim = 128 # Very high complexity
        
        # Generate latent features through random network
        z = torch.randn(batch_size * seq_len, hidden_dim, device=self.device)
        
        # Neural network transformation
        W1 = torch.randn(hidden_dim, n_features, device=self.device) * 0.5
        W2 = torch.randn(n_features, n_features, device=self.device) * 0.3
        
        x = torch.tanh(z @ W1)
        x = x + 0.1 * torch.tanh(x @ W2)  # Skip connection
        x = x.reshape(batch_size, seq_len, n_features)
        
        # Generate labels with random hyperplane
        w = torch.randn(n_features, n_classes, device=self.device)
        logits = x @ w
        y = torch.argmax(logits, dim=-1).float()
        
        single_eval_pos = seq_len // 2
        
        return dict(
            x=x,
            y=y,
            target_y=y,
            single_eval_pos=single_eval_pos
        )


# ============================================================================
# IDEA 2: HYBRID-GEOMETRY MIXTURE PRIOR
# ============================================================================

class HybridMixturePrior:
    """
    Mixture of priors:
    - 40% Manifold (smooth, for Wine/Cancer)
    - 40% DAG/Causal (geometric, for Digits/Vehicle)
    - 20% Tree/Step (hard rules, for Vowel)
    """
    def __init__(self, n_classes=3, device='cpu'):
        self.n_classes = n_classes
        self.device = device
        self.mix_probs = [0.4, 0.4, 0.2]  # Manifold, DAG, Tree
        print(f"  HybridMixturePrior: 40% Manifold, 40% DAG, 20% Tree")
    
    def _manifold_batch(self, batch_size, seq_len, n_features, n_classes):
        """Smooth manifold prior (your current approach)."""
        z = torch.randn(batch_size * seq_len, 32, device=self.device)
        W = torch.randn(32, n_features, device=self.device) * 0.5
        x = torch.tanh(z @ W)
        x = x.reshape(batch_size, seq_len, n_features)
        
        w = torch.randn(n_features, n_classes, device=self.device)
        logits = x @ w
        y = torch.argmax(logits, dim=-1).float()
        return x, y
    
    def _dag_batch(self, batch_size, seq_len, n_features, n_classes):
        """Causal DAG prior (geometric, sharp boundaries)."""
        # Create a random DAG structure
        n_parents = min(3, n_features - 1)
        x = torch.zeros(batch_size * seq_len, n_features, device=self.device)
        
        # Root nodes from Gaussian
        x[:, :n_parents] = torch.randn(batch_size * seq_len, n_parents, device=self.device)
        
        # Child nodes as nonlinear functions of parents
        for i in range(n_parents, n_features):
            parent_idx = np.random.choice(i, min(n_parents, i), replace=False)
            weights = torch.randn(len(parent_idx), device=self.device)
            x[:, i] = torch.relu((x[:, parent_idx] * weights).sum(dim=1)) + 0.1 * torch.randn(batch_size * seq_len, device=self.device)
        
        x = x.reshape(batch_size, seq_len, n_features)
        
        # Sharp decision boundaries
        w = torch.randn(n_features, n_classes, device=self.device)
        logits = x @ w
        y = torch.argmax(logits * 5.0, dim=-1).float()  # Sharper boundaries
        return x, y
    
    def _tree_batch(self, batch_size, seq_len, n_features, n_classes):
        """Tree/Step-function prior (hard rules like decision trees)."""
        x = torch.randn(batch_size * seq_len, n_features, device=self.device)
        x = x.reshape(batch_size, seq_len, n_features)
        
        # Create step function labels based on random thresholds
        thresholds = torch.randn(n_classes - 1, device=self.device)
        thresholds, _ = torch.sort(thresholds)
        
        # Use first feature as decision variable
        feature_idx = np.random.randint(0, n_features)
        y = torch.zeros(batch_size, seq_len, device=self.device)
        
        for c in range(n_classes - 1):
            y += (x[:, :, feature_idx] > thresholds[c]).float()
        
        return x, y
    
    def get_batch(self, batch_size, seq_len, n_features, n_classes=None):
        if n_classes is None:
            n_classes = self.n_classes
        
        # Sample which prior to use
        prior_type = np.random.choice(['manifold', 'dag', 'tree'], p=self.mix_probs)
        
        if prior_type == 'manifold':
            x, y = self._manifold_batch(batch_size, seq_len, n_features, n_classes)
        elif prior_type == 'dag':
            x, y = self._dag_batch(batch_size, seq_len, n_features, n_classes)
        else:
            x, y = self._tree_batch(batch_size, seq_len, n_features, n_classes)
        
        single_eval_pos = seq_len // 2
        return dict(
            x=x,
            y=y,
            target_y=y,
            single_eval_pos=single_eval_pos
        )


# ============================================================================
# IDEA 3: THERMODYNAMIC TEMPERATURE PRIOR
# ============================================================================

class ThermodynamicPrior:
    """
    Prior with variable temperature τ for label generation.
    - Low τ (0.01): Hard, deterministic boundaries (like Vehicle/Segment)
    - High τ (1.0): Soft, probabilistic boundaries (like Wine)
    """
    def __init__(self, n_classes=3, device='cpu', tau_range=(0.01, 1.0)):
        self.n_classes = n_classes
        self.device = device
        self.tau_range = tau_range
        print(f"  ThermodynamicPrior: τ ~ U{tau_range}")
    
    def get_batch(self, batch_size, seq_len, n_features, n_classes=None):
        if n_classes is None:
            n_classes = self.n_classes
        
        # Sample temperature for this batch
        tau = np.random.uniform(self.tau_range[0], self.tau_range[1])
        
        # Generate features
        z = torch.randn(batch_size * seq_len, 32, device=self.device)
        W = torch.randn(32, n_features, device=self.device) * 0.5
        x = torch.tanh(z @ W)
        x = x.reshape(batch_size, seq_len, n_features)
        
        # Generate labels with temperature scaling
        w = torch.randn(n_features, n_classes, device=self.device)
        logits = x @ w
        
        # Temperature-scaled softmax for probabilistic labels
        probs = torch.softmax(logits / tau, dim=-1)
        
        if tau < 0.1:
            # Hard labels (deterministic)
            y = torch.argmax(logits, dim=-1).float()
        else:
            # Sample from distribution
            y = torch.multinomial(probs.reshape(-1, n_classes), 1).reshape(batch_size, seq_len).float()
        
        single_eval_pos = seq_len // 2
        return dict(
            x=x,
            y=y,
            target_y=y,
            single_eval_pos=single_eval_pos
        )


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_prior(prior, idea_name, steps=10000, batch_size=4):
    """Train a model with a given prior."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {idea_name}")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    device = get_default_device()
    print(f"Device: {device}")
    
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return prior.get_batch(batch_size, seq_len, n_features, n_classes=3)
    
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=steps,
        batch_size=batch_size,
        num_datapoints_max=100,
        num_features=20,
        device=device
    )
    
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3,
    )
    
    print(f"Steps: {steps}")
    print(f"Training...")
    
    criterion = nn.CrossEntropyLoss()
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=criterion,
        epochs=1,
        accumulate_gradients=1,
        lr=1e-4,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name=idea_name
    )
    
    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "models", "advanced_priors")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{idea_name}.pt")
    torch.save(trained_model.state_dict(), save_path)
    
    print(f"\n✅ Model saved: {save_path}")
    print(f"Final loss: {loss:.4f}")
    
    return save_path, trained_model


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, device, datasets):
    """Evaluate a model on all target datasets."""
    results = {}
    
    for ds_name, X, y in datasets:
        n_classes = len(np.unique(y))
        
        # Filter to 3 classes
        if n_classes > 3:
            classes = np.unique(y)[:3]
            mask = np.isin(y, classes)
            X = X[mask]
            y = y[mask]
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[yi] for yi in y])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Ensemble inference
        all_logits = []
        ctx_size = min(50, len(Xtr))
        
        for _ in range(20):
            with torch.no_grad():
                idx = np.random.choice(len(Xtr), ctx_size, replace=False)
                x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
                y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
                x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
                
                x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
                all_logits.append(logits.squeeze(0).cpu().numpy())
        
        preds = np.argmax(np.mean(all_logits, axis=0), axis=-1)
        acc = accuracy_score(yte, preds)
        results[ds_name] = acc
    
    return results


def load_target_datasets():
    """Load target datasets: Iris, Vehicle, Vowel + Wine, BC for sanity check."""
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
    ]
    
    for name in ["vehicle", "vowel"]:
        try:
            data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
            df = data.data
            y = data.target
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    try:
                        df[col] = df[col].astype(float)
                    except:
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            X = df.values.astype(float)
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                X = SimpleImputer(strategy='mean').fit_transform(X)
            if y.dtype == object or y.dtype == 'category':
                y = LabelEncoder().fit_transform(y.astype(str))
            else:
                y = y.values.astype(int)
            datasets.append((name.capitalize(), X, y))
        except Exception as e:
            print(f"  Warning: Could not load {name}")
    
    return datasets


# ============================================================================
# MAIN
# ============================================================================

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883,
    "Vehicle": 0.8154, "Vowel": 1.0
}

BASELINE = {
    "Iris": 0.9333, "Wine": 1.0, "Breast Cancer": 0.9883,
    "Vehicle": 0.6718, "Vowel": 0.8395
}


def main():
    parser = argparse.ArgumentParser(description='Test Advanced Prior Ideas')
    parser.add_argument('--idea', type=int, choices=[1, 2, 3], help='Which idea to test (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', help='Test all 3 ideas')
    parser.add_argument('--steps', type=int, default=10000, help='Training steps')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 ADVANCED PRIOR IDEAS: Testing 3 Key Improvements 🔬")
    print("=" * 70)
    
    device = get_default_device()
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading target datasets...")
    datasets = load_target_datasets()
    print(f"Loaded: {[d[0] for d in datasets]}")
    
    # Determine which ideas to test
    ideas_to_test = []
    if args.all:
        ideas_to_test = [1, 2, 3]
    elif args.idea:
        ideas_to_test = [args.idea]
    else:
        ideas_to_test = [1, 2, 3]  # Default: all
    
    all_results = {}
    
    for idea_num in ideas_to_test:
        if idea_num == 1:
            print("\n" + "=" * 70)
            print("IDEA 1: DYNAMIC SPECTRAL PRIOR")
            print("H ~ U(1.0, 4.0) - Forces In-Context Spectral Adaptation")
            print("=" * 70)
            prior = DynamicSpectralPrior(n_classes=3, device=device)
            idea_name = "dynamic_spectral"
            
        elif idea_num == 2:
            print("\n" + "=" * 70)
            print("IDEA 2: HYBRID-GEOMETRY MIXTURE PRIOR")
            print("40% Manifold + 40% DAG + 20% Tree")
            print("=" * 70)
            prior = HybridMixturePrior(n_classes=3, device=device)
            idea_name = "hybrid_mixture"
            
        else:  # idea_num == 3
            print("\n" + "=" * 70)
            print("IDEA 3: THERMODYNAMIC TEMPERATURE PRIOR")
            print("τ ~ U(0.01, 1.0) - Hard to Soft Boundaries")
            print("=" * 70)
            prior = ThermodynamicPrior(n_classes=3, device=device)
            idea_name = "thermodynamic"
        
        # Train
        model_path, model = train_prior(prior, idea_name, steps=args.steps)
        
        # Evaluate
        print(f"\n📊 Evaluating {idea_name}...")
        model.to(device)
        model.eval()
        results = evaluate_model(model, device, datasets)
        all_results[idea_name] = results
        
        # Show results
        print(f"\n  {'Dataset':<15} | {'Result':>8} | {'Baseline':>8} | {'NeuralK':>8} | {'Δ Base':>7} | {'Δ NK':>7}")
        print(f"  {'-'*65}")
        for ds_name, acc in results.items():
            base = BASELINE.get(ds_name, 0)
            nk = NEURALK.get(ds_name, 0)
            delta_base = (acc - base) * 100
            delta_nk = (acc - nk) * 100
            status = "🏆" if delta_nk >= 0 else ("⬆️" if delta_base > 0 else "")
            print(f"  {ds_name:<15} | {acc*100:>7.2f}% | {base*100:>7.2f}% | {nk*100:>7.2f}% | {delta_base:>+6.2f}% | {delta_nk:>+6.2f}% {status}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY: ALL IDEAS 🏆")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'Baseline':>8} | ", end="")
    for idea in all_results.keys():
        print(f"{idea[:8]:>10} | ", end="")
    print(f"{'NeuralK':>8}")
    print("-" * 80)
    
    for ds in ["Iris", "Wine", "Breast Cancer", "Vehicle", "Vowel"]:
        base = BASELINE.get(ds, 0)
        nk = NEURALK.get(ds, 0)
        print(f"{ds:<15} | {base*100:>7.2f}% | ", end="")
        for idea, results in all_results.items():
            acc = results.get(ds, 0)
            print(f"{acc*100:>9.2f}% | ", end="")
        print(f"{nk*100:>7.2f}%")
    
    # Save results
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "results", "advanced_priors_results.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("ADVANCED PRIOR IDEAS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for idea, results in all_results.items():
            f.write(f"\n{idea}:\n")
            for ds, acc in results.items():
                f.write(f"  {ds}: {acc*100:.2f}%\n")
    
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
