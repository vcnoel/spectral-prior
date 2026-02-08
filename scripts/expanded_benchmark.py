"""
Expanded Benchmark Script with Spectral Entropy Analysis and NeuralK Comparison.

This script:
1. Loads multiple datasets (sklearn + OpenML)
2. Calculates spectral entropy (H) for each dataset
3. Evaluates our DeepSpectralPrior model
4. Compares against NeuralK API (SOTA)
5. Outputs results with H to analyze performance vs entropy
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_iris,
    fetch_openml
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# NeuralK API
try:
    from neuralk import NICLClassifier
    NEURALK_AVAILABLE = True
except ImportError:
    print("Warning: neuralk not installed. NeuralK comparison will be skipped.")
    NEURALK_AVAILABLE = False


def calculate_dataset_entropy(X):
    """Calculate the spectral entropy H of a dataset."""
    spectrum = get_singular_spectrum(X, normalized=True)
    H = spectral_entropy(spectrum)
    return H


def evaluate_nanotabpfn(model, X_train, y_train, X_test, y_test, device):
    """Evaluate NanoTabPFN model."""
    model.eval()
    batch_size = 100
    n_test = X_test.shape[0]
    preds = []
    context_size = min(50, X_train.shape[0])
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
            
            indices = np.random.choice(X_train.shape[0], context_size, replace=False)
            x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
            y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
            
            x_c_uns = x_c.unsqueeze(0)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            x_b_uns = x_b.unsqueeze(0)
            
            x_full = torch.cat([x_c_uns, x_b_uns], dim=1)
            
            logits = model((x_full, y_c_uns), single_eval_pos=x_c_uns.shape[1])
            batch_preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            if batch_preds.ndim == 0:
                preds.append(batch_preds.item())
            else:
                preds.extend(batch_preds)
                
    return accuracy_score(y_test, preds)


def evaluate_neuralk(X_train, y_train, X_test, y_test):
    """Evaluate NeuralK API."""
    if not NEURALK_AVAILABLE:
        return None
    try:
        clf = NICLClassifier()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        return accuracy_score(y_test, predictions)
    except Exception as e:
        print(f"  NeuralK Error: {e}")
        return None


def load_openml_dataset(name, version=1):
    """Load a dataset from OpenML."""
    try:
        data = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        df = data.data
        y = data.target
        
        # Convert categorical columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    # Encode as label
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        X = df.values.astype(float)
        
        # Handle NaN
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # Encode labels if string
        if y.dtype == object or y.dtype == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.values.astype(int)
            
        return X, y
    except Exception as e:
        print(f"  Failed to load {name}: {e}")
        return None, None


def run_expanded_benchmark():
    print("=" * 60)
    print("EXPANDED BENCHMARK: Spectral Entropy Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"NeuralK Available: {NEURALK_AVAILABLE}")
    
    # Load our model (DeepSpectralPrior)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tournament_10k", "prior_complex_manifold.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    model = NanoTabPFNModel(128, 4, 512, 4, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded: DeepSpectralPrior (prior_complex_manifold.pt)")
    
    # Define datasets
    datasets = []
    
    # sklearn datasets
    print("\nLoading sklearn datasets...")
    datasets.append(("Iris", *load_iris(return_X_y=True)))
    datasets.append(("Wine", *load_wine(return_X_y=True)))
    datasets.append(("Breast Cancer", *load_breast_cancer(return_X_y=True)))
    datasets.append(("Digits", *load_digits(return_X_y=True)))
    
    # OpenML datasets (harder)
    print("Loading OpenML datasets...")
    openml_datasets = [
        ("vehicle", 1),       # Vehicle silhouettes (4 classes)
        ("segment", 1),       # Image segmentation (7 classes)
        ("vowel", 1),         # Vowel recognition (11 classes)
        ("satimage", 1),      # Satellite image (6 classes)
        ("letter", 1),        # Letter recognition (26 classes)
    ]
    
    for name, version in openml_datasets:
        print(f"  Loading {name}...")
        X, y = load_openml_dataset(name, version)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)
    
    results = []
    target_H = 2.44  # Our tuned spectral entropy
    
    for name, X, y in datasets:
        print(f"\n--- {name} ---")
        print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Calculate spectral entropy
        H = calculate_dataset_entropy(X)
        print(f"  Spectral Entropy H: {H:.3f} (Target: {target_H:.2f}, Δ: {abs(H - target_H):.3f})")
        
        # Filter classes if > 3 (model limitation)
        n_classes = len(np.unique(y))
        if n_classes > 3:
            print(f"  Filtering to first 3 classes (model outputs 3)...")
            mask = y < 3
            X_filtered = X[mask]
            y_filtered = y[mask]
        else:
            X_filtered = X
            y_filtered = y
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_filtered, test_size=0.3, random_state=42
        )
        
        # Evaluate DeepSpectralPrior
        try:
            our_acc = evaluate_nanotabpfn(model, X_train, y_train, X_test, y_test, device)
            print(f"  DeepSpectralPrior: {our_acc*100:.2f}%")
        except Exception as e:
            print(f"  DeepSpectralPrior Error: {e}")
            our_acc = None
        
        # Evaluate NeuralK
        nk_acc = evaluate_neuralk(X_train, y_train, X_test, y_test)
        if nk_acc is not None:
            print(f"  NeuralK (SOTA):    {nk_acc*100:.2f}%")
        else:
            print(f"  NeuralK: N/A")
        
        # Delta
        if our_acc is not None and nk_acc is not None:
            delta = (our_acc - nk_acc) * 100
            print(f"  Δ (Ours - NeuralK): {delta:+.2f}%")
        
        results.append({
            "Dataset": name,
            "N": X.shape[0],
            "D": X.shape[1],
            "Classes": n_classes,
            "H": H,
            "DeepSpectralPrior": our_acc,
            "NeuralK": nk_acc
        })
    
    # Summary Table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Dataset':<15} | {'N':>6} | {'D':>3} | {'H':>5} | {'Ours':>7} | {'NeuralK':>7} | {'Δ':>7}")
    print("-" * 70)
    
    for r in results:
        ours_str = f"{r['DeepSpectralPrior']*100:.2f}%" if r['DeepSpectralPrior'] else "N/A"
        nk_str = f"{r['NeuralK']*100:.2f}%" if r['NeuralK'] else "N/A"
        if r['DeepSpectralPrior'] and r['NeuralK']:
            delta = (r['DeepSpectralPrior'] - r['NeuralK']) * 100
            delta_str = f"{delta:+.2f}%"
        else:
            delta_str = "N/A"
        print(f"{r['Dataset']:<15} | {r['N']:>6} | {r['D']:>3} | {r['H']:>5.2f} | {ours_str:>7} | {nk_str:>7} | {delta_str:>7}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ENTROPY ANALYSIS")
    print("=" * 60)
    print(f"Target Entropy (H*): {target_H}")
    print("\nPerformance by Entropy Distance:")
    
    for r in sorted(results, key=lambda x: abs(x['H'] - target_H)):
        dist = abs(r['H'] - target_H)
        ours = r['DeepSpectralPrior']*100 if r['DeepSpectralPrior'] else 0
        print(f"  {r['Dataset']:<15}: H={r['H']:.2f}, Δ_H={dist:.2f}, Acc={ours:.1f}%")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "expanded_benchmark_results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("EXPANDED BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Dataset':<15} | {'N':>6} | {'D':>3} | {'H':>5} | {'Ours':>7} | {'NeuralK':>7} | {'Δ':>7}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            ours_str = f"{r['DeepSpectralPrior']*100:.2f}%" if r['DeepSpectralPrior'] else "N/A"
            nk_str = f"{r['NeuralK']*100:.2f}%" if r['NeuralK'] else "N/A"
            if r['DeepSpectralPrior'] and r['NeuralK']:
                delta = (r['DeepSpectralPrior'] - r['NeuralK']) * 100
                delta_str = f"{delta:+.2f}%"
            else:
                delta_str = "N/A"
            f.write(f"{r['Dataset']:<15} | {r['N']:>6} | {r['D']:>3} | {r['H']:>5.2f} | {ours_str:>7} | {nk_str:>7} | {delta_str:>7}\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_expanded_benchmark()
