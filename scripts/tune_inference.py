# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tuning Script: Experiment with inference tricks to improve performance.

Tricks to try:
1. Larger context size (more training examples as context)
2. Ensemble predictions (multiple random contexts, average)
3. Different standardization (RobustScaler vs StandardScaler)
4. Feature selection (PCA to reduce dimensionality)
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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# Fixed NeuralK results (from previous run)
NEURALK_RESULTS = {
    "Iris": 1.0,
    "Wine": 1.0,
    "Breast Cancer": 0.9883,
    "Digits": 1.0,
    "Segment": 0.9933,
    "Satimage": 0.9926,
    "Letter": 0.9985
}


def evaluate_with_tricks(model, X_train, y_train, X_test, y_test, device, 
                         context_size=50, n_ensemble=1):
    """Evaluate with configurable tricks."""
    model.eval()
    batch_size = 100
    n_test = X_test.shape[0]
    
    # Ensemble: run multiple times with different contexts
    all_logits = []
    
    for _ in range(n_ensemble):
        preds_logits = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
                
                # Sample context
                ctx_size = min(context_size, X_train.shape[0])
                indices = np.random.choice(X_train.shape[0], ctx_size, replace=False)
                x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
                y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
                
                x_c_uns = x_c.unsqueeze(0)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                x_b_uns = x_b.unsqueeze(0)
                
                x_full = torch.cat([x_c_uns, x_b_uns], dim=1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c_uns.shape[1])
                preds_logits.append(logits.squeeze(0).cpu())
        
        all_logits.append(torch.cat(preds_logits, dim=0))
    
    # Average logits across ensemble
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = torch.argmax(avg_logits, dim=-1).numpy()
    
    return accuracy_score(y_test, preds)


def load_openml_safe(name, version=1):
    """Load OpenML dataset with proper handling."""
    try:
        data = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        df = data.data
        y = data.target
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        X = df.values.astype(float)
        
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            X = SimpleImputer(strategy='mean').fit_transform(X)
        
        if y.dtype == object or y.dtype == 'category':
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values.astype(int)
            
        return X, y
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def run_tuning():
    print("=" * 70)
    print("TUNING EXPERIMENTS: Finding optimal inference settings")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "models", "tournament_10k", "prior_complex_manifold.pt")
    model = NanoTabPFNModel(128, 4, 512, 4, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded: DeepSpectralPrior")
    
    # Datasets (excluding Vowel, Vehicle)
    datasets = []
    datasets.append(("Iris", *load_iris(return_X_y=True)))
    datasets.append(("Wine", *load_wine(return_X_y=True)))
    datasets.append(("Breast Cancer", *load_breast_cancer(return_X_y=True)))
    datasets.append(("Digits", *load_digits(return_X_y=True)))
    
    print("Loading OpenML datasets...")
    for name in ["segment", "satimage", "letter", "vehicle", "vowel"]:
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
    
    # Tricks to try
    tricks = [
        {"name": "Baseline (ctx=50, ens=1)", "context_size": 50, "n_ensemble": 1, "scaler": "standard", "pca": None},
        {"name": "Large Context (ctx=100)", "context_size": 100, "n_ensemble": 1, "scaler": "standard", "pca": None},
        {"name": "Ensemble x5", "context_size": 50, "n_ensemble": 5, "scaler": "standard", "pca": None},
        {"name": "Ensemble x10", "context_size": 50, "n_ensemble": 10, "scaler": "standard", "pca": None},
        {"name": "Large + Ensemble x5", "context_size": 100, "n_ensemble": 5, "scaler": "standard", "pca": None},
        {"name": "RobustScaler", "context_size": 50, "n_ensemble": 1, "scaler": "robust", "pca": None},
        {"name": "PCA (d=10)", "context_size": 50, "n_ensemble": 1, "scaler": "standard", "pca": 10},
    ]
    
    results = []
    
    for name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        n_classes = len(np.unique(y))
        print(f"Shape: {X.shape}, Classes: {n_classes}, H: {H:.2f}")
        
        # Filter to 3 classes if needed
        if n_classes > 3:
            mask = y < 3
            X = X[mask]
            y = y[mask]
            print(f"  (Filtered to 3 classes: {X.shape[0]} samples)")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        nk_acc = NEURALK_RESULTS.get(name, None)
        
        dataset_results = {"Dataset": name, "H": H, "NeuralK": nk_acc}
        
        best_acc = 0
        best_trick = None
        
        for trick in tricks:
            # Apply scaler
            if trick["scaler"] == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply PCA if specified
            if trick["pca"] is not None and X_train_scaled.shape[1] > trick["pca"]:
                pca = PCA(n_components=trick["pca"])
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)
            
            # Evaluate
            acc = evaluate_with_tricks(
                model, X_train_scaled, y_train, X_test_scaled, y_test, device,
                context_size=trick["context_size"],
                n_ensemble=trick["n_ensemble"]
            )
            
            dataset_results[trick["name"]] = acc
            
            if acc > best_acc:
                best_acc = acc
                best_trick = trick["name"]
            
            delta = (acc - nk_acc) * 100 if nk_acc else 0
            print(f"  {trick['name']:<25}: {acc*100:>6.2f}% (Δ: {delta:+.2f}%)")
        
        dataset_results["Best"] = best_trick
        dataset_results["Best_Acc"] = best_acc
        results.append(dataset_results)
        
        print(f"  >>> Best: {best_trick} ({best_acc*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Settings per Dataset")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'H':>5} | {'Baseline':>8} | {'Best':>8} | {'NeuralK':>8} | {'Δ':>7} | Trick")
    print("-" * 80)
    
    for r in results:
        baseline = r.get("Baseline (ctx=50, ens=1)", 0)
        best = r.get("Best_Acc", 0)
        nk = r.get("NeuralK", 0) or 0
        delta = (best - nk) * 100
        trick = r.get("Best", "N/A")
        print(f"{r['Dataset']:<15} | {r['H']:>5.2f} | {baseline*100:>7.2f}% | {best*100:>7.2f}% | {nk*100:>7.2f}% | {delta:>+6.2f}% | {trick}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "results", "tuning_results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TUNING EXPERIMENT RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"{r['Dataset']}: Best={r.get('Best', 'N/A')} ({r.get('Best_Acc', 0)*100:.2f}%)\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_tuning()
