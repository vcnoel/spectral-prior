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
Benchmark the 10-class model on all datasets WITHOUT class filtering.
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

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# NeuralK baseline (from previous API calls)
NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Letter": 0.9985,
    "Vehicle": 0.8154, "Vowel": 1.0
}


def evaluate_10class(model, X_train, y_train, X_test, y_test, device, n_ensemble=10):
    """Evaluate with ensembling."""
    model.eval()
    batch_size = 100
    n_test = X_test.shape[0]
    context_size = min(100, X_train.shape[0])
    
    all_logits = []
    
    for _ in range(n_ensemble):
        preds_logits = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
                
                indices = np.random.choice(len(X_train), context_size, replace=False)
                x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
                y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
                
                x_c_uns = x_c.unsqueeze(0)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                x_b_uns = x_b.unsqueeze(0)
                
                x_full = torch.cat([x_c_uns, x_b_uns], dim=1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c_uns.shape[1])
                preds_logits.append(logits.squeeze(0).cpu())
        
        all_logits.append(torch.cat(preds_logits, dim=0))
    
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = torch.argmax(avg_logits, dim=-1).numpy()
    
    return accuracy_score(y_test, preds)


def load_openml_safe(name, version=1):
    """Load OpenML dataset."""
    try:
        data = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        df = data.data
        y = data.target
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
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


def run_benchmark():
    print("=" * 70)
    print("🔥 10-CLASS MODEL BENCHMARK (NO CLASS FILTERING) 🔥")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load 10-class model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "models", "10class", "spectral_10class_adaptive.pt")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=10  # 10 classes!
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded: spectral_10class_adaptive.pt")
    
    # Define datasets (focus on multi-class)
    datasets = []
    datasets.append(("Iris", *load_iris(return_X_y=True)))
    datasets.append(("Wine", *load_wine(return_X_y=True)))
    datasets.append(("Breast Cancer", *load_breast_cancer(return_X_y=True)))
    datasets.append(("Digits", *load_digits(return_X_y=True)))
    
    print("\nLoading OpenML datasets...")
    for name in ["segment", "satimage", "letter", "vehicle", "vowel"]:
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
    
    results = []
    
    for name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        n_classes = len(np.unique(y))
        print(f"Shape: {X.shape}, Classes: {n_classes}, H: {H:.2f}")
        
        # NO CLASS FILTERING for datasets with <=10 classes
        if n_classes > 10:
            print(f"  Filtering to first 10 classes (model max)...")
            mask = y < 10
            X = X[mask]
            y = y[mask]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Evaluate
        try:
            acc = evaluate_10class(model, X_train, y_train, X_test, y_test, device, n_ensemble=10)
            print(f"  Accuracy: {acc*100:.2f}%")
        except Exception as e:
            print(f"  Error: {e}")
            acc = None
        
        nk = NEURALK.get(name, 0)
        if acc and nk:
            delta = (acc - nk) * 100
            status = "🏆 WIN!" if acc > nk else ("= TIE" if abs(acc - nk) < 0.005 else "")
            print(f"  NeuralK: {nk*100:.2f}%, Δ: {delta:+.2f}% {status}")
        
        results.append({
            "Dataset": name,
            "Classes": n_classes,
            "H": H,
            "Ours": acc,
            "NeuralK": nk
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY 🏆")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'Classes':>7} | {'H':>5} | {'Ours':>8} | {'NeuralK':>8} | {'Δ':>7}")
    print("-" * 70)
    
    wins = 0
    for r in results:
        ours = r['Ours'] * 100 if r['Ours'] else 0
        nk = r['NeuralK'] * 100 if r['NeuralK'] else 0
        delta = ours - nk
        status = "🏆" if delta > 0 else ("=" if abs(delta) < 0.5 else "")
        if delta > 0:
            wins += 1
        print(f"{r['Dataset']:<15} | {r['Classes']:>7} | {r['H']:>5.2f} | {ours:>7.2f}% | {nk:>7.2f}% | {delta:>+6.2f}% {status}")
    
    print(f"\nTotal Wins: {wins}/{len(results)}")
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "results", "10class_benchmark_results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("10-CLASS MODEL BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            ours = r['Ours'] * 100 if r['Ours'] else 0
            nk = r['NeuralK'] * 100 if r['NeuralK'] else 0
            f.write(f"{r['Dataset']}: {ours:.2f}% vs NeuralK {nk:.2f}%\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
