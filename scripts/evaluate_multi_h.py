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
EVALUATE MULTI-H PRIORS
Compare H=1.1, H=2.4 (original), and H=3.9 models on all datasets.

Hypothesis:
- H=1.1 should work best on Iris (H=1.09)
- H=2.4 should work best on Wine, BC (H≈2.4-2.8)
- H=3.9 should work best on Digits (H=3.92)
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Letter": 0.9985,
    "Vehicle": 0.8154, "Vowel": 1.0
}


def load_model(model_path, device):
    """Load a NanoTabPFN model."""
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, X_train, y_train, X_test, y_test, device, n_ensemble=20):
    """Ensemble evaluation."""
    all_logits = []
    context_size = min(50, len(X_train))
    
    for _ in range(n_ensemble):
        with torch.no_grad():
            indices = np.random.choice(len(X_train), context_size, replace=False)
            x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
            y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
            x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            
            x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            
            logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
            all_logits.append(logits.squeeze(0).cpu().numpy())
    
    avg_logits = np.mean(all_logits, axis=0)
    preds = np.argmax(avg_logits, axis=-1)
    return accuracy_score(y_test, preds)


def load_openml_safe(name, version=1):
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


def main():
    print("=" * 80)
    print("🎯 MULTI-H PRIOR EVALUATION: H=1.1 vs H=2.4 vs H=3.9 🎯")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load all three models
    models = {
        "H=1.1": os.path.join(base_dir, "models", "multi_h", "spectral_H1.1.pt"),
        "H=2.4": os.path.join(base_dir, "models", "tournament_10k", "prior_complex_manifold.pt"),
        "H=3.9": os.path.join(base_dir, "models", "multi_h", "spectral_H3.9.pt"),
    }
    
    loaded_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            loaded_models[name] = load_model(path, device)
            print(f"Loaded: {name}")
        else:
            print(f"Missing: {name} ({path})")
    
    # All datasets
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
        ("Digits", *load_digits(return_X_y=True)),
    ]
    
    print("\nLoading OpenML datasets...")
    for name in ["segment", "satimage", "vehicle", "vowel"]:
        print(f"  Loading {name}...", end=" ")
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
            print("OK")
    
    results = []
    
    for ds_name, X, y in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*80}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        n_classes = len(np.unique(y))
        print(f"Shape: {X.shape}, Classes: {n_classes}, H: {H:.2f}")
        
        # Filter to 3 classes
        if n_classes > 3:
            classes = np.unique(y)[:3]
            mask = np.isin(y, classes)
            X = X[mask]
            y = y[mask]
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[yi] for yi in y])
            print(f"  (Filtered to 3 classes)")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        nk = NEURALK.get(ds_name, 0)
        
        # Evaluate each model
        scores = {}
        print(f"\n  {'Model':<10} | {'Accuracy':>8} | {'Δ NeuralK':>10}")
        print(f"  {'-'*40}")
        
        for model_name, model in loaded_models.items():
            acc = evaluate(model, X_train, y_train, X_test, y_test, device)
            scores[model_name] = acc
            delta = (acc - nk) * 100
            status = "🏆" if delta >= 0 else ("⭐" if delta > -5 else "")
            print(f"  {model_name:<10} | {acc*100:>7.2f}% | {delta:>+9.2f}% {status}")
        
        # Find best model for this dataset
        best_model = max(scores, key=scores.get)
        best_acc = scores[best_model]
        
        results.append({
            "Dataset": ds_name,
            "H": H,
            "Best": best_acc,
            "BestModel": best_model,
            "NeuralK": nk,
            **scores
        })
        
        # Was the best model the one matching H?
        expected_best = "H=1.1" if H < 1.5 else ("H=3.9" if H > 3.5 else "H=2.4")
        matched = "✓" if best_model == expected_best else "✗"
        print(f"\n  Best: {best_model} ({best_acc*100:.2f}%), Expected: {expected_best} {matched}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 FINAL SUMMARY: WHICH H WORKS BEST? 🏆")
    print("=" * 80)
    print(f"{'Dataset':<15} | {'H':>5} | {'H=1.1':>8} | {'H=2.4':>8} | {'H=3.9':>8} | {'Best':>8} | {'NeuralK':>8}")
    print("-" * 80)
    
    wins_by_model = {"H=1.1": 0, "H=2.4": 0, "H=3.9": 0}
    
    for r in results:
        best = r["BestModel"]
        wins_by_model[best] += 1
        
        h11 = r.get("H=1.1", 0) * 100
        h24 = r.get("H=2.4", 0) * 100
        h39 = r.get("H=3.9", 0) * 100
        best_score = r["Best"] * 100
        nk = r["NeuralK"] * 100
        
        # Highlight winner
        h11_str = f"**{h11:.1f}%**" if best == "H=1.1" else f"{h11:.1f}%"
        h24_str = f"**{h24:.1f}%**" if best == "H=2.4" else f"{h24:.1f}%"
        h39_str = f"**{h39:.1f}%**" if best == "H=3.9" else f"{h39:.1f}%"
        
        print(f"{r['Dataset']:<15} | {r['H']:>5.2f} | {h11:>7.1f}% | {h24:>7.1f}% | {h39:>7.1f}% | {best:>8} | {nk:>7.1f}%")
    
    print("\n" + "=" * 80)
    print("📊 MODEL WIN COUNT 📊")
    print("=" * 80)
    for model, wins in wins_by_model.items():
        print(f"  {model}: {wins} dataset(s)")
    
    # Analysis
    print("\n" + "=" * 80)
    print("🔬 HYPOTHESIS TEST: Does matching H to dataset help? 🔬")
    print("=" * 80)
    for r in results:
        expected = "H=1.1" if r["H"] < 1.5 else ("H=3.9" if r["H"] > 3.5 else "H=2.4")
        actual = r["BestModel"]
        status = "✅ YES" if actual == expected else "❌ NO"
        print(f"  {r['Dataset']:<15}: H={r['H']:.2f} → Expected {expected}, Got {actual} {status}")
    
    # Save results
    output = os.path.join(base_dir, "results", "multi_h_results.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("MULTI-H PRIOR EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model Win Count:\n")
        for model, wins in wins_by_model.items():
            f.write(f"  {model}: {wins} datasets\n")
        f.write("\nDetailed Results:\n")
        for r in results:
            f.write(f"\n{r['Dataset']} (H={r['H']:.2f}):\n")
            f.write(f"  H=1.1: {r.get('H=1.1', 0)*100:.2f}%\n")
            f.write(f"  H=2.4: {r.get('H=2.4', 0)*100:.2f}%\n")
            f.write(f"  H=3.9: {r.get('H=3.9', 0)*100:.2f}%\n")
            f.write(f"  Best: {r['BestModel']} ({r['Best']*100:.2f}%)\n")
    
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
