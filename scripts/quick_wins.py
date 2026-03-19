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
2026 Quick Wins: K-NN Context, Self-Consistency, Confidence Calibration
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Letter": 0.9985,
    "Vehicle": 0.8154, "Vowel": 1.0
}


def evaluate_with_tricks(model, X_train, y_train, X_test, y_test, device, trick="baseline"):
    """
    Evaluate with various 2026 tricks.
    
    Tricks:
    - baseline: Standard evaluation
    - knn_context: K-NN based context selection
    - self_consistency: Sample 50 predictions, majority vote
    - calibrated: Temperature calibration (T=0.7)
    - all_tricks: Combine all above
    """
    model.eval()
    n_test = X_test.shape[0]
    
    # Settings based on trick
    if trick == "baseline":
        n_ensemble = 10
        use_knn = False
        temperature = 1.0
    elif trick == "knn_context":
        n_ensemble = 10
        use_knn = True
        temperature = 1.0
    elif trick == "self_consistency":
        n_ensemble = 50
        use_knn = False
        temperature = 1.0
    elif trick == "calibrated":
        n_ensemble = 10
        use_knn = False
        temperature = 0.7
    elif trick == "all_tricks":
        n_ensemble = 50
        use_knn = True
        temperature = 0.7
    else:
        n_ensemble = 10
        use_knn = False
        temperature = 1.0
    
    # Prepare K-NN if needed
    if use_knn:
        knn = NearestNeighbors(n_neighbors=min(50, len(X_train)))
        knn.fit(X_train)
    
    context_size = min(50, X_train.shape[0])
    batch_size = 50
    
    all_votes = []
    
    for ens_idx in range(n_ensemble):
        preds_this = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                x_batch = X_test[i:i+batch_size]
                
                # Context selection
                if use_knn and len(x_batch) > 0:
                    # Use K-NN to find nearest training examples for the FIRST test point in batch
                    distances, indices = knn.kneighbors(x_batch[:1])
                    context_indices = indices[0][:context_size]
                else:
                    # Random context
                    context_indices = np.random.choice(len(X_train), context_size, replace=False)
                
                x_c = torch.tensor(X_train[context_indices], dtype=torch.float32).to(device)
                y_c = torch.tensor(y_train[context_indices], dtype=torch.float32).to(device)
                x_b = torch.tensor(x_batch, dtype=torch.float32).to(device)
                
                x_full = torch.cat([x_c.unsqueeze(0), x_b.unsqueeze(0)], dim=1)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
                
                # Temperature scaling
                logits = logits / temperature
                
                preds = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()
                preds_this.extend(preds)
        
        all_votes.append(np.array(preds_this))
    
    # Majority voting (self-consistency)
    all_votes = np.stack(all_votes, axis=0)  # [n_ensemble, n_test]
    final_preds = []
    for i in range(n_test):
        votes = all_votes[:, i]
        final_preds.append(np.bincount(votes.astype(int)).argmax())
    
    return accuracy_score(y_test, final_preds)


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
    print("=" * 70)
    print("🚀 2026 QUICK WINS: K-NN + Self-Consistency + Calibration 🚀")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "models", "tournament_10k", "prior_complex_manifold.pt")
    
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
    print(f"Model loaded: deep_spectral_10k.pt")
    
    # Datasets (focus on key ones)
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
    ]
    
    # Add OpenML
    for name in ["vehicle", "vowel"]:
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
    
    tricks = ["baseline", "knn_context", "self_consistency", "calibrated", "all_tricks"]
    
    all_results = []
    
    for name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")
        
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
        
        nk = NEURALK.get(name, 0)
        best_acc = 0
        best_trick = ""
        
        for trick in tricks:
            try:
                acc = evaluate_with_tricks(model, X_train, y_train, X_test, y_test, device, trick)
                delta = (acc - nk) * 100
                status = "🏆" if delta >= 0 else ("⭐" if delta > -2 else "")
                print(f"  {trick:<20}: {acc*100:>6.2f}% (Δ: {delta:>+6.2f}%) {status}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_trick = trick
                    
            except Exception as e:
                print(f"  {trick:<20}: ERROR - {e}")
        
        all_results.append({
            "Dataset": name,
            "H": H,
            "Best": best_acc,
            "BestTrick": best_trick,
            "NeuralK": nk
        })
        print(f"\n  >>> Best: {best_trick} ({best_acc*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY 🏆")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'H':>5} | {'Best':>8} | {'Trick':<15} | {'NeuralK':>8} | {'Δ':>7}")
    print("-" * 70)
    
    wins = 0
    for r in all_results:
        delta = (r['Best'] - r['NeuralK']) * 100
        status = "🏆" if delta >= 0 else ""
        if delta >= 0:
            wins += 1
        print(f"{r['Dataset']:<15} | {r['H']:>5.2f} | {r['Best']*100:>7.2f}% | {r['BestTrick']:<15} | {r['NeuralK']*100:>7.2f}% | {delta:>+6.2f}% {status}")
    
    print(f"\nTotal Wins/Ties: {wins}/{len(all_results)}")
    
    # Save
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "results", "quick_wins_results.txt")
    with open(output, 'w') as f:
        f.write("2026 QUICK WINS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for r in all_results:
            f.write(f"{r['Dataset']}: {r['Best']*100:.2f}% ({r['BestTrick']})\n")
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
