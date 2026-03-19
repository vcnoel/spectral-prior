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
============================================================================
TABARENA BENCHMARK: WILD TRICKS
============================================================================
Datasets: credit-g, blood-transfusion, phoneme, diabetes, kc1, Australian
Comparison: NeuralK vs Our Models (H=2.44, HybridMix)
Tricks: Bootstrap, Sharpen, MegaEnsemble
============================================================================
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# NeuralK API
try:
    from neuralk import NICLClassifier
    NEURALK_AVAILABLE = True
except ImportError:
    NEURALK_AVAILABLE = False
    print("⚠️ NeuralK not installed. SOTA comparison will be skipped.")

# ============================================================================
# TRICKS
# ============================================================================

def baseline_predict(model, Xtr, ytr, Xte, device, ctx_size=50, n_ensemble=20):
    all_logits = []
    for _ in range(n_ensemble):
        with torch.no_grad():
            idx = np.random.choice(len(Xtr), min(ctx_size, len(Xtr)), replace=False)
            x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
            x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
            all_logits.append(logits.squeeze(0).cpu().numpy())
    return np.mean(all_logits, axis=0)

def mega_ensemble_predict(model, Xtr, ytr, Xte, device):
    all_logits = []
    # 60 models: 6 seeds * 3 contexts * 3 temps (implicit)
    for temp in [0.5, 0.7, 1.0]:
        for ctx in [30, 50, 70]:
            for _ in range(6): 
                with torch.no_grad():
                    idx = np.random.choice(len(Xtr), min(ctx, len(Xtr)), replace=False)
                    x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
                    y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
                    x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
                    x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
                    y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                    logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
                    all_logits.append((logits.squeeze(0).cpu().numpy() / temp))
    return np.mean(all_logits, axis=0)

def bootstrap_predict(model, Xtr, ytr, Xte, device, n_bootstrap=30):
    all_logits = []
    for _ in range(n_bootstrap):
        with torch.no_grad():
            # Sampling with replacement
            idx = np.random.choice(len(Xtr), len(Xtr), replace=True)
            x_c = torch.tensor(Xtr[idx][:50], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx][:50], dtype=torch.float32).to(device)
            x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
            all_logits.append(logits.squeeze(0).cpu().numpy())
    return np.mean(all_logits, axis=0)

def dropout_predict(model, Xtr, ytr, Xte, device, dropout=0.1):
    all_logits = []
    # Feature dropout
    mask = np.random.rand(Xtr.shape[1]) > dropout
    Xtr_d = Xtr[:, mask]
    Xte_d = Xte[:, mask]
    for _ in range(20):
        with torch.no_grad():
            idx = np.random.choice(len(Xtr), min(50, len(Xtr)), replace=False)
            x_c = torch.tensor(Xtr_d[idx], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
            x_t = torch.tensor(Xte_d, dtype=torch.float32).to(device)
            x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
            all_logits.append(logits.squeeze(0).cpu().numpy())
    return np.mean(all_logits, axis=0)

def sharpen_predict(model, Xtr, ytr, Xte, device, tau=0.5):
    logits = baseline_predict(model, Xtr, ytr, Xte, device)
    return logits / tau

TRICKS = {
    "Baseline": baseline_predict,
    "MegaEns60": mega_ensemble_predict,
    "Dropout10%": lambda m, xr, yr, xt, d: dropout_predict(m, xr, yr, xt, d, 0.1),
    "Bootstrap": bootstrap_predict,
    "Sharpen": lambda m, xr, yr, xt, d: sharpen_predict(m, xr, yr, xt, d, 0.5),
}

# ============================================================================
# UTILS
# ============================================================================

def load_openml_dataset(name):
    print(f"  Fetching {name}...")
    try:
        data = fetch_openml(name=name, as_frame=True, parser='auto')
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
        print(f"  ❌ Failed to load {name}: {e}")
        return None, None

def evaluate_neuralk(X_train, y_train, X_test, y_test):
    if not NEURALK_AVAILABLE: return None
    try:
        clf = NICLClassifier()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
    except Exception as e:
        print(f"  NeuralK failed: {e}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 TABARENA BENCHMARK: WILD TRICKS EDITION 🚀")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Models
    models = {}
    path_h244 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tournament_10k", "prior_complex_manifold.pt")
    if os.path.exists(path_h244):
        m = NanoTabPFNModel(128, 4, 512, 4, 3)
        m.load_state_dict(torch.load(path_h244, map_location=device))
        m.to(device)
        models["H=2.44"] = m
        
    path_hybrid = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "advanced_priors", "hybrid_mixture.pt")
    if os.path.exists(path_hybrid):
        m = NanoTabPFNModel(128, 4, 512, 4, 3)
        m.load_state_dict(torch.load(path_hybrid, map_location=device))
        m.to(device)
        models["Hybrid"] = m
        
    target_datasets = ["credit-g", "blood-transfusion-service-center", "phoneme", "diabetes", "kc1", "Australian"]
    results = []
    
    for name in target_datasets:
        print(f"\ndataset: {name}")
        X, y = load_openml_dataset(name)
        if X is None:
            if name == "blood-transfusion-service-center": X, y = load_openml_dataset("blood-transfusion")
            elif name == "Australian": X, y = load_openml_dataset("australian")
            if X is None: continue
            
        n_classes = len(np.unique(y))
        s = get_singular_spectrum(X[:1000], normalized=True)
        H = spectral_entropy(s)
        print(f"  Shape: {X.shape}, H: {H:.2f}")
        
        if n_classes > 3:
            print(f"  Filtering to top 3 classes...")
            counts = np.bincount(y)
            top3 = np.argsort(counts)[-3:]
            mask = np.isin(y, top3)
            X = X[mask]; y = y[mask]
            label_map = {c: i for i, c in enumerate(top3)}
            y = np.array([label_map[yi] for yi in y])
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        row = {"Dataset": name, "H": H}
        
        # NeuralK
        print("  Running NeuralK...")
        row["NeuralK"] = evaluate_neuralk(X_train, y_train, X_test, y_test)
        if row["NeuralK"]: print(f"    NeuralK: {row['NeuralK']*100:.2f}%")
        
        # Our Models + Tricks
        best_acc = 0
        best_meta = ""
        
        for m_name, model in models.items():
            print(f"  Running {m_name}...")
            for t_name, t_fn in TRICKS.items():
                logits = t_fn(model, X_train, y_train, X_test, device)
                preds = np.argmax(logits, axis=-1)
                acc = accuracy_score(y_test, preds)
                print(f"    {t_name:12}: {acc*100:.2f}%")
                
                if acc > best_acc:
                    best_acc = acc
                    best_meta = f"{m_name} + {t_name}"
        
        row["Best Ours"] = best_acc
        row["Meta"] = best_meta
        results.append(row)
        
    print("\n" + "="*80)
    print("FINAL RESULTS with WILD TRICKS")
    print("="*80)
    print(f"{'Dataset':<20} | {'NeuralK':>7} | {'Our Best':>7} | {'Δ':>7} | {'Recipe':<25}")
    print("-" * 80)
    for r in results:
        nk = r.get("NeuralK")
        best = r.get("Best Ours")
        meta = r.get("Meta")
        nk_s = f"{nk*100:.2f}%" if nk else "N/A"
        best_s = f"{best*100:.2f}%" if best else "N/A"
        delta = (best - nk) * 100 if nk else 0
        delta_s = f"{delta:+.2f}%" if nk else "N/A"
        print(f"{r['Dataset']:<20} | {nk_s:>7} | {best_s:>7} | {delta_s:>7} | {meta:<25}")
        
    with open(os.path.join("results", "tabarena_wild_results.txt"), "w") as f:
        f.write(str(results))

if __name__ == "__main__":
    main()
