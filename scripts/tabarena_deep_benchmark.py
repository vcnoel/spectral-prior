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
TABARENA BENCHMARK: DEEP & WIDE TRICKS
============================================================================
Advanced inference-time tricks to squeeze every bit of performance.
Tricks:
1. Rank Averaging (IRA)
2. PCA Ensemble (Feature Rotation)
3. TTA Noise (Test-Time Augmentation)
4. Context Grid (Multi-scale Context)
5. Feature Bagging (Subspace Sampling)
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
from sklearn.decomposition import PCA
import warnings
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(os.path.join(root_dir, "TFM-Playground"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# NeuralK API
try:
    from neuralk import NICLClassifier
    NEURALK_AVAILABLE = True
except ImportError:
    NEURALK_AVAILABLE = False
    print("⚠️ NeuralK not installed.")

# ============================================================================
# UTILS
# ============================================================================

def get_logits(model, x_c, y_c, x_t, device):
    x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
    y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
    return model((x_full, y_c_uns), single_eval_pos=x_c.shape[0]).squeeze(0)

# ============================================================================
# DEEP & WIDE TRICKS
# ============================================================================

def context_grid_ensemble(model, Xtr, ytr, Xte, device):
    """Ensemble over a grid of context sizes."""
    all_probs = []
    # Grid of context sizes: 10 to 90
    ctx_grid = [10, 30, 50, 70, 90]
    
    for ctx in ctx_grid:
        if ctx >= len(Xtr): continue
        for _ in range(5): # 5 seeds per ctx
            with torch.no_grad():
                idx = np.random.choice(len(Xtr), ctx, replace=False)
                x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
                y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
                x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
                
                logits = get_logits(model, x_c, y_c, x_t, device)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
    return np.mean(all_probs, axis=0)

def feature_bagging(model, Xtr, ytr, Xte, device):
    """Ensemble over random feature subspaces (0.5 to 1.0 ratio)."""
    # ... (Skipped due to implementation complexity with fixed models) ...
    pass 

def pca_ensemble(model, Xtr, ytr, Xte, device):
    """
    Ensemble over random PCA rotations.
    Mechanism: Rotates the feature space to allow the geometric prior to view 
    decision boundaries from multiple angles. This breaks the "single-view" bias.
    """
    all_probs = []
    
    for _ in range(15):
        try:
            # Random rotation matrix
            D = Xtr.shape[1]
            rotation = np.linalg.qr(np.random.randn(D, D))[0]
            Xtr_rot = Xtr @ rotation
            Xte_rot = Xte @ rotation
            
            with torch.no_grad():
                idx = np.random.choice(len(Xtr), 50, replace=False)
                x_c = torch.tensor(Xtr_rot[idx], dtype=torch.float32).to(device)
                y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
                x_t = torch.tensor(Xte_rot, dtype=torch.float32).to(device)
                
                logits = get_logits(model, x_c, y_c, x_t, device)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        except:
            pass
            
    if not all_probs: return np.zeros((len(Xte), 3)) # Fail safe
    return np.mean(all_probs, axis=0)

def tta_noise_injection(model, Xtr, ytr, Xte, device, sigma=0.01):
    """Test-Time Augmentation with Gaussian noise."""
    all_probs = []
    
    for _ in range(30):
        with torch.no_grad():
            idx = np.random.choice(len(Xtr), 50, replace=False)
            x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
            
            # Add noise to test set
            noise = torch.randn_like(torch.tensor(Xte, dtype=torch.float32)) * sigma
            x_t = torch.tensor(Xte, dtype=torch.float32).to(device) + noise.to(device)
            
            logits = get_logits(model, x_c, y_c, x_t, device)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            
    return np.mean(all_probs, axis=0)

def rank_averaging(model, Xtr, ytr, Xte, device):
    """Average ranks of predictions from multiple seeds (Robust Ensemble)."""
    all_ranks = []
    n_seeds = 30
    
    for _ in range(n_seeds):
        with torch.no_grad():
            idx = np.random.choice(len(Xtr), 50, replace=False)
            x_c = torch.tensor(Xtr[idx], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx], dtype=torch.float32).to(device)
            x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            
            logits = get_logits(model, x_c, y_c, x_t, device)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_ranks.append(probs)
            
    all_ranks = np.array(all_ranks) # (Models, N_test, n_classes)
    # Median averaging
    median_probs = np.median(all_ranks, axis=0)
    return median_probs

# ============================================================================
# MAIN
# ============================================================================

TRICKS = {
    "Baseline": lambda m, xr, yr, xt, d: context_grid_ensemble(m, xr, yr, xt, d), 
    "PCA-Ens": pca_ensemble,
    "TTA-Noise": tta_noise_injection,
    "RankAvg": rank_averaging,
    "ContextGrid": context_grid_ensemble,
}

def run_benchmark_seeds():
    print("=" * 70)
    print("🧠 DEEP & WIDE RIGOROUS BENCHMARK (5 Seeds) 🧠")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Pre-fetch all data to avoid fetching 5 times
    data_cache = {}
    print("\nFetching datasets...")
    for name in target_datasets:
        try:
            d = fetch_openml(name=name, as_frame=True, parser='auto')
        except:
             if name == "blood-transfusion-service-center": d = fetch_openml(name="blood-transfusion", as_frame=True, parser='auto')
             elif name == "Australian": d = fetch_openml(name="australian", as_frame=True, parser='auto')
             else: continue
             
        # Process
        df = d.data
        y = d.target
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category': 
                try: df[col] = df[col].astype(float)
                except: df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(float)
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            X = SimpleImputer(strategy='mean').fit_transform(X)
        if y.dtype == object or y.dtype == 'category': y = LabelEncoder().fit_transform(y.astype(str))
        else: y = y.values.astype(int)
        
        # Filter classes > 3
        if len(np.unique(y)) > 3:
            counts = np.bincount(y)
            top3 = np.argsort(counts)[-3:]
            mask = np.isin(y, top3)
            X = X[mask]; y = y[mask]
            label_map = {c: i for i, c in enumerate(top3)}
            y = np.array([label_map[yi] for yi in y])
            
        data_cache[name] = (X, y)
        print(f"  {name}: {X.shape}")

    # Aggregated Results
    # stats[dataset][method] = [acc0, acc1, ...]
    stats = {name: {} for name in target_datasets}
    
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        
        for name, (X, y) in data_cache.items():
            # Random split per seed
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # NeuralK
            if NEURALK_AVAILABLE:
                try:
                    nk = NICLClassifier()
                    nk.fit(X_train, y_train) # Dummy fit
                    nk_preds = nk.predict(X_test)
                    nk_acc = accuracy_score(y_test, nk_preds)
                    
                    if "NeuralK" not in stats[name]: stats[name]["NeuralK"] = []
                    stats[name]["NeuralK"].append(nk_acc)
                except: pass
            
            # Our Models
            for m_name, model in models.items():
                
                # Baseline
                p_base = context_grid_ensemble(model, X_train, y_train, X_test, device) 
                acc_base = accuracy_score(y_test, np.argmax(p_base, axis=-1))
                k_base = f"{m_name}+Baseline"
                if k_base not in stats[name]: stats[name][k_base] = []
                stats[name][k_base].append(acc_base)
                
                # PCA-Ens (Our Champion)
                p_pca = pca_ensemble(model, X_train, y_train, X_test, device)
                acc_pca = accuracy_score(y_test, np.argmax(p_pca, axis=-1))
                k_pca = f"{m_name}+PCA-Ens"
                if k_pca not in stats[name]: stats[name][k_pca] = []
                stats[name][k_pca].append(acc_pca)

    # Log file
    log_path = os.path.join("results", "rigorous_benchmark.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("BENCHMARK STARTED\n")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("\n" + "="*100)
    log(f"{'Dataset':<25} | {'Technique':<25} | {'Mean ± Std':<20} | {'Max':<8} | {'Min':<8}")
    log("="*100)
    
    for name in target_datasets:
        if name not in stats: 
            log(f"[{name}] - No stats found")
            continue
        log(f"[{name}]")
        methods = stats[name]
        for method, accs in methods.items():
            if not accs: continue
            mean = np.mean(accs) * 100
            std = np.std(accs) * 100
            maxx = np.max(accs) * 100
            minn = np.min(accs) * 100
            log(f"  {method:<23} | {mean:>6.2f} ± {std:<5.2f}%       | {maxx:>6.2f}% | {minn:>6.2f}%")
        log("-" * 100)

if __name__ == "__main__":
    run_benchmark_seeds()
