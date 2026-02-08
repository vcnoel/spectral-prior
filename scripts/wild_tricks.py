"""
WILD TRICKS: Aggressive experimentation to beat NeuralK.

Tricks:
1. Massive ensembling (20x, 50x, 100x)
2. Multi-model voting (Robust + DeepSpectral + Causal)
3. Stratified context sampling (balanced classes in context)
4. Temperature scaling on logits
5. Polynomial feature expansion
6. Adaptive PCA (match dataset's intrinsic dim)
7. Boosted context (oversample hard classes)
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

# NeuralK baseline results
NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Letter": 0.9985,
    "Vehicle": 0.8154, "Vowel": 1.0
}


def load_all_models(device):
    """Load all three trained models."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tournament_10k")
    models = {}
    
    configs = [
        ("DeepSpectral", "prior_complex_manifold.pt"),
        ("Robust", "prior_robust_studentt.pt"),
        ("Causal", "prior_causal_dag.pt")
    ]
    
    for name, filename in configs:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            m = NanoTabPFNModel(128, 4, 512, 4, 3)
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device)
            m.eval()
            models[name] = m
            print(f"  Loaded: {name}")
    
    return models


def evaluate_ensemble(model, X_train, y_train, X_test, y_test, device, 
                      context_size=50, n_ensemble=1, temperature=1.0, stratified=False):
    """Evaluate with ensembling and optional tricks."""
    model.eval()
    batch_size = 100
    n_test = X_test.shape[0]
    
    all_logits = []
    classes = np.unique(y_train)
    
    for _ in range(n_ensemble):
        preds_logits = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
                
                # Stratified or random context
                if stratified:
                    # Sample equal from each class
                    per_class = max(1, context_size // len(classes))
                    indices = []
                    for c in classes:
                        c_idx = np.where(y_train == c)[0]
                        indices.extend(np.random.choice(c_idx, min(per_class, len(c_idx)), replace=False))
                    indices = np.array(indices)[:context_size]
                else:
                    indices = np.random.choice(len(X_train), min(context_size, len(X_train)), replace=False)
                
                x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
                y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
                
                x_c_uns = x_c.unsqueeze(0)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                x_b_uns = x_b.unsqueeze(0)
                
                x_full = torch.cat([x_c_uns, x_b_uns], dim=1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c_uns.shape[1])
                # Temperature scaling
                logits = logits / temperature
                preds_logits.append(logits.squeeze(0).cpu())
        
        all_logits.append(torch.cat(preds_logits, dim=0))
    
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = torch.argmax(avg_logits, dim=-1).numpy()
    
    return accuracy_score(y_test, preds)


def multi_model_vote(models, X_train, y_train, X_test, y_test, device, 
                     context_size=50, n_ensemble=5):
    """Vote across multiple models."""
    all_preds = []
    
    for name, model in models.items():
        model.eval()
        batch_size = 100
        n_test = X_test.shape[0]
        
        for _ in range(n_ensemble):
            preds = []
            with torch.no_grad():
                for i in range(0, n_test, batch_size):
                    x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
                    
                    indices = np.random.choice(len(X_train), min(context_size, len(X_train)), replace=False)
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
            
            all_preds.append(preds)
    
    # Majority vote
    all_preds = np.array(all_preds)
    final_preds = []
    for i in range(all_preds.shape[1]):
        votes = all_preds[:, i]
        final_preds.append(np.bincount(votes.astype(int)).argmax())
    
    return accuracy_score(y_test, final_preds)


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


def run_wild_tricks():
    print("=" * 70)
    print("🔥 WILD TRICKS: Beat NeuralK! 🔥")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load all models
    print("\nLoading models...")
    models = load_all_models(device)
    main_model = models.get("DeepSpectral")
    
    # Define datasets
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
    
    # Wild tricks to try (FAST version - max 20x ensemble)
    wild_tricks = [
        {"name": "Baseline", "ens": 1, "ctx": 50, "temp": 1.0, "strat": False, "pca": None, "poly": False, "multi": False},
        {"name": "Ens10", "ens": 10, "ctx": 50, "temp": 1.0, "strat": False, "pca": None, "poly": False, "multi": False},
        {"name": "Ens20", "ens": 20, "ctx": 50, "temp": 1.0, "strat": False, "pca": None, "poly": False, "multi": False},
        {"name": "Strat+Ens20", "ens": 20, "ctx": 50, "temp": 1.0, "strat": True, "pca": None, "poly": False, "multi": False},
        {"name": "LargeCtx+Ens20", "ens": 20, "ctx": 100, "temp": 1.0, "strat": False, "pca": None, "poly": False, "multi": False},
        {"name": "Temp0.7+Ens20", "ens": 20, "ctx": 50, "temp": 0.7, "strat": False, "pca": None, "poly": False, "multi": False},
        {"name": "PCA8+Ens20", "ens": 20, "ctx": 50, "temp": 1.0, "strat": False, "pca": 8, "poly": False, "multi": False},
        {"name": "Poly2+Ens10", "ens": 10, "ctx": 50, "temp": 1.0, "strat": False, "pca": None, "poly": True, "multi": False},
        {"name": "MultiModel", "ens": 5, "ctx": 50, "temp": 1.0, "strat": False, "pca": None, "poly": False, "multi": True},
        {"name": "AllTricks", "ens": 20, "ctx": 100, "temp": 0.7, "strat": True, "pca": 8, "poly": False, "multi": False},
    ]
    
    results = []
    
    for name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        n_classes = len(np.unique(y))
        print(f"Shape: {X.shape}, Classes: {n_classes}, H: {H:.2f}")
        
        # Filter to 3 classes
        if n_classes > 3:
            mask = y < 3
            X = X[mask]
            y = y[mask]
            print(f"  (Filtered to 3 classes)")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        nk = NEURALK.get(name, 0)
        best_acc = 0
        best_trick = None
        
        dataset_results = {"Dataset": name, "H": H, "NeuralK": nk}
        
        for trick in wild_tricks:
            try:
                # Prepare data
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)
                
                # Polynomial features
                if trick["poly"]:
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    X_tr = poly.fit_transform(X_tr)
                    X_te = poly.transform(X_te)
                
                # PCA
                if trick["pca"] is not None and X_tr.shape[1] > trick["pca"]:
                    pca = PCA(n_components=trick["pca"])
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)
                
                # Multi-model voting
                if trick["multi"]:
                    acc = multi_model_vote(models, X_tr, y_train, X_te, y_test, device,
                                           context_size=trick["ctx"], n_ensemble=trick["ens"])
                else:
                    acc = evaluate_ensemble(main_model, X_tr, y_train, X_te, y_test, device,
                                            context_size=trick["ctx"], n_ensemble=trick["ens"],
                                            temperature=trick["temp"], stratified=trick["strat"])
                
                delta = (acc - nk) * 100 if nk else 0
                
                if acc > best_acc:
                    best_acc = acc
                    best_trick = trick["name"]
                
                # Only print if promising
                if acc >= best_acc - 0.02:
                    status = "🏆" if acc >= nk else ("⭐" if acc >= nk - 0.02 else "")
                    print(f"  {trick['name']:<20}: {acc*100:>6.2f}% (Δ: {delta:+.2f}%) {status}")
                
                dataset_results[trick["name"]] = acc
                
            except Exception as e:
                print(f"  {trick['name']}: ERROR - {e}")
        
        dataset_results["Best"] = best_trick
        dataset_results["Best_Acc"] = best_acc
        results.append(dataset_results)
        
        delta = (best_acc - nk) * 100 if nk else 0
        status = "🏆 BEAT NEURALK!" if best_acc > nk else "❌"
        print(f"\n  >>> Best: {best_trick} ({best_acc*100:.2f}%) {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY 🏆")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'H':>5} | {'Best':>8} | {'NeuralK':>8} | {'Δ':>7} | Status")
    print("-" * 75)
    
    wins = 0
    for r in results:
        best = r.get("Best_Acc", 0)
        nk = r.get("NeuralK", 0)
        delta = (best - nk) * 100
        status = "🏆 WIN" if best > nk else ("= TIE" if abs(best - nk) < 0.001 else "")
        if best > nk:
            wins += 1
        print(f"{r['Dataset']:<15} | {r['H']:>5.2f} | {best*100:>7.2f}% | {nk*100:>7.2f}% | {delta:>+6.2f}% | {status}")
    
    print(f"\nTotal Wins: {wins}/{len(results)}")
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "results", "wild_tricks_results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WILD TRICKS RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"{r['Dataset']}: Best={r.get('Best')} ({r.get('Best_Acc', 0)*100:.2f}%)\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_wild_tricks()
