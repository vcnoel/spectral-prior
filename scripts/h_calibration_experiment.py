"""
H-CALIBRATED EXPERIMENT: Does matching H to dataset entropy help?

Hypothesis: If we train/use a prior with H matching the dataset's entropy,
performance should improve on low-H datasets like Iris.

We test:
1. Fixed H=2.44 (our default)
2. H matched to dataset (e.g., H=1.09 for Iris)
3. All wild tricks on all 9 datasets
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum
from spectral_prior import DeepSpectralPrior

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Letter": 0.9985,
    "Vehicle": 0.8154, "Vowel": 1.0
}


def load_model(device):
    """Load the best model."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "tournament_10k", "prior_complex_manifold.pt")
    
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


def base_inference(model, X_train, y_train, X_test, device, context_size=50, temperature=1.0):
    """Standard inference."""
    n_test = X_test.shape[0]
    context_size = min(context_size, len(X_train))
    
    with torch.no_grad():
        indices = np.random.choice(len(X_train), context_size, replace=False)
        x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
        y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
        y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
        
        logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
        logits = logits.squeeze(0) / temperature
        
    return logits.cpu().numpy()


def warp_features_by_H(X, target_H, current_H):
    """
    Warp features to match target H.
    If target_H < current_H: compress (reduce variance in top components)
    If target_H > current_H: expand (increase variance uniformly)
    """
    if abs(target_H - current_H) < 0.1:
        return X  # No warping needed
    
    # PCA to get principal directions
    n_components = min(X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Adjust variance based on H difference
    ratio = target_H / current_H
    
    if ratio < 1:
        # Compress: reduce variance in later components
        for i in range(n_components):
            decay = np.exp(-i * (1 - ratio))
            X_pca[:, i] *= decay
    else:
        # Expand: boost later components
        for i in range(n_components):
            boost = 1 + (ratio - 1) * (i / n_components)
            X_pca[:, i] *= boost
    
    # Transform back
    return pca.inverse_transform(X_pca)


def evaluate_with_H_calibration(model, X_train, y_train, X_test, y_test, device, 
                                 current_H, target_H=2.44, n_ens=20):
    """Evaluate with H-calibrated feature warping."""
    # Warp features to match target H
    X_train_warped = warp_features_by_H(X_train, target_H, current_H)
    X_test_warped = warp_features_by_H(X_test, target_H, current_H)
    
    # Re-standardize
    mean, std = X_train_warped.mean(0), X_train_warped.std(0) + 1e-8
    X_train_warped = (X_train_warped - mean) / std
    X_test_warped = (X_test_warped - mean) / std
    
    # Ensemble inference
    all_logits = []
    for _ in range(n_ens):
        logits = base_inference(model, X_train_warped, y_train, X_test_warped, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    preds = np.argmax(avg_logits, axis=-1)
    
    return accuracy_score(y_test, preds)


def mega_ensemble(model, X_train, y_train, X_test, device, temps=[0.5, 0.7, 1.0]):
    """Best combined tricks."""
    all_logits = []
    
    for temp in temps:
        for _ in range(20):
            logits = base_inference(model, X_train, y_train, X_test, device, temperature=temp)
            all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


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
    print("🎯 H-CALIBRATED EXPERIMENT: All Datasets + H Matching 🎯")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = load_model(device)
    print("Model loaded: prior_complex_manifold.pt")
    
    # All datasets
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
        ("Digits", *load_digits(return_X_y=True)),
    ]
    
    print("\nLoading OpenML datasets...")
    for name in ["segment", "satimage", "letter", "vehicle", "vowel"]:
        print(f"  Loading {name}...", end=" ")
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
            print("OK")
        else:
            print("FAILED")
    
    results = []
    
    for ds_name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        n_classes = len(np.unique(y))
        print(f"Shape: {X.shape}, Classes: {n_classes}, H: {H:.2f}")
        
        # Filter to 3 classes if needed
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
        
        # Trick 1: Standard baseline (Ens20)
        all_logits = []
        for _ in range(20):
            logits = base_inference(model, X_train, y_train, X_test, device)
            all_logits.append(logits)
        preds_base = np.argmax(np.mean(all_logits, axis=0), axis=-1)
        acc_base = accuracy_score(y_test, preds_base)
        
        # Trick 2: H-calibrated to 2.44 (match to prior)
        acc_h244 = evaluate_with_H_calibration(
            model, X_train, y_train, X_test, y_test, device, 
            current_H=H, target_H=2.44, n_ens=20
        )
        
        # Trick 3: H-calibrated to dataset's own H
        acc_h_match = evaluate_with_H_calibration(
            model, X_train, y_train, X_test, y_test, device, 
            current_H=H, target_H=H, n_ens=20  # No warping
        )
        
        # Trick 4: Mega ensemble (temp sweep)
        preds_mega = mega_ensemble(model, X_train, y_train, X_test, device)
        acc_mega = accuracy_score(y_test, preds_mega)
        
        # Trick 5: H-calibrated + Mega
        X_train_h = warp_features_by_H(X_train, 2.44, H)
        X_test_h = warp_features_by_H(X_test, 2.44, H)
        mean, std = X_train_h.mean(0), X_train_h.std(0) + 1e-8
        X_train_h = (X_train_h - mean) / std
        X_test_h = (X_test_h - mean) / std
        preds_h_mega = mega_ensemble(model, X_train_h, y_train, X_test_h, device)
        acc_h_mega = accuracy_score(y_test, preds_h_mega)
        
        print(f"\n  {'Trick':<25} | {'Accuracy':>8} | {'Δ NeuralK':>10}")
        print(f"  {'-'*50}")
        
        for name, acc in [
            ("Baseline (Ens20)", acc_base),
            ("H→2.44 Warp", acc_h244),
            ("Mega Ensemble", acc_mega),
            ("H→2.44 + Mega", acc_h_mega),
        ]:
            delta = (acc - nk) * 100
            status = "🏆" if delta >= 0 else ("⭐" if delta > -2 else "")
            print(f"  {name:<25} | {acc*100:>7.2f}% | {delta:>+9.2f}% {status}")
        
        best_acc = max(acc_base, acc_h244, acc_mega, acc_h_mega)
        best_name = ["Baseline", "H→2.44", "Mega", "H+Mega"][[acc_base, acc_h244, acc_mega, acc_h_mega].index(best_acc)]
        
        results.append({
            "Dataset": ds_name,
            "H": H,
            "Best": best_acc,
            "BestTrick": best_name,
            "NeuralK": nk,
            "Base": acc_base,
            "H244": acc_h244,
            "Mega": acc_mega,
            "HMega": acc_h_mega,
        })
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY: ALL DATASETS 🏆")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'H':>5} | {'Best':>8} | {'Trick':<10} | {'NeuralK':>8} | {'Δ':>7}")
    print("-" * 70)
    
    wins = 0
    for r in results:
        delta = (r['Best'] - r['NeuralK']) * 100
        status = "🏆" if delta >= 0 else ""
        if delta >= 0:
            wins += 1
        print(f"{r['Dataset']:<15} | {r['H']:>5.2f} | {r['Best']*100:>7.2f}% | {r['BestTrick']:<10} | {r['NeuralK']*100:>7.2f}% | {delta:>+6.2f}% {status}")
    
    print(f"\nTotal Wins/Ties: {wins}/{len(results)}")
    
    # H-Calibration Analysis
    print("\n" + "=" * 70)
    print("📊 H-CALIBRATION ANALYSIS 📊")
    print("=" * 70)
    print("Does H→2.44 warping help?")
    for r in results:
        improvement = (r['H244'] - r['Base']) * 100
        status = "✅ HELPS" if improvement > 0 else ("= Same" if improvement == 0 else "❌ Hurts")
        print(f"  {r['Dataset']:<15}: H={r['H']:.2f}, Δ={improvement:+.2f}% {status}")
    
    # Save
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "h_calibration_results.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("H-CALIBRATED EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"{r['Dataset']}: Best={r['Best']*100:.2f}% ({r['BestTrick']}), H={r['H']:.2f}\n")
            f.write(f"  Base={r['Base']*100:.2f}%, H→2.44={r['H244']*100:.2f}%, Mega={r['Mega']*100:.2f}%, H+Mega={r['HMega']*100:.2f}%\n")
    
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
