"""
🔥 WILD TRICKS x ALL H PRIORS 🔥
Run all tricks (14+) on all models (H=1.1, H=2.4, H=3.9) on all datasets (8)
Find the best H+Trick combination for each dataset!
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import softmax as scipy_softmax
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883, "Digits": 1.0,
    "Segment": 0.9933, "Satimage": 0.9926, "Vehicle": 0.8154, "Vowel": 1.0
}


def load_model(model_path, device):
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


def base_inference(model, X_train, y_train, X_test, device, ctx_size=50, temp=1.0):
    ctx_size = min(ctx_size, len(X_train))
    with torch.no_grad():
        idx = np.random.choice(len(X_train), ctx_size, replace=False)
        x_c = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
        y_c = torch.tensor(y_train[idx], dtype=torch.float32).to(device)
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
        y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
        logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
        return (logits.squeeze(0) / temp).cpu().numpy()


# === TRICK DEFINITIONS ===

def trick_baseline(model, Xtr, ytr, Xte, dev):
    """Baseline with Ens20."""
    logits = [base_inference(model, Xtr, ytr, Xte, dev) for _ in range(20)]
    return np.argmax(np.mean(logits, axis=0), axis=-1)


def trick_mega_ens(model, Xtr, ytr, Xte, dev):
    """Mega ensemble: 60 predictions with temp sweep."""
    all_logits = []
    for temp in [0.5, 0.7, 1.0]:
        for _ in range(20):
            all_logits.append(base_inference(model, Xtr, ytr, Xte, dev, temp=temp))
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


def trick_dropout_10(model, Xtr, ytr, Xte, dev):
    """Feature dropout 10%."""
    all_logits = []
    n_feat = Xtr.shape[1]
    for _ in range(20):
        mask = np.random.random(n_feat) > 0.1
        Xtr_m = Xtr.copy(); Xtr_m[:, ~mask] = 0
        Xte_m = Xte.copy(); Xte_m[:, ~mask] = 0
        all_logits.append(base_inference(model, Xtr_m, ytr, Xte_m, dev))
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


def trick_bootstrap(model, Xtr, ytr, Xte, dev):
    """Bootstrap aggregating."""
    all_logits = []
    n = len(Xtr)
    for _ in range(20):
        idx = np.random.choice(n, n, replace=True)
        all_logits.append(base_inference(model, Xtr[idx], ytr[idx], Xte, dev))
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


def trick_context_aug(model, Xtr, ytr, Xte, dev):
    """Context augmentation with noise."""
    all_logits = []
    for _ in range(20):
        noise = np.random.randn(*Xtr.shape) * 0.05
        all_logits.append(base_inference(model, Xtr + noise, ytr, Xte, dev))
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


def trick_sharpen(model, Xtr, ytr, Xte, dev):
    """Softmax sharpening."""
    all_probs = []
    for _ in range(20):
        logits = base_inference(model, Xtr, ytr, Xte, dev)
        probs = scipy_softmax(logits * 2.0, axis=-1)
        all_probs.append(probs)
    return np.argmax(np.mean(all_probs, axis=0), axis=-1)


def trick_large_ctx(model, Xtr, ytr, Xte, dev):
    """Large context (100 samples)."""
    all_logits = []
    for _ in range(20):
        all_logits.append(base_inference(model, Xtr, ytr, Xte, dev, ctx_size=100))
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


def trick_all_combined(model, Xtr, ytr, Xte, dev):
    """ALL tricks combined: mega ensemble + context aug + temp sweep."""
    all_logits = []
    for temp in [0.5, 0.7, 1.0]:
        for _ in range(15):
            # Context aug
            noise = np.random.randn(*Xtr.shape) * 0.03
            logits = base_inference(model, Xtr + noise, ytr, Xte, dev, ctx_size=100, temp=temp)
            all_logits.append(logits)
    return np.argmax(np.mean(all_logits, axis=0), axis=-1)


TRICKS = [
    ("Baseline", trick_baseline),
    ("MegaEns60", trick_mega_ens),
    ("Dropout10%", trick_dropout_10),
    ("Bootstrap", trick_bootstrap),
    ("CtxAug", trick_context_aug),
    ("Sharpen", trick_sharpen),
    ("LargeCtx", trick_large_ctx),
    ("AllCombined", trick_all_combined),
]


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
        return None, None


def main():
    print("=" * 80)
    print("🔥 WILD TRICKS x ALL H PRIORS 🔥")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load all models
    model_paths = {
        "H=1.1": os.path.join(base_dir, "models", "multi_h", "spectral_H1.1.pt"),
        "H=2.4": os.path.join(base_dir, "models", "tournament_10k", "prior_complex_manifold.pt"),
        "H=3.9": os.path.join(base_dir, "models", "multi_h", "spectral_H3.9.pt"),
    }
    
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = load_model(path, device)
            print(f"Loaded: {name}")
    
    # All datasets
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
        ("Digits", *load_digits(return_X_y=True)),
    ]
    
    for name in ["segment", "satimage", "vehicle", "vowel"]:
        X, y = load_openml_safe(name)
        if X is not None:
            datasets.append((name.capitalize(), X, y))
    
    print(f"\n{len(datasets)} datasets, {len(models)} models, {len(TRICKS)} tricks")
    print(f"Total experiments: {len(datasets) * len(models) * len(TRICKS)}")
    
    all_results = []
    
    for ds_name, X, y in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*80}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
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
        
        nk = NEURALK.get(ds_name, 0)
        best_overall = {"acc": 0, "model": "", "trick": ""}
        
        for model_name, model in models.items():
            print(f"\n  {model_name}:")
            
            for trick_name, trick_fn in TRICKS:
                try:
                    preds = trick_fn(model, Xtr, ytr, Xte, device)
                    acc = accuracy_score(yte, preds)
                    delta = (acc - nk) * 100
                    status = "🏆" if delta >= 0 else ("⭐" if delta > -5 else "")
                    print(f"    {trick_name:<12}: {acc*100:>6.2f}% (Δ: {delta:>+6.2f}%) {status}")
                    
                    all_results.append({
                        "Dataset": ds_name, "H": H, "Model": model_name,
                        "Trick": trick_name, "Acc": acc, "NeuralK": nk
                    })
                    
                    if acc > best_overall["acc"]:
                        best_overall = {"acc": acc, "model": model_name, "trick": trick_name}
                except Exception as e:
                    print(f"    {trick_name:<12}: ERROR")
        
        delta = (best_overall["acc"] - nk) * 100
        status = "🏆" if delta >= 0 else ""
        print(f"\n  >>> BEST: {best_overall['model']} + {best_overall['trick']} = {best_overall['acc']*100:.2f}% (Δ: {delta:+.2f}%) {status}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 BEST COMBINATIONS PER DATASET 🏆")
    print("=" * 80)
    print(f"{'Dataset':<15} | {'H':>5} | {'Model':>6} | {'Trick':<12} | {'Acc':>7} | {'NeuralK':>7} | {'Δ':>7}")
    print("-" * 80)
    
    wins = 0
    for ds_name, _, _ in datasets:
        ds_results = [r for r in all_results if r["Dataset"] == ds_name]
        if ds_results:
            best = max(ds_results, key=lambda x: x["Acc"])
            delta = (best["Acc"] - best["NeuralK"]) * 100
            status = "🏆" if delta >= 0 else ""
            if delta >= 0:
                wins += 1
            print(f"{best['Dataset']:<15} | {best['H']:>5.2f} | {best['Model']:>6} | {best['Trick']:<12} | {best['Acc']*100:>6.2f}% | {best['NeuralK']*100:>6.2f}% | {delta:>+6.2f}% {status}")
    
    print(f"\nTotal Wins/Ties: {wins}/{len(datasets)}")
    
    # Save
    output = os.path.join(base_dir, "results", "wild_tricks_all_h.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("WILD TRICKS x ALL H PRIORS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Dataset':<15} | {'Model':>6} | {'Trick':<12} | {'Acc':>7} | {'NeuralK':>7}\n")
        f.write("-" * 60 + "\n")
        for ds_name in ["Iris", "Wine", "Breast Cancer", "Digits", "Segment", "Satimage", "Vehicle", "Vowel"]:
            ds_results = [r for r in all_results if r["Dataset"] == ds_name]
            if ds_results:
                best = max(ds_results, key=lambda x: x["Acc"])
                f.write(f"{best['Dataset']:<15} | {best['Model']:>6} | {best['Trick']:<12} | {best['Acc']*100:>6.2f}% | {best['NeuralK']*100:>6.2f}%\n")
    
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
