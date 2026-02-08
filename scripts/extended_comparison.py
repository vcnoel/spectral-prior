"""
============================================================================
EXTENDED PRIOR COMPARISON
============================================================================
1. Apply wild tricks to 50k Idea 1 & 2 models
2. Train H=2.44 at 50k steps
3. Compare everything
============================================================================
"""

import sys
import os
import torch
from torch import nn
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import ConsoleLoggerCallback
from spectral_prior.priors import DeepSpectralPrior

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TRICKS
# ============================================================================

def baseline_predict(model, Xtr, ytr, Xte, device, ctx_size=50, n_ensemble=20):
    """Baseline ensemble prediction."""
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
    """60-member ensemble with temperature sweep."""
    all_logits = []
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

def dropout_predict(model, Xtr, ytr, Xte, device, dropout=0.1):
    """Feature dropout prediction."""
    all_logits = []
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

def bootstrap_predict(model, Xtr, ytr, Xte, device, n_bootstrap=30):
    """Bootstrap aggregating."""
    all_logits = []
    for _ in range(n_bootstrap):
        with torch.no_grad():
            idx = np.random.choice(len(Xtr), len(Xtr), replace=True)
            x_c = torch.tensor(Xtr[idx][:50], dtype=torch.float32).to(device)
            y_c = torch.tensor(ytr[idx][:50], dtype=torch.float32).to(device)
            x_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            
            x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
            
            logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
            all_logits.append(logits.squeeze(0).cpu().numpy())
    
    return np.mean(all_logits, axis=0)

def sharpen_predict(model, Xtr, ytr, Xte, device, tau=0.5):
    """Sharpen predictions with temperature."""
    logits = baseline_predict(model, Xtr, ytr, Xte, device)
    return logits / tau


# ============================================================================
# MAIN
# ============================================================================

TRICKS = {
    "Baseline": baseline_predict,
    "MegaEns60": mega_ensemble_predict,
    "Dropout10%": lambda m, xr, yr, xt, d: dropout_predict(m, xr, yr, xt, d, 0.1),
    "Bootstrap": bootstrap_predict,
    "Sharpen": lambda m, xr, yr, xt, d: sharpen_predict(m, xr, yr, xt, d, 0.5),
}

NEURALK = {
    "Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883,
    "Vehicle": 0.8154, "Vowel": 1.0
}

def load_datasets():
    """Load target datasets."""
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
    ]
    
    for name in ["vehicle", "vowel"]:
        try:
            data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
            df = data.data
            y = data.target
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    try:
                        df[col] = df[col].astype(float)
                    except:
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            X = df.values.astype(float)
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                X = SimpleImputer(strategy='mean').fit_transform(X)
            if y.dtype == object or y.dtype == 'category':
                y = LabelEncoder().fit_transform(y.astype(str))
            else:
                y = y.values.astype(int)
            datasets.append((name.capitalize(), X, y))
        except:
            pass
    
    return datasets


def evaluate_with_tricks(model, datasets, device):
    """Evaluate model with all tricks."""
    results = {}
    
    for ds_name, X, y in datasets:
        n_classes = len(np.unique(y))
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
        
        results[ds_name] = {}
        
        for trick_name, trick_fn in TRICKS.items():
            logits = trick_fn(model, Xtr, ytr, Xte, device)
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(yte, preds)
            results[ds_name][trick_name] = acc
    
    return results


def train_h244_50k(device):
    """Train classical H=2.44 model with 50k steps."""
    print("\n" + "=" * 70)
    print("TRAINING: H=2.44 (Classical) - 50k Steps")
    print("=" * 70)
    
    prior = DeepSpectralPrior(hidden_dim=32, n_classes=3, device=device)
    
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return prior.get_batch(batch_size, seq_len, n_features, n_classes=3)
    
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=50000,
        batch_size=4,
        num_datapoints_max=100,
        num_features=20,
        device=device
    )
    
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3,
    )
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=1,
        accumulate_gradients=1,
        lr=1e-4,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name="h244_50k"
    )
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "models", "h244_50k")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "h244_50k.pt")
    torch.save(trained_model.state_dict(), save_path)
    print(f"✅ Saved: {save_path}")
    
    return trained_model


def main():
    print("=" * 70)
    print("🔬 EXTENDED PRIOR COMPARISON 🔬")
    print("=" * 70)
    
    device = get_default_device()
    print(f"Device: {device}")
    
    datasets = load_datasets()
    print(f"Datasets: {[d[0] for d in datasets]}")
    
    all_results = {}
    
    # 1. Load 50k Idea 1 & 2 models
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "advanced_priors")
    
    for idea_name in ["dynamic_spectral", "hybrid_mixture"]:
        model_path = os.path.join(model_dir, f"{idea_name}.pt")
        if os.path.exists(model_path):
            print(f"\n📦 Loading {idea_name} (50k)...")
            model = NanoTabPFNModel(
                num_attention_heads=4, embedding_size=128,
                mlp_hidden_size=512, num_layers=4, num_outputs=3
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            print(f"📊 Evaluating {idea_name} with tricks...")
            results = evaluate_with_tricks(model, datasets, device)
            all_results[f"{idea_name}_50k"] = results
    
    # 2. Train H=2.44 50k
    h244_model = train_h244_50k(device)
    h244_model.to(device)
    h244_model.eval()
    
    print("\n📊 Evaluating H=2.44 50k with tricks...")
    all_results["h244_50k"] = evaluate_with_tricks(h244_model, datasets, device)
    
    # 3. Load H=2.44 10k for comparison
    h244_10k_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "models", "tournament_10k", "deep_spectral.pt")
    if os.path.exists(h244_10k_path):
        print("\n📦 Loading H=2.44 10k for comparison...")
        model_10k = NanoTabPFNModel(
            num_attention_heads=4, embedding_size=128,
            mlp_hidden_size=512, num_layers=4, num_outputs=3
        )
        model_10k.load_state_dict(torch.load(h244_10k_path, map_location=device))
        model_10k.to(device)
        model_10k.eval()
        all_results["h244_10k"] = evaluate_with_tricks(model_10k, datasets, device)
    
    # 4. Print comparison
    print("\n" + "=" * 70)
    print("🏆 FINAL COMPARISON 🏆")
    print("=" * 70)
    
    for ds in ["Iris", "Wine", "Breast Cancer", "Vehicle", "Vowel"]:
        print(f"\n{ds} (NeuralK: {NEURALK.get(ds, 0)*100:.0f}%)")
        print("-" * 60)
        for model_name, results in all_results.items():
            if ds in results:
                best_trick = max(results[ds].items(), key=lambda x: x[1])
                print(f"  {model_name:20} | Best: {best_trick[0]:12} = {best_trick[1]*100:6.2f}%")
    
    # Save results
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "results", "extended_comparison.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("EXTENDED PRIOR COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        for model_name, results in all_results.items():
            f.write(f"\n{model_name}:\n")
            for ds, tricks in results.items():
                best = max(tricks.items(), key=lambda x: x[1])
                f.write(f"  {ds}: {best[0]} = {best[1]*100:.2f}%\n")
    
    print(f"\n✅ Results saved: {output}")


if __name__ == "__main__":
    main()
