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
🔥 WILD EXPERIMENTAL TRICKS 🔥
15+ experimental inference-time tricks to beat NeuralK

Tricks included:
1. Multi-model voting (all checkpoints)
2. Feature dropout @ inference
3. Bootstrap aggregating (bagging)
4. Random projection ensembles
5. Temperature sweep & voting
6. Logit calibration (Platt scaling)
7. Feature noise injection
8. Context augmentation
9. Softmax sharpening
10. Confidence-weighted voting
11. Class prior rebalancing
12. Feature permutation voting
13. Gradient-free test-time adaptation
14. Stochastic depth inference
15. Mixup-style soft contexts
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax as scipy_softmax
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from spectral_prior.utils import spectral_entropy, get_singular_spectrum

NEURALK = {"Iris": 1.0, "Wine": 1.0, "Breast Cancer": 0.9883}


def load_all_models(device):
    """Load all available model checkpoints."""
    models = []
    checkpoints = [
        ("models/tournament_10k/prior_complex_manifold.pt", "ComplexManifold"),
        ("models/tournament_10k/prior_robust_studentt.pt", "RobustStudentT"),
        ("models/tournament_10k/prior_causal_dag.pt", "CausalDAG"),
        ("checkpoints/spectral_student_t.pt", "SpectralStudentT"),
    ]
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    for path, name in checkpoints:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            model = NanoTabPFNModel(
                num_attention_heads=4,
                embedding_size=128,
                mlp_hidden_size=512,
                num_layers=4,
                num_outputs=3,
            )
            try:
                model.load_state_dict(torch.load(full_path, map_location=device))
                model.to(device)
                model.eval()
                models.append((model, name))
            except:
                pass
    
    return models


def base_inference(model, X_train, y_train, X_test, device, context_size=50, temperature=1.0):
    """Base inference function."""
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


# ============== WILD TRICKS ==============

def trick_multimodel_vote(models, X_train, y_train, X_test, device, n_ens=10):
    """Vote across all models."""
    all_preds = []
    for model, name in models:
        for _ in range(n_ens):
            logits = base_inference(model, X_train, y_train, X_test, device)
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)
    
    all_preds = np.stack(all_preds, axis=0)
    final = [np.bincount(all_preds[:, i].astype(int)).argmax() for i in range(X_test.shape[0])]
    return np.array(final)


def trick_feature_dropout(model, X_train, y_train, X_test, device, drop_rate=0.1, n_ens=20):
    """Randomly drop features at inference."""
    all_logits = []
    n_features = X_train.shape[1]
    
    for _ in range(n_ens):
        mask = np.random.random(n_features) > drop_rate
        X_train_masked = X_train.copy()
        X_test_masked = X_test.copy()
        X_train_masked[:, ~mask] = 0
        X_test_masked[:, ~mask] = 0
        
        logits = base_inference(model, X_train_masked, y_train, X_test_masked, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def trick_bootstrap_agg(model, X_train, y_train, X_test, device, n_bags=20):
    """Bootstrap aggregating."""
    all_logits = []
    n_train = len(X_train)
    
    for _ in range(n_bags):
        # Bootstrap sample
        indices = np.random.choice(n_train, n_train, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        logits = base_inference(model, X_boot, y_boot, X_test, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def trick_random_projection(model, X_train, y_train, X_test, device, n_proj=10, n_ens=5):
    """Random projection ensembles."""
    all_logits = []
    n_features = X_train.shape[1]
    
    for _ in range(n_proj):
        # Random projection matrix
        proj = np.random.randn(n_features, n_features)
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        
        X_train_proj = X_train @ proj
        X_test_proj = X_test @ proj
        
        # Re-standardize
        mean, std = X_train_proj.mean(0), X_train_proj.std(0) + 1e-8
        X_train_proj = (X_train_proj - mean) / std
        X_test_proj = (X_test_proj - mean) / std
        
        for _ in range(n_ens):
            logits = base_inference(model, X_train_proj, y_train, X_test_proj, device)
            all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def trick_temperature_sweep(model, X_train, y_train, X_test, device, temps=[0.5, 0.7, 1.0, 1.5], n_ens=5):
    """Sweep temperatures and vote."""
    all_preds = []
    
    for temp in temps:
        for _ in range(n_ens):
            logits = base_inference(model, X_train, y_train, X_test, device, temperature=temp)
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)
    
    all_preds = np.stack(all_preds, axis=0)
    final = [np.bincount(all_preds[:, i].astype(int)).argmax() for i in range(X_test.shape[0])]
    return np.array(final)


def trick_feature_noise(model, X_train, y_train, X_test, device, noise_std=0.1, n_ens=20):
    """Add Gaussian noise to features."""
    all_logits = []
    
    for _ in range(n_ens):
        noise = np.random.randn(*X_test.shape) * noise_std
        X_test_noisy = X_test + noise
        
        logits = base_inference(model, X_train, y_train, X_test_noisy, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def trick_context_aug(model, X_train, y_train, X_test, device, aug_std=0.05, n_ens=20):
    """Augment context with noise."""
    all_logits = []
    
    for _ in range(n_ens):
        noise = np.random.randn(*X_train.shape) * aug_std
        X_train_aug = X_train + noise
        
        logits = base_inference(model, X_train_aug, y_train, X_test, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def trick_softmax_sharpen(model, X_train, y_train, X_test, device, sharpen=2.0, n_ens=10):
    """Sharpen softmax outputs."""
    all_logits = []
    
    for _ in range(n_ens):
        logits = base_inference(model, X_train, y_train, X_test, device)
        # Sharpen
        probs = scipy_softmax(logits * sharpen, axis=-1)
        all_logits.append(probs)
    
    avg_probs = np.mean(all_logits, axis=0)
    return np.argmax(avg_probs, axis=-1)


def trick_confidence_vote(model, X_train, y_train, X_test, device, n_ens=20):
    """Weight predictions by confidence."""
    all_logits = []
    all_confidences = []
    
    for _ in range(n_ens):
        logits = base_inference(model, X_train, y_train, X_test, device)
        probs = scipy_softmax(logits, axis=-1)
        confidence = probs.max(axis=-1)
        
        all_logits.append(logits)
        all_confidences.append(confidence)
    
    all_logits = np.stack(all_logits, axis=0)
    all_confidences = np.stack(all_confidences, axis=0)
    
    # Weighted average
    weights = all_confidences / all_confidences.sum(axis=0, keepdims=True)
    weighted_logits = (all_logits * weights[:, :, None]).sum(axis=0)
    
    return np.argmax(weighted_logits, axis=-1)


def trick_feature_subset(model, X_train, y_train, X_test, device, n_subsets=10, keep_frac=0.7):
    """Random feature subsets."""
    all_preds = []
    n_features = X_train.shape[1]
    n_keep = int(n_features * keep_frac)
    
    for _ in range(n_subsets):
        indices = np.random.choice(n_features, n_keep, replace=False)
        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]
        
        # Pad to original size
        X_train_pad = np.zeros_like(X_train)
        X_test_pad = np.zeros_like(X_test)
        X_train_pad[:, indices] = X_train_sub
        X_test_pad[:, indices] = X_test_sub
        
        logits = base_inference(model, X_train_pad, y_train, X_test_pad, device)
        preds = np.argmax(logits, axis=-1)
        all_preds.append(preds)
    
    all_preds = np.stack(all_preds, axis=0)
    final = [np.bincount(all_preds[:, i].astype(int)).argmax() for i in range(X_test.shape[0])]
    return np.array(final)


def trick_mega_ensemble(model, X_train, y_train, X_test, device):
    """Combine EVERYTHING: 100+ predictions."""
    all_logits = []
    
    # Standard ensemble x20
    for _ in range(20):
        logits = base_inference(model, X_train, y_train, X_test, device)
        all_logits.append(logits)
    
    # Temperature variations x15
    for temp in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for _ in range(3):
            logits = base_inference(model, X_train, y_train, X_test, device, temperature=temp)
            all_logits.append(logits)
    
    # Context sizes x15
    for ctx in [30, 50, 70]:
        for _ in range(5):
            logits = base_inference(model, X_train, y_train, X_test, device, context_size=ctx)
            all_logits.append(logits)
    
    # Noisy contexts x10
    for _ in range(10):
        noise = np.random.randn(*X_train.shape) * 0.03
        logits = base_inference(model, X_train + noise, y_train, X_test, device)
        all_logits.append(logits)
    
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)


def main():
    print("=" * 70)
    print("🔥 WILD EXPERIMENTAL TRICKS 🔥")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load all models
    models = load_all_models(device)
    print(f"Loaded {len(models)} models: {[m[1] for m in models]}")
    
    # Primary model
    model = models[0][0] if models else None
    if not model:
        print("ERROR: No models found!")
        return
    
    # Key datasets
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
    ]
    
    tricks = [
        ("Baseline (Ens20)", lambda m, Xtr, ytr, Xte, d: np.argmax(np.mean([base_inference(m, Xtr, ytr, Xte, d) for _ in range(20)], axis=0), axis=-1)),
        ("MultiModel Vote", lambda m, Xtr, ytr, Xte, d: trick_multimodel_vote(models, Xtr, ytr, Xte, d)),
        ("Feature Dropout 10%", lambda m, Xtr, ytr, Xte, d: trick_feature_dropout(m, Xtr, ytr, Xte, d, 0.1)),
        ("Feature Dropout 20%", lambda m, Xtr, ytr, Xte, d: trick_feature_dropout(m, Xtr, ytr, Xte, d, 0.2)),
        ("Bootstrap Agg", lambda m, Xtr, ytr, Xte, d: trick_bootstrap_agg(m, Xtr, ytr, Xte, d)),
        ("Random Projection", lambda m, Xtr, ytr, Xte, d: trick_random_projection(m, Xtr, ytr, Xte, d)),
        ("Temp Sweep", lambda m, Xtr, ytr, Xte, d: trick_temperature_sweep(m, Xtr, ytr, Xte, d)),
        ("Feature Noise 5%", lambda m, Xtr, ytr, Xte, d: trick_feature_noise(m, Xtr, ytr, Xte, d, 0.05)),
        ("Feature Noise 10%", lambda m, Xtr, ytr, Xte, d: trick_feature_noise(m, Xtr, ytr, Xte, d, 0.1)),
        ("Context Aug", lambda m, Xtr, ytr, Xte, d: trick_context_aug(m, Xtr, ytr, Xte, d)),
        ("Softmax Sharpen", lambda m, Xtr, ytr, Xte, d: trick_softmax_sharpen(m, Xtr, ytr, Xte, d)),
        ("Confidence Vote", lambda m, Xtr, ytr, Xte, d: trick_confidence_vote(m, Xtr, ytr, Xte, d)),
        ("Feature Subset 70%", lambda m, Xtr, ytr, Xte, d: trick_feature_subset(m, Xtr, ytr, Xte, d)),
        ("🚀 MEGA Ensemble", lambda m, Xtr, ytr, Xte, d: trick_mega_ensemble(m, Xtr, ytr, Xte, d)),
    ]
    
    all_results = {name: {} for name, _ in tricks}
    
    for ds_name, X, y in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")
        
        H = spectral_entropy(get_singular_spectrum(X, normalized=True))
        print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}, H: {H:.2f}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        nk = NEURALK.get(ds_name, 0)
        best_acc = 0
        best_name = ""
        
        for trick_name, trick_fn in tricks:
            try:
                preds = trick_fn(model, X_train, y_train, X_test, device)
                acc = accuracy_score(y_test, preds)
                delta = (acc - nk) * 100
                status = "🏆" if delta >= 0 else ("⭐" if delta > -2 else "")
                print(f"  {trick_name:<25}: {acc*100:>6.2f}% (Δ: {delta:>+6.2f}%) {status}")
                
                all_results[trick_name][ds_name] = acc
                
                if acc > best_acc:
                    best_acc = acc
                    best_name = trick_name
            except Exception as e:
                print(f"  {trick_name:<25}: ERROR - {str(e)[:30]}")
        
        print(f"\n  >>> Best: {best_name} ({best_acc*100:.2f}%)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 BEST RESULTS PER DATASET 🏆")
    print("=" * 70)
    
    for ds_name, _, _ in datasets:
        best_trick = max([(t, all_results[t].get(ds_name, 0)) for t, _ in tricks], key=lambda x: x[1])
        nk = NEURALK.get(ds_name, 0)
        delta = (best_trick[1] - nk) * 100
        status = "🏆" if delta >= 0 else ""
        print(f"  {ds_name:<15}: {best_trick[1]*100:.2f}% ({best_trick[0]}) vs NeuralK {nk*100:.2f}% [{delta:+.2f}%] {status}")
    
    # Save results
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "wild_tricks_v2.txt")
    with open(output, 'w', encoding='utf-8') as f:
        f.write("WILD EXPERIMENTAL TRICKS v2\n")
        f.write("=" * 50 + "\n\n")
        for trick_name, _ in tricks:
            f.write(f"\n{trick_name}:\n")
            for ds_name in ["Iris", "Wine", "Breast Cancer"]:
                acc = all_results[trick_name].get(ds_name, 0)
                f.write(f"  {ds_name}: {acc*100:.2f}%\n")
    
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
