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


import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_digits

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfmplayground.model import NanoTabPFNModel

def evaluate_nanotabpfn(model, X_train, y_train, X_test, y_test, device):
    # Prepare data for NanoTabPFN (Batch, Seq, Feat)
    # We treat the entire training set as the "context" for each test point?
    # No, TabPFN takes (X_train, y_train, X_test) and predicts y_test.
    # But for large datasets, we can't pass all training data as context (context window limit).
    # We will sample a random context of size 50 for each test batch.
    
    model.eval()
    batch_size = 100
    n_test = X_test.shape[0]
    preds = []
    
    # Context Selection
    # Fixed context for stability or random? TabPFN papers say random.
    context_size = min(50, X_train.shape[0])
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            # Batch of test points
            x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
            
            # Random context
            indices = np.random.choice(X_train.shape[0], context_size, replace=False)
            x_c = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
            y_c = torch.tensor(y_train[indices], dtype=torch.float32).to(device)
            
            # Model forward
            # model(x_train, y_train, x_test) -> logits
            # But NanoTabPFN expects specific shapes
            # (Batch, Seq, Feat).
            # We need to replicate context for each test point? 
            # Or does the model handle "Batch of tasks"?
            # Standard TabPFN: Input is (B, N, D).
            # Here B=1 (single task) or B=batch_size (if we treat each test point as a task with same context).
            
            # Let's treat it as 1 task with many query points?
            # Model definition: forward((x,y), single_eval_pos)
            # x = concat(x_c, x_b)
            # y = y_c
            
            # We need to reshape for the expected input (1, Seq, Feat)
            x_c_uns = x_c.unsqueeze(0) # (1, 50, D)
            y_c_uns = y_c.unsqueeze(0).unsqueeze(-1) # (1, 50, 1)
            x_b_uns = x_b.unsqueeze(0) # (1, batch, D)
            
            # Prepare Full Input
            x_full = torch.cat([x_c_uns, x_b_uns], dim=1) # (1, 50+batch, D)
            
            # Explicit call
            logits = model((x_full, y_c_uns), single_eval_pos=x_c_uns.shape[1]) # (1, batch, n_classes)
            
            batch_preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Print debug
            print(f"Batch {i}: inputs {x_b.shape} -> logits {logits.shape} -> preds {batch_preds.shape}")
            
            if batch_preds.ndim == 0:
                preds.append(batch_preds.item())
            else:
                preds.extend(batch_preds)
                
    preds = np.array(preds)
    # print(f"Total preds: {len(preds)}, y_test: {len(y_test)}")
    return accuracy_score(y_test, preds)

def check_path(filename):
    # Check specific folder first, then current dir
    paths = [
        f"models/tournament_10k/{filename}",
        filename
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def benchmark_models():
    print("--- STARTING TRIPLE THREAT + UNIVERSAL BENCHMARK ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Datasets
    # breast_cancer: 30 features, 2 classes.
    # wine: 13 features, 3 classes.
    # digits: 64 features, 10 classes.
    
    try:
        datasets = [
            ("Breast Cancer", load_breast_cancer()),
            ("Wine", load_wine()),
            ("Digits", load_digits())
        ]
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # 2. Load Models
    print("Loading models...")
    
    models = {}
    
    # Helper to load
    def load_net(path):
        m = NanoTabPFNModel(128, 4, 512, 4, 3)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.to(device)
        return m

    # Gaussian
    p_gauss = check_path("baseline_gaussian.pt")
    if p_gauss: models['Gaussian'] = load_net(p_gauss)
    
    # Robust
    p_robust = check_path("prior_robust_studentt.pt")
    if p_robust: models['Robust'] = load_net(p_robust)
    
    # Manifold
    p_manifold = check_path("prior_complex_manifold.pt")
    if p_manifold: models['DeepSpectralPrior'] = load_net(p_manifold)
    
    # Causal
    p_causal = check_path("prior_causal_dag.pt")
    if p_causal: models['Causal'] = load_net(p_causal)

    results = []

    for name, data in datasets:
        print(f"\nBenchmarking on {name}...")
        X, y = data.data, data.target
        
        # Filter classes if > 3 (because our models outputs 3 classes)
        if len(np.unique(y)) > 3:
            print(f"  Filtering {name} to first 3 classes...")
            mask = y < 3
            X = X[mask]
            y = y[mask]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        row = {"dataset": name}

        # A. NeuralK SOTA (Hardcoded from Phase 6 results to avoid API dependency)
        nk_acc = 0.0
        print("  1. Querying NeuralK API (SOTA) [Cached]...")
        if name == "Breast Cancer":
            nk_acc = 0.9708
        elif name == "Wine":
            nk_acc = 0.9815
        elif name == "Digits":
            nk_acc = 1.0000
        print(f"     -> NeuralK Accuracy: {nk_acc:.4f}")
        row["NeuralK"] = nk_acc
        
        # Evaluate Loaded Models
        for m_name, model in models.items():
            print(f"  Evaluating {m_name}...")
            acc = evaluate_nanotabpfn(model, X_train, y_train, X_test, y_test, device)
            print(f"     -> {m_name}: {acc:.4f}")
            row[m_name] = acc
            
        results.append(row)

    print("\n--- UNIVERSAL TOURNAMENT RESULTS ---")
    # Dynamic Headers
    model_names = list(models.keys())
    headers = ["Dataset", "NeuralK"] + model_names + ["Winner"]
    
    # Print Headers
    header_str = f"{headers[0]:<15} | {headers[1]:<8}"
    for m in model_names:
        header_str += f" | {m:<10}"
    header_str += f" | {headers[-1]:<10}"
    print(header_str)
    print("-" * len(header_str))
    
    for r in results:
        # Determine Winner (excluding NeuralK)
        local_scores = {k: v for k, v in r.items() if k in model_names}
        if local_scores:
            winner = max(local_scores, key=local_scores.get)
        else:
            winner = "N/A"
            
        row_str = f"{r['dataset']:<15} | {r.get('NeuralK', 0.0):.4f}  "
        for m in model_names:
            score = r.get(m, 0.0)
            row_str += f" | {score:.4f}    "
        row_str += f" | {winner:<10}"
        print(row_str)

if __name__ == "__main__":
    benchmark_models()
