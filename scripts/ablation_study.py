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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import numpy as np
import spectral_trust
from spectral_prior import SpectralStudentTPrior, DeepSpectralPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.priors.dataloader import PriorDataLoader
from torch import nn
from tfmplayground.callbacks import ConsoleLoggerCallback
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, X_train, y_test, y_train, device):
    model.eval()
    batch_size = 100
    context_size = min(50, X_train.shape[0])
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            x_b = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
            indices = np.random.choice(X_train.shape[0], context_size, replace=False)
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
    return accuracy_score(y_test, preds)

def run_ablations():
    device = get_default_device()
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")
    
    # Check Wine for Eval
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # === ABLATION 1: LATENT DISTRIBUTION ===
    print("\n=== ABLATION 1: LATENT DISTRIBUTION ===")
    nus = [100.0, 2.0]
    p_fixed = 0.2 # Optimal from Phase 3
    
    res_dist = []
    
    for nu in nus:
        print(f"\n--- Training with nu={nu} (Gaussian-ish if >=100) ---")
        prior = SpectralStudentTPrior(nu=nu, p=p_fixed, device=device)
        
        # Wrap in PriorDataLoader
        # 3000 steps * 32 batch size? No, num_steps is total steps.
        # But PriorDataLoader num_steps is "steps per epoch"? 
        # In train_tournament, steps=10000 passed to loader.
        # train() loop does: for epoch in range(epochs): for batch in loader: ...
        # If loader.num_steps is large, one epoch is enough.
        
        loader = PriorDataLoader(
            prior.get_batch,
            num_steps=50,
            batch_size=128,
            num_datapoints_max=50,
            num_features=30,
            device=device
        )
        
        model = NanoTabPFNModel(128, 4, 512, 4, 3).to(device)
        criterion = nn.CrossEntropyLoss()
        
        train(
            model,
            loader,
            criterion,
            epochs=1,
            lr=3e-4,
            device=device
        )
        
        acc = evaluate_model(model, X_test, X_train, y_test, y_train, device)
        print(f"Result nu={nu}: Acc={acc:.4f}")
        res_dist.append({"nu": nu, "acc": acc})
        
    print("\nRESULTS: LATENT DISTRIBUTION")
    for r in res_dist:
        print(r)

    # === ABLATION 2: SPECTRAL ENTROPY (Via DeepSpectralPrior Hidden Dim) ===
    print("\n=== ABLATION 2: SPECTRAL ENTROPY (Via Hidden Dim) ===")
    # Reduced: Low Entropy (8 -> 2.07) vs High/Default (128 -> 3.32)
    # Target was ~2.44. 
    # Maybe 16 -> 2.76 is better "High".
    # Let's do [8, 16, 128].
    dims = [8, 16, 128]
    
    res_ent = []
    
    for d in dims:
        print(f"\n--- Training DeepSpectralPrior with hidden_dim={d} ---")
        
        # Instantiate DeepSpectralPrior with specific dim
        prior = DeepSpectralPrior(hidden_dim=d, device=device)
        
        loader = PriorDataLoader(
            prior.get_batch,
            num_steps=50,
            batch_size=128,
            num_datapoints_max=50,
            num_features=30,
            device=device
        )
        
        model = NanoTabPFNModel(128, 4, 512, 4, 3).to(device)
        criterion = nn.CrossEntropyLoss()

        train(
            model,
            loader,
            criterion,
            epochs=1,
            lr=3e-4,
            device=device
        )
        
        acc = evaluate_model(model, X_test, X_train, y_test, y_train, device)
        print(f"Result dim={d}: Acc={acc:.4f}")
        res_ent.append({"dim": d, "acc": acc})

    print("\nRESULTS: SPECTRAL ENTROPY")
    for r in res_ent:
        print(r)

if __name__ == "__main__":
    run_ablations()
