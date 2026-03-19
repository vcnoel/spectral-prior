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
Overnight Experiments - Run these before going to bed!

1. Train 3-class model for 100k steps (2x current)
2. Full expanded benchmark with all tricks
3. Feature ablation ensembling

Estimated runtime: ~2-3 hours
"""

import sys
import os
import torch
from torch import nn
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import DeepSpectralPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback


def train_100k():
    """Train for 100k steps - the main overnight experiment."""
    print("=" * 70)
    print("🌙 OVERNIGHT TRAINING: 100k Steps 🌙")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Device: {device}")
    
    # DeepSpectralPrior with optimal H
    HIDDEN_DIM = 32  # For H ≈ 2.44
    prior = DeepSpectralPrior(hidden_dim=HIDDEN_DIM, n_classes=3, device=device)
    
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return prior.get_batch(batch_size, seq_len, n_features, n_classes=3)
    
    # 100k steps, larger context
    STEPS = 100000
    BATCH_SIZE = 4
    
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        num_datapoints_max=100,  # Larger context
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
    
    print(f"Model: NanoTabPFN (3 classes)")
    print(f"Steps: {STEPS}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training... (estimated ~90 min)")
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=criterion,
        epochs=1,
        accumulate_gradients=1,
        lr=1e-4,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name="spectral_100k"
    )
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "overnight")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "spectral_100k.pt")
    torch.save(trained_model.state_dict(), save_path)
    
    print(f"\n✅ Training complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved: {save_path}")
    print(f"Final loss: {loss:.4f}")
    
    return save_path


def run_full_benchmark(model_path):
    """Run full benchmark with all tricks on the new model."""
    print("\n" + "=" * 70)
    print("🔬 RUNNING FULL BENCHMARK ON 100k MODEL 🔬")
    print("=" * 70)
    
    # Import benchmark function
    from sklearn.datasets import load_breast_cancer, load_wine, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    datasets = [
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
        ("Iris", *load_iris(return_X_y=True)),
    ]
    
    NEURALK = {"Wine": 1.0, "Breast Cancer": 0.9883, "Iris": 1.0}
    
    print("\nResults with 100k model + 50x ensemble:")
    
    for name, X, y in datasets:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # 50x ensemble
        all_preds = []
        for _ in range(50):
            preds_batch = []
            with torch.no_grad():
                context_idx = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
                x_c = torch.tensor(X_train[context_idx], dtype=torch.float32).to(device)
                y_c = torch.tensor(y_train[context_idx], dtype=torch.float32).to(device)
                x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                
                x_full = torch.cat([x_c.unsqueeze(0), x_t.unsqueeze(0)], dim=1)
                y_c_uns = y_c.unsqueeze(0).unsqueeze(-1)
                
                logits = model((x_full, y_c_uns), single_eval_pos=x_c.shape[0])
                preds = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()
                all_preds.append(preds)
        
        # Majority vote
        all_preds = np.stack(all_preds, axis=0)
        final = [np.bincount(all_preds[:, i]).argmax() for i in range(len(y_test))]
        acc = accuracy_score(y_test, final)
        
        nk = NEURALK.get(name, 0)
        delta = (acc - nk) * 100
        status = "🏆" if delta >= 0 else ""
        print(f"  {name}: {acc*100:.2f}% (NeuralK: {nk*100:.2f}%, Δ: {delta:+.2f}%) {status}")


if __name__ == "__main__":
    print("🌙" * 35)
    print("    OVERNIGHT EXPERIMENTS - GO TO BED!")
    print("🌙" * 35)
    
    # Run training
    model_path = train_100k()
    
    # Run benchmark
    run_full_benchmark(model_path)
    
    print("\n" + "=" * 70)
    print("🎉 ALL OVERNIGHT EXPERIMENTS COMPLETE! 🎉")
    print("=" * 70)
