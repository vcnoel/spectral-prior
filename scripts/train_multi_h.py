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
MULTI-H PRIOR TRAINING
Train models with H≈1.1 (for Iris) and H≈3.9 (for Digits)

The spectral entropy H is controlled by the hidden_dim in DeepSpectralPrior:
- Lower hidden_dim → Lower H (simpler, more concentrated spectrum)
- Higher hidden_dim → Higher H (more complex, flatter spectrum)

Based on our analysis:
- H ≈ 1.1: hidden_dim ~ 8
- H ≈ 2.4: hidden_dim ~ 32 (our current default)
- H ≈ 3.9: hidden_dim ~ 128
"""

import sys
import os
import torch
from torch import nn
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import DeepSpectralPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback


def train_with_H(target_H, steps=10000, batch_size=4):
    """Train a model with a specific target H."""
    
    # Map H to hidden_dim (empirical relationship)
    # Lower hidden_dim → Lower H
    if target_H < 1.5:
        hidden_dim = 8
    elif target_H < 2.0:
        hidden_dim = 16
    elif target_H < 2.5:
        hidden_dim = 32
    elif target_H < 3.0:
        hidden_dim = 64
    else:
        hidden_dim = 128
    
    print(f"\n{'='*70}")
    print(f"Training with target H≈{target_H} (hidden_dim={hidden_dim})")
    print(f"{'='*70}")
    
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Device: {device}")
    
    # Create prior with specified hidden_dim
    prior = DeepSpectralPrior(hidden_dim=hidden_dim, n_classes=3, device=device)
    
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return prior.get_batch(batch_size, seq_len, n_features, n_classes=3)
    
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=steps,
        batch_size=batch_size,
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
    
    print(f"Model: NanoTabPFN (3 classes)")
    print(f"Prior hidden_dim: {hidden_dim}")
    print(f"Steps: {steps}")
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=criterion,
        epochs=1,
        accumulate_gradients=1,
        lr=1e-4,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name=f"spectral_H{target_H}"
    )
    
    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "multi_h")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"spectral_H{target_H:.1f}.pt")
    torch.save(trained_model.state_dict(), save_path)
    
    print(f"\n✅ Model saved: {save_path}")
    print(f"Final loss: {loss:.4f}")
    
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=float, required=True, help="Target H value (e.g., 1.1 or 3.9)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    train_with_H(args.H, args.steps, args.batch_size)


if __name__ == "__main__":
    main()
