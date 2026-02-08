"""
Training Script for DeepSpectralPrior with:
- 10 output classes (up from 3)
- Adaptive H based on target dataset entropy
- 50k training steps
- GPU acceleration
"""

import sys
import os
import torch
from torch import nn
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import DeepSpectralPrior, SpectralStudentTPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback


class AdaptiveSpectralPrior:
    """
    Prior that samples H adaptively from empirical distribution.
    Targets: H ~ U(1.5, 3.5) to cover Iris (1.09) to Digits (3.92).
    """
    def __init__(self, n_classes=10, device='cpu', H_range=(1.5, 3.5)):
        self.n_classes = n_classes
        self.device = device
        self.H_range = H_range
        
        # Create base priors with different H targets (controlled by hidden_dim)
        self.priors = {}
        for H_target in [1.5, 2.0, 2.5, 3.0, 3.5]:
            # Map H to hidden_dim (higher H = higher hidden_dim for more complexity)
            hidden_dim = int(16 + (H_target - 1.0) * 16)  # 16-56 range
            self.priors[H_target] = DeepSpectralPrior(
                hidden_dim=hidden_dim,
                n_classes=n_classes,
                device=device
            )
    
    def get_batch(self, batch_size, seq_len, n_features, n_classes=None):
        """Generate batch with random H target."""
        if n_classes is None:
            n_classes = self.n_classes
            
        # Sample H uniformly from range
        H_target = np.random.uniform(self.H_range[0], self.H_range[1])
        
        # Find closest prior
        H_keys = list(self.priors.keys())
        closest_H = min(H_keys, key=lambda x: abs(x - H_target))
        
        # Generate batch
        return self.priors[closest_H].get_batch(batch_size, seq_len, n_features, n_classes)


def train_10class():
    parser = argparse.ArgumentParser(description='Train NanoTabPFN with 10 classes')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive H')
    parser.add_argument('--output', type=str, default='models/10class', help='Output directory')
    args = parser.parse_args()
    
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Using device: {device}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Adaptive H: {args.adaptive}")
    
    # Prior selection
    if args.adaptive:
        print("Initializing AdaptiveSpectralPrior (H ~ U(1.5, 3.5))...")
        prior = AdaptiveSpectralPrior(n_classes=10, device=device)
    else:
        print("Initializing DeepSpectralPrior (H=2.44 fixed)...")
        prior = DeepSpectralPrior(hidden_dim=32, n_classes=10, device=device)
    
    # Wrapper for 10 classes
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return prior.get_batch(batch_size, seq_len, n_features, n_classes=10)
    
    # Data Loader
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=args.steps,
        batch_size=args.batch_size,
        num_datapoints_max=100,  # Larger context for better generalization
        num_features=20,  # More features for complex datasets
        device=device
    )
    
    # Model: NanoTabPFN with 10 output classes
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=10,  # 10 classes!
    )
    
    print(f"Model: NanoTabPFN (10 classes)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=criterion,
        epochs=1,
        accumulate_gradients=1,
        lr=args.lr,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name="spectral_10class"
    )
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    
    if args.adaptive:
        save_path = os.path.join(args.output, "spectral_10class_adaptive.pt")
    else:
        save_path = os.path.join(args.output, "spectral_10class_fixed.pt")
    
    torch.save(trained_model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final loss: {loss:.4f}")


if __name__ == "__main__":
    train_10class()
