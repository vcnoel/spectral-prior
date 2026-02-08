import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from spectral_prior import SpectralStudentTPrior, DeepSpectralPrior, SpectralDAGPrior

def test_priors():
    print("Testing SpectralStudentTPrior...")
    prior = SpectralStudentTPrior(nu=2.0, p=0.2, device='cpu')
    batch = prior.get_batch(batch_size=2, seq_len=10, n_features=5)
    print("Batch keys:", batch.keys())
    print("x shape:", batch['x'].shape)
    assert batch['x'].shape == (2, 10, 5)
    
    print("\nTesting DeepSpectralPrior...")
    prior = DeepSpectralPrior(hidden_dim=16, device='cpu')
    batch = prior.get_batch(batch_size=2, seq_len=10, n_features=5)
    print("x shape:", batch['x'].shape)
    assert batch['x'].shape == (2, 10, 5)

    print("\nTesting SpectralDAGPrior...")
    prior = SpectralDAGPrior(device='cpu')
    batch = prior.get_batch(batch_size=2, seq_len=10, n_features=5)
    print("x shape:", batch['x'].shape)
    assert batch['x'].shape == (2, 10, 5)
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_priors()
