
import sys
import os
import numpy as np
import torch

# Add parent directory to path to import spectral_prior
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import SpectralStudentTPrior
from spectral_prior.utils import get_singular_spectrum, fit_power_law, spectral_entropy

def tune_prior():
    target_gamma = 1.29
    target_H = 2.44
    
    # Grid Search
    nu_values = [2.0, 3.0, 5.0, 10.0, 30.0]
    p_values = [0.1, 0.2, 0.3, 0.5, 0.7] # Sparsity
    
    print(f"Target: Gamma={target_gamma}, H={target_H}")
    print(f"{'nu':<5} | {'p':<5} | {'Gamma':<10} | {'H':<10} | {'Score':<10}")
    print("-" * 50)
    
    best_score = float('inf')
    best_params = {}
    
    dataset_size = 100
    n_features = 20
    n_trials = 5
    
    # We want to minimize distance to target gamma/H
    
    for nu in nu_values:
        for p in p_values:
            prior = SpectralStudentTPrior(nu=nu, p=p)
            
            avg_gamma = 0
            avg_H = 0
            
            for _ in range(n_trials):
                # Sample batch
                # Use sample_t directly
                try:
                    X = prior.sample_t(dataset_size, n_features).cpu().numpy()
                    
                    s = get_singular_spectrum(X)
                    gamma = fit_power_law(s)
                    H = spectral_entropy(s)
                    
                    avg_gamma += gamma
                    avg_H += H
                except Exception as e:
                    print(f"Error with nu={nu}, p={p}: {e}")
                    avg_gamma += 10 # Penalty
                    avg_H += 10
            
            avg_gamma /= n_trials
            avg_H /= n_trials
            
            score = abs(avg_gamma - target_gamma) + abs(avg_H - target_H)
            
            print(f"{nu:<5} | {p:<5} | {avg_gamma:.4f}     | {avg_H:.4f}     | {score:.4f}")
            
            if score < best_score:
                best_score = score
                best_params = {'nu': nu, 'p': p}
                
    print("-" * 50)
    print(f"Best Params: {best_params} with Score: {best_score:.4f}")

if __name__ == "__main__":
    tune_prior()
