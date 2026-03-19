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
Demo script to generate synthetic tabular data using Spectral Priors.
Usage: python scripts/demo_data_generation.py
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import SpectralStudentTPrior, DeepSpectralPrior

def main():
    print("="*60)
    print("🔮 Spectral Prior Data Generation Demo")
    print("="*60)
    
    device = 'cpu'
    batch_size = 5
    seq_len = 100
    n_features = 10
    
    # 1. Spectral Student-T Prior
    print(f"\n[1] Generating data from Spectral Student-T Prior...")
    print(f"    Parameters: nu=2.0 (Heavy Tail), p=0.2 (Sparsity)")
    
    prior_t = SpectralStudentTPrior(nu=2.0, p=0.2, device=device)
    batch_t = prior_t.get_batch(batch_size, seq_len, n_features)
    
    X_t = batch_t['x'].numpy() # (B, S, F)
    y_t = batch_t['y'].numpy() # (B, S)
    
    print(f"    Generated Batch Shape: {X_t.shape}")
    print(f"    Sample 0, Context 0 info:")
    print(f"      Features: {X_t[0, 0, :3]} ...")
    print(f"      Label: {y_t[0, 0]}")
    
    # Save one sample to CSV
    df_t = pd.DataFrame(X_t[0], columns=[f"feat_{i}" for i in range(n_features)])
    df_t['target'] = y_t[0]
    df_t.to_csv("results/data/demo_spectral_student_t.csv", index=False)
    print("    -> Saved sample to 'results/data/demo_spectral_student_t.csv'")

    # 2. Deep Spectral Prior
    print(f"\n[2] Generating data from Deep Spectral Prior...")
    print(f"    Parameters: hidden_dim=128 (High Entropy)")
    
    prior_d = DeepSpectralPrior(hidden_dim=128, device=device)
    batch_d = prior_d.get_batch(batch_size, seq_len, n_features)
    
    X_d = batch_d['x'].numpy()
    y_d = batch_d['y'].numpy()
    
    print(f"    Generated Batch Shape: {X_d.shape}")
    
    # Save one sample
    df_d = pd.DataFrame(X_d[0], columns=[f"feat_{i}" for i in range(n_features)])
    df_d['target'] = y_d[0]
    df_d.to_csv("results/data/demo_deep_spectral.csv", index=False)
    print("    -> Saved sample to 'results/data/demo_deep_spectral.csv'")
    
    print("\n✅ Demo Complete.\n")

if __name__ == "__main__":
    main()
