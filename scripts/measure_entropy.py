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
from spectral_prior import SpectralStudentTPrior
from sklearn.preprocessing import StandardScaler

def get_singular_spectrum(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    # Normalize S (as done in compute_ground_truth.py)
    return S / np.sum(S)

def spectral_entropy(spectrum):
    # H = - sum p * log(p)
    return -np.sum(spectrum * np.log(spectrum + 1e-12))

from spectral_prior import SpectralStudentTPrior, DeepSpectralPrior

# ... (utils same) ...

def measure():
    device = torch.device('cpu')
    
    print("=== SpectralStudentTPrior (p sweep) ===")
    ps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    print("p      | H (Entropy)")
    print("-------|------------")
    for p in ps:
        prior = SpectralStudentTPrior(nu=2.0, p=p, device=device)
        hs = []
        for _ in range(3):
            X = prior.sample_t(n_samples=1000, n_features=30)
            spectrum = get_singular_spectrum(X.numpy())
            h = spectral_entropy(spectrum)
            hs.append(h)
        print(f"{p:<6} | {np.mean(hs):.4f}")

    print("\n=== DeepSpectralPrior (Hidden Dim sweep) ===")
    dims = [4, 8, 16, 32, 64, 128, 256, 512]
    print("dim    | H (Entropy)")
    print("-------|------------")
    for d in dims:
        prior = DeepSpectralPrior(hidden_dim=d, device=device)
        hs = []
        for _ in range(3):
            # get_batch returns dictionary with 'x' [batch, seq, feat]
            batch = prior.get_batch(batch_size=1, seq_len=1000, n_features=30)
            X = batch['x'][0] # [1000, 30]
            spectrum = get_singular_spectrum(X.numpy())
            h = spectral_entropy(spectrum)
            hs.append(h)
        print(f"{d:<6} | {np.mean(hs):.4f}")

if __name__ == "__main__":
    measure()
