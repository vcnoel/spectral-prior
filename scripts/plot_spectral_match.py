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
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from spectral_prior import SpectralStudentTPrior
from spectral_prior.utils import get_singular_spectrum

def plot_spectral_match():
    print("Generating Spectral Match Plot...")
    
    # 1. Real Data (Simulated Average)
    # Based on Phase 2 results: gamma=1.29
    # S_k ~ k^(-1.29)
    k = np.arange(1, 21)
    real_spectrum = k ** (-1.29)
    real_spectrum /= real_spectrum.sum()
    
    # 2. Gaussian Baseline
    # Gaussian noise spectrum is roughly flat (Modified Marchenko-Pastur)
    # but for small N it's relatively flat.
    X_gauss = np.random.randn(100, 20)
    gauss_spectrum = get_singular_spectrum(X_gauss)
    
    # 3. Spectral Prior
    # Sample from our prior
    prior = SpectralStudentTPrior(nu=2.0, p=0.2)
    X_spec = prior.sample_t(100, 20).cpu().numpy()
    spec_spectrum = get_singular_spectrum(X_spec)
    
    # Normalize lengths to match (first 20 singular values)
    length = min(len(gauss_spectrum), len(spec_spectrum), 20)
    gauss_spectrum = gauss_spectrum[:length]
    spec_spectrum = spec_spectrum[:length]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), real_spectrum, 'k--', linewidth=2, label='Real Data Geometry ($\gamma \\approx 1.29$)')
    plt.plot(range(1, length+1), gauss_spectrum, 'r-o', label='Gaussian Baseline')
    plt.plot(range(1, length+1), spec_spectrum, 'g-^', label='Spectral Prior (Ours)')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Rank (k)')
    plt.ylabel('Singular Value ($\sigma_k$)')
    plt.title('Spectral Match: Prior vs Real Data')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    plt.savefig('spectral_match.png')
    print("Plot saved to spectral_match.png")

if __name__ == "__main__":
    plot_spectral_match()
