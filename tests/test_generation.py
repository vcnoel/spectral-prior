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
