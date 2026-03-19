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

import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler

def get_singular_spectrum(X, normalized=True):
    """Compute the singular spectrum of the data matrix X."""
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute SVD
    # X = U S V^T
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    if normalized:
        S = S / np.sum(S)
        
    return S

def fit_power_law(spectrum):
    """Fit a power law to the spectrum: sigma_k ~ k^(-gamma)."""
    ranks = np.arange(1, len(spectrum) + 1)
    
    log_ranks = np.log(ranks)
    log_spectrum = np.log(spectrum + 1e-12)
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_ranks, log_spectrum)
    
    return -slope

def spectral_entropy(spectrum):
    """Compute the spectral entropy."""
    spectrum_norm = spectrum / np.sum(spectrum)
    entropy = -np.sum(spectrum_norm * np.log(spectrum_norm + 1e-12))
    return entropy
