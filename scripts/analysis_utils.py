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
from sklearn.preprocessing import StandardScaler
import scipy.sparse.linalg
from sklearn.neighbors import kneighbors_graph
import scipy.sparse.csgraph

def get_singular_spectrum(X):
    """Compute normalized singular spectrum."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    return S / np.sum(S)

def fit_power_law(spectrum):
    """Fit power law decay exponent gamma."""
    ranks = np.arange(1, len(spectrum) + 1)
    log_ranks = np.log(ranks)
    log_spectrum = np.log(spectrum + 1e-12)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_ranks, log_spectrum)
    return -slope

def spectral_entropy(spectrum):
    """Compute spectral entropy."""
    return -np.sum(spectrum * np.log(spectrum + 1e-12))

def get_fiedler_value(X):
    """Compute Fiedler value (lambda_2) of 5-NN graph."""
    A = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=False)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    try:
        eigvals = scipy.sparse.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=False)
        return sorted(eigvals)[1]
    except:
        return 0.0
