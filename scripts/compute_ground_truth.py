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
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import scipy.stats

def get_singular_spectrum(X, normalized=True):
    """Compute the singular spectrum of the data matrix X."""
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute SVD
    # X = U S V^T
    # singular values are S
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    if normalized:
        S = S / np.sum(S)
        
    return S

def fit_power_law(spectrum):
    """Fit a power law to the spectrum: sigma_k ~ k^(-gamma).
    Returns gamma.
    """
    # Exclude the first singular value as it's often the mean/dominant component
    # and tail (small values) which might be noise.
    # We fit on the bulk.
    
    # Rank indices (1-based)
    ranks = np.arange(1, len(spectrum) + 1)
    
    # Log-log plot
    log_ranks = np.log(ranks)
    log_spectrum = np.log(spectrum + 1e-12) # Avoid log(0)
    
    # Linear regression: log(y) = -gamma * log(x) + c
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_ranks, log_spectrum)
    
    return -slope

def spectral_entropy(spectrum):
    """Compute the spectral entropy of the normalized spectrum."""
    # Ensure spectrum is generated as a probability distribution (sum=1)
    spectrum_norm = spectrum / np.sum(spectrum)
    
    # Entropy H = - sum p * log(p)
    entropy = -np.sum(spectrum_norm * np.log(spectrum_norm + 1e-12))
    return entropy

def get_fiedler_value(X):
    """Compute Fiedler value (lambda_2) of the k-NN graph Laplacian."""
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse.csgraph
    
    # Construct k-NN graph
    A = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=False)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    
    # Eigenvalues
    try:
        # subset_by_index=[0, 1] gets the first two smallest eigenvalues
        eigvals = scipy.sparse.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=False)
        # lambda_2 is the second one. The first is 0.
        return sorted(eigvals)[1]
    except:
        return 0.0

def main():
    print("Loading reference datasets...")
    
    data_loaders = [
        ("Breast Cancer", datasets.load_breast_cancer),
        ("Wine", datasets.load_wine),
        ("Iris", datasets.load_iris),
        ("Digits", datasets.load_digits),
    ]
    
    # Table Header
    # Dataset | N | d | H | Gamma | Lambda2 | Type
    print(f"{'Dataset':<15} | {'N':<5} | {'d':<4} | {'H_L':<8} | {'Gamma':<8} | {'Lambda2':<8} | {'Type'}")
    print("-" * 80)
    
    tabular_metrics = []
    
    for name, loader in data_loaders:
        try:
            data = loader()
            X = data.data
            N, d = X.shape
            
            spectrum = get_singular_spectrum(X)
            gamma = fit_power_law(spectrum)
            H = spectral_entropy(spectrum)
            lam2 = get_fiedler_value(X)
            
            # Heuristic for type
            m_type = "Tabular" if name != "Digits" else "Spatial"
            
            print(f"{name:<15} | {N:<5} | {d:<4} | {H:.4f}   | {gamma:.4f}   | {lam2:.4f}   | {m_type}")
            
            if m_type == "Tabular":
                tabular_metrics.append((H, gamma, lam2))
                
        except Exception as e:
            print(f"Skipping {name}: {e}")
            
    # Compute Average for Tabular
    if tabular_metrics:
        avg_H = np.mean([x[0] for x in tabular_metrics])
        avg_gamma = np.mean([x[1] for x in tabular_metrics])
        avg_lam2 = np.mean([x[2] for x in tabular_metrics])
        print("-" * 80)
        print(f"{'Mean (Tabular)':<15} | {'-':<5} | {'-':<4} | {avg_H:.4f}   | {avg_gamma:.4f}   | {avg_lam2:.4f}   | {'Manifold'}")

if __name__ == "__main__":
    main()
