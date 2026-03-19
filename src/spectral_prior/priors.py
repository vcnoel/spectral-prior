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

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.chi2 import Chi2
from typing import Dict, List, Optional, Tuple, Union

class SpectralStudentTPrior:
    """
    A prior that generates data from a Student-t distribution with a covariance matrix
    derived from a random graph Laplacian. This encourages spectral properties in the
    generated data.
    """
    def __init__(self, nu: float = 3.0, gamma: float = 1.29, entropy: float = 2.44, p: float = 0.3, device: str = 'cpu'):
        """
        Initialize the SpectralStudentTPrior.

        Args:
            nu (float): Degrees of freedom for the Student-t distribution. Lower values mean heavier tails.
            gamma (float): Parameter for spectral distribution shape (not currently used in generation logic explicitly, but part of config).
            entropy (float): Target spectral entropy (not currently used in generation logic explicitly).
            p (float): Probability of edge connection in the random graph used for Laplacian covariance.
            device (str): Device to run calculations on ('cpu' or 'cuda').
        """
        self.nu = nu
        self.gamma = gamma
        self.entropy = entropy
        self.p = p
        self.device = device

    def generate_laplacian_covariance(self, n_features: int) -> torch.Tensor:
        """
        Generate a covariance matrix from a random graph Laplacian.
        
        Args:
            n_features (int): Number of features (nodes in the graph).
            
        Returns:
            torch.Tensor: Covariance matrix of shape (n_features, n_features).
        """
        # Random Erdos-Renyi-like graph adjacency
        # Adjust p to control sparsity/spectral properties
        p = self.p
        
        # Generate random adjacency (upper triangle)
        # Using torch for GPU support if needed
        tril_indices = torch.tril_indices(row=n_features, col=n_features, offset=-1, device=self.device)
        rand_vals = torch.rand(tril_indices.shape[1], device=self.device)
        edges = rand_vals < p
        
        rows = tril_indices[0, edges]
        cols = tril_indices[1, edges]
        
        # Build Adjacency matrix
        A = torch.zeros((n_features, n_features), device=self.device)
        A[rows, cols] = 1.0
        A[cols, rows] = 1.0 # Symmetric
        
        # Laplacian
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        
        # Pseudo-inverse
        # Use dimensions to stabilize
        # Add small regularization to avoid singular
        # Sigma = pinv(L)
        Sigma = torch.linalg.pinv(L, rcond=1e-5)
        
        # Normalize Sigma?
        # Ensure positive definite (it is because L is PSD, so pinv is PSD)
        return Sigma

    def sample_t(self, n_samples: int, n_features: int) -> torch.Tensor:
        """
        Sample from Multivariate t-distribution using the Laplacian covariance.
        
        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features.
            
        Returns:
            torch.Tensor: Generated samples of shape (n_samples, n_features).
        """
        Sigma = self.generate_laplacian_covariance(n_features)
        
        mean = torch.zeros(n_features, device=self.device)
        
        # Ensure Positive Definiteness robustly
        Sigma = Sigma + 1e-6 * torch.eye(n_features, device=self.device)
        Sigma = (Sigma + Sigma.T) / 2
        eigval, eigvec = torch.linalg.eigh(Sigma)
        eigval[eigval < 1e-6] = 1e-6
        Sigma = eigvec @ torch.diag(eigval) @ eigvec.T
        
        mvn = MultivariateNormal(mean, covariance_matrix=Sigma)
        y = mvn.sample((n_samples,))
        
        # Chi2 sample
        # nu degrees of freedom
        u = Chi2(torch.tensor(self.nu, device=self.device)).sample((n_samples,))
        
        # Scale: x = mean + y * sqrt(nu / u)
        # u is [n_samples], y is [n_samples, n_features]
        scale = torch.sqrt(self.nu / u).unsqueeze(1)
        x = mean + y * scale
        
        return x

    def mixed_type_injection(self, x: torch.Tensor, discrete_fraction: float = 0.3) -> torch.Tensor:
        """
        Discretize a fraction of columns to simulate mixed-type data.
        
        Args:
            x (torch.Tensor): Input data.
            discrete_fraction (float): Fraction of columns to discretize.
            
        Returns:
            torch.Tensor: Data with some columns discretized.
        """
        n_samples, n_features = x.shape
        n_discrete = int(n_features * discrete_fraction)
        
        if n_discrete > 0:
            # Randomly select columns
            indices = torch.randperm(n_features)[:n_discrete]
            
            for idx in indices:
                # Quantile binning simulation
                # Number of bins random between 2 and 10
                n_bins = torch.randint(2, 11, (1,)).item()
                
                col = x[:, idx]
                col_min = col.min()
                col_max = col.max()
                
                if col_max > col_min:
                    # Normalize to 0-1
                    col_norm = (col - col_min) / (col_max - col_min)
                    # Discretize
                    x[:, idx] = torch.floor(col_norm * n_bins).clamp(0, n_bins - 1)
        
        return x

    def get_batch(self, batch_size: int, seq_len: int, n_features: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Generate a batch of datasets using the Spectral Student-t Prior.
        Returns dictionary compatible with TFM-Playground PriorDataLoader.
        
        Args:
            batch_size (int): Number of datasets in the batch.
            seq_len (int): Number of samples per dataset (sequence length).
            n_features (int): Number of features per sample.
            
        Returns:
            dict: Dictionary containing 'x', 'y', 'target_y', 'single_eval_pos'.
        """
        X_list = []
        y_list = []
        
        for _ in range(batch_size):
            # Sample features using Student-t and Laplacian Covariance
            features = self.sample_t(seq_len, n_features)
            
            # Apply mixed type injection
            features = self.mixed_type_injection(features)
            
            # Generate synthetic labels using a random Teacher MLP
            # Simple linear + nonlinearity or just linear
            # Weights from Normal(0, 1)
            W = torch.randn(n_features, 3, device=self.device) # 3 classes
            logits = features @ W
            y = torch.argmax(logits, dim=1)
            
            X_list.append(features)
            y_list.append(y)
            
        x = torch.stack(X_list)
        y = torch.stack(y_list)
        
        # single_eval_pos is usually random or fixed. TabPFN uses standard split.
        single_eval_pos = seq_len // 2 
        
        return dict(
            x=x,
            y=y.float(),
            target_y=y.float(),
            single_eval_pos=single_eval_pos
        )

class DeepSpectralPrior(torch.nn.Module):
    """
    A deep prior that generates data by passing random noise through a neural network
    initialized with orthogonal weights (spectral initialization).
    """
    def __init__(self, hidden_dim: int = 128, n_classes: int = 3, device: str = 'cpu'):
        """
        Initialize the DeepSpectralPrior.
        
        Args:
            hidden_dim (int): Hidden dimension of the generating network, which controls the spectral entropy.
            n_classes (int): Number of output classes.
            device (str): Device to run on.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.device = device
        
    def get_net(self, n_features: int, hidden_dim: int = 128) -> torch.nn.Sequential:
        """
        Create a random neural network with spectral initialization.
        """
        net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_features)
        ).to(self.device)
        
        # Spectral Initialization
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
        return net

    def get_batch(self, batch_size: int, seq_len: int, n_features: int, n_classes: int = None) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Generate a batch of datasets.
        
        Args:
            batch_size (int): Number of datasets.
            seq_len (int): Samples per dataset.
            n_features (int): Number of features.
            n_classes (int): Number of classes (optional, defaults to self.n_classes).
        """
        if n_classes is None:
            n_classes = self.n_classes
            
        X_list = []
        y_list = []
        
        for _ in range(batch_size):
            # Dynamic network per sample or shared? 
            # TabPFN usually assumes each dataset is from a different distribution/mechanism.
            # So we re-init the network for each dataset in the batch.
            net = self.get_net(n_features, hidden_dim=self.hidden_dim)
            
            # Latent z
            z = torch.randn(seq_len, n_features, device=self.device)
            
            # Wrap in no_grad because this is data generation
            with torch.no_grad():
                x = net(z)
                
                # Label generation: Random Hyperplane in warped space
                w = torch.randn(n_features, n_classes, device=self.device)
                logits = x @ w
                y = torch.argmax(logits, dim=1)
            
            X_list.append(x)
            y_list.append(y)
            
        x = torch.stack(X_list)
        y = torch.stack(y_list)
        single_eval_pos = seq_len // 2
        
        return dict(
            x=x,
            y=y.float(),
            target_y=y.float(),
            single_eval_pos=single_eval_pos
        )

class SpectralDAGPrior:
    """
    A prior that generates data using a structural equation model (SEM) over a random DAG.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def get_batch(self, batch_size: int, seq_len: int, n_features: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Generate a batch of datasets.
        """
        X_list = []
        y_list = []
        
        for _ in range(batch_size):
            # 1. Random DAG (Upper Triangular Adjacency)
            # Ensure modularity (clusters)
            # Simple approach: Block diagonal-ish
            adj = torch.triu(torch.rand(n_features, n_features, device=self.device) > 0.7).float()
            
            # 2. SEM Generation
            data = torch.randn(seq_len, n_features, device=self.device)
            
            for i in range(n_features):
                parents = adj[:, i].nonzero()
                if len(parents) > 0:
                    # Signal from parents
                    signal = data[:, parents[:, 0]].sum(dim=1)
                    # Non-linearity
                    data[:, i] += torch.tanh(signal)
            
            # 3. Label: Function of 'sink' nodes or all nodes
            # Sink nodes: cols with all zeros in adj (no outgoing edges)? 
            # Or rows with all zeros (no incoming)?
            # Adj is A[i,j] = 1 means i -> j.
            # Sink: node i st sum(A[i, :]) == 0.
            # Let's use random projection of data
            W = torch.randn(n_features, 3, device=self.device)
            logits = data @ W
            y = torch.argmax(logits, dim=1)
            
            X_list.append(data)
            y_list.append(y)
            
        x = torch.stack(X_list)
        y = torch.stack(y_list)
        single_eval_pos = seq_len // 2
        
        return dict(
            x=x,
            y=y.float(),
            target_y=y.float(),
            single_eval_pos=single_eval_pos
        )
