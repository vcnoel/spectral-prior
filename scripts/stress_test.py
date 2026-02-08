
import sys
import os
import torch
import numpy as np
from torch import nn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import SpectralStudentTPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed

def get_disconnected_covariance(n_features, device):
    """Generate a covariance matrix for a disconnected graph (2 components)."""
    # Simply block diagonal 
    n1 = n_features // 2
    n2 = n_features - n1
    
    # Create two random Laplacians
    def make_lap(n):
        A = torch.rand(n, n, device=device) < 0.5
        A = A.float()
        A = torch.triu(A, 1) + torch.triu(A, 1).T
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        return L

    L1 = make_lap(n1)
    L2 = make_lap(n2)
    
    # Block diagonal L
    L = torch.zeros(n_features, n_features, device=device)
    L[:n1, :n1] = L1
    L[n1:, n1:] = L2
    
    # Pinv
    Sigma = torch.linalg.pinv(L, rcond=1e-5)
    return Sigma

def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        x = data['x'].to(device)
        y = data['y'].to(device)
        target_y = data['target_y'].to(device)
        single_eval_pos = data['single_eval_pos']
        
        # Output: [batch, num_targets, num_classes]
        # Targets: [batch, num_targets] (Long)
        output = model((x, y[:, :single_eval_pos]), single_eval_pos=single_eval_pos)
        
        # Targets for eval part (after single_eval_pos)
        targets = target_y[:, single_eval_pos:].long()
        
        # Predictions
        preds = torch.argmax(output, dim=-1)
        
        acc = (preds == targets).float().mean().item()
    return acc

def stress_test():
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Load Models
    print("Loading models...")
    
    # 1. Spectral Model
    spectral_model = NanoTabPFNModel(128, 4, 512, 4, 3).to(device)
    spectral_ckpt = torch.load("spectral_student_t.pt", map_location=device)
    spectral_model.load_state_dict(spectral_ckpt)
    
    # 2. Gaussian Baseline
    gaussian_model = NanoTabPFNModel(128, 4, 512, 4, 3).to(device)
    gaussian_ckpt = torch.load("baseline_gaussian.pt", map_location=device)
    gaussian_model.load_state_dict(gaussian_ckpt)

    # Test Setup
    n_features = 10
    batch_size = 100
    seq_len = 50
    
    # 1. Normal (Connected) Performance
    print("\nEvaluating on Normal (Connected) Data...")
    normal_prior = SpectralStudentTPrior(nu=2.0, p=0.2, device=device)
    normal_batch = normal_prior.get_batch(batch_size, seq_len, n_features)
    # Fix float/long issue if needed (simulated)
    
    acc_spec_norm = evaluate_model(spectral_model, normal_batch, device)
    acc_gauss_norm = evaluate_model(gaussian_model, normal_batch, device)
    
    print(f"Spectral Model: {acc_spec_norm:.4f}")
    print(f"Gaussian Model: {acc_gauss_norm:.4f}")
    
    # 2. Disconnected (Stressed) Performance
    print("\nEvaluating on Disconnected (Stressed) Data...")
    
    # Manually generate batch with disconnected covariance
    # We subclass or monkeypatch?
    # Or just use the function I wrote above.
    
    stressed_X_list = []
    stressed_y_list = []
    
    for _ in range(batch_size):
        Sigma = get_disconnected_covariance(n_features, device)
        
        # Sample t with this Sigma
        # Copy-paste sampling logic for brevity or import?
        # I'll re-implement sampling here slightly
        mean = torch.zeros(n_features, device=device)
        Sigma = Sigma + 1e-6 * torch.eye(n_features, device=device)
        Sigma = (Sigma + Sigma.T) / 2
        e, v = torch.linalg.eigh(Sigma)
        e[e < 1e-6] = 1e-6
        Sigma = v @ torch.diag(e) @ v.T
        
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma)
        y_samp = mvn.sample((seq_len,))
        u = torch.distributions.Chi2(torch.tensor(2.0, device=device)).sample((seq_len,))
        scale = torch.sqrt(2.0 / u).unsqueeze(1)
        x = mean + y_samp * scale
        
        # Labels
        W = torch.randn(n_features, 3, device=device)
        logits = x @ W
        y = torch.argmax(logits, dim=1).float()
        
        stressed_X_list.append(x)
        stressed_y_list.append(y)
        
    stressed_data = {
        'x': torch.stack(stressed_X_list),
        'y': torch.stack(stressed_y_list),
        'target_y': torch.stack(stressed_y_list),
        'single_eval_pos': seq_len // 2
    }
    
    acc_spec_stress = evaluate_model(spectral_model, stressed_data, device)
    acc_gauss_stress = evaluate_model(gaussian_model, stressed_data, device)
    
    print(f"Spectral Model: {acc_spec_stress:.4f}")
    print(f"Gaussian Model: {acc_gauss_stress:.4f}")
    
    # Deltas
    delta_spec = acc_spec_norm - acc_spec_stress
    delta_gauss = acc_gauss_norm - acc_gauss_stress
    
    print("-" * 30)
    print(f"Spectral Delta: {delta_spec:.4f}")
    print(f"Gaussian Delta: {delta_gauss:.4f}")
    
    if delta_spec < delta_gauss:
        print("SUCCESS: Spectral Model is more robust.")
    else:
        print("FAILURE: Gaussian Model is more robust.")

if __name__ == "__main__":
    stress_test()
