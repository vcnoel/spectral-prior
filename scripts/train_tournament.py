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
from torch import nn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback

from spectral_prior import SpectralStudentTPrior, DeepSpectralManifoldPrior, SpectralDAGPrior

def train_prior(name, prior_instance, steps=10000, device='cuda'):
    print(f"\n=== Training {name} Prior for {steps} steps on {device} ===")
    
    # Hyperparameters
    BATCH_SIZE = 4 # Small batch for speed/memory on consumer GPU
    EPOCHS = 1
    RUN_NAME = f"tournament_{name}"

    # Wrapper for PriorDataLoader
    # It expects (batch_size, seq_len, num_features)
    # Our priors match this signature.
    
    loader = PriorDataLoader(
        get_batch_function=prior_instance.get_batch,
        num_steps=steps,
        batch_size=BATCH_SIZE,
        num_datapoints_max=50,
        num_features=10,
        device=device
    )

    # Standard NanoTabPFN config
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    trained_model, loss = train(
        model=model,
        prior=loader,
        criterion=criterion,
        epochs=EPOCHS,
        accumulate_gradients=1,
        lr=1e-4,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name=RUN_NAME
    )
    
    save_path = f"prior_{name.lower()}.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"saved to {save_path}")
    return save_path

def run_tournament():
    set_randomness_seed(42)
    # Force CUDA if available, else CPU (but user said 4080 so CUDA should be there)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: CUDA not found. Training will be slow.")

    # 1. The "Robust" Candidate: SpectralStudentT
    # nu=2.0, p=0.2 (from Phase 3 Tuning)
    robust_prior = SpectralStudentTPrior(nu=2.0, p=0.2, device=device)
    train_prior("Robust_StudentT", robust_prior, steps=10000, device=device)
    
    # 2. The "Complex" Candidate: DeepSpectralManifold
    manifold_prior = DeepSpectralManifoldPrior(device=device)
    train_prior("Complex_Manifold", manifold_prior, steps=10000, device=device)
    
    # 3. The "Causal" Candidate: SpectralDAGPrior
    dag_prior = SpectralDAGPrior(device=device)
    train_prior("Causal_DAG", dag_prior, steps=10000, device=device)
    
    print("\nTournament Training Complete. All contenders ready.")

if __name__ == "__main__":
    run_tournament()
