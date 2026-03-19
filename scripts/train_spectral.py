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

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from spectral_prior import SpectralStudentTPrior
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader # Generic loader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback

def train_spectral():
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Using device: {device}")

    # Optimized Parameters from Phase 3
    NU = 2.0
    P = 0.2
    
    print(f"Initializing SpectralStudentTPrior with nu={NU}, p={P}...")
    spectral_prior = SpectralStudentTPrior(nu=NU, p=P, device=device)

    # Wrapper for PriorDataLoader
    def get_batch_wrapper(batch_size, seq_len, n_features):
        return spectral_prior.get_batch(batch_size, seq_len, n_features)

    # Hyperparameters
    BATCH_SIZE = 4
    STEPS = 100
    EPOCHS = 1 # Keep it short for demo/verification
    RUN_NAME = "spectral_student_t"

    # Prior Loader
    # The PriorDataLoader generates infinite synthetic data on the fly.
    # We define 'num_steps' as the number of batches per epoch.
    # This differs from standard static datasets where epoch length is fixed.
    loader = PriorDataLoader(
        get_batch_function=get_batch_wrapper,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        num_datapoints_max=50,
        num_features=10,
        device=device
    )

    # Model Setup
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3, # Fixed 3 classes in prior generation
    )
    
    criterion = nn.CrossEntropyLoss()

    print("Starting training with Spectral Prior...")
    
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
    
    # Save Model
    save_path = f"{RUN_NAME}.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_spectral()
