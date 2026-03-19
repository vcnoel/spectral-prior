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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import PriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback

def gaussian_batch(batch_size, seq_len, n_features):
    device = get_default_device()
    
    # Simple Gaussian + Random Linear Classification
    x = torch.randn(batch_size, seq_len, n_features, device=device)
    W = torch.randn(n_features, 3, device=device)
    y = torch.argmax(x @ W, dim=-1).float()
    
    single_eval_pos = seq_len // 2
    
    return dict(
        x=x,
        y=y,
        target_y=y,
        single_eval_pos=single_eval_pos
    )

def train_baseline_gaussian():
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 4
    STEPS = 100
    EPOCHS = 1
    RUN_NAME = "baseline_gaussian"

    loader = PriorDataLoader(
        get_batch_function=gaussian_batch,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        num_datapoints_max=50,
        num_features=10,
        device=device
    )

    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=3,
    )
    
    criterion = nn.CrossEntropyLoss()

    print("Starting training with Gaussian Prior...")
    
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
    
    save_path = f"{RUN_NAME}.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_baseline_gaussian()
