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
from torch import nn
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataloader import TabICLPriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from tfmplayground.callbacks import ConsoleLoggerCallback

def train_baseline():
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 4
    STEPS = 100
    EPOCHS = 1 # Keep it short for verification
    RUN_NAME = "baseline_mix_scm"

    # Prior Setup (Standard Gaussian Mixture / mix_scm)
    prior = TabICLPriorDataLoader(
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        num_datapoints_min=10,
        num_datapoints_max=50,
        min_features=3,
        max_features=10,
        max_num_classes=3,
        device=device,
        prior_type="mix_scm"
    )

    # Model Setup
    model = NanoTabPFNModel(
        num_attention_heads=4,
        embedding_size=128,
        mlp_hidden_size=512,
        num_layers=4,
        num_outputs=prior.max_num_classes,
    )
    
    criterion = nn.CrossEntropyLoss()

    print("Starting training on dummy batch (batch_size=4)...")
    
    trained_model, loss = train(
        model=model,
        prior=prior,
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
    train_baseline()
