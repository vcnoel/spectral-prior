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

"""
======================================================================
TRAINING ENTRY POINT
======================================================================
run with: python train.py

This script trains the best model (Hybrid Mixture) for 10k steps.
For full reproduction, see scripts/advanced_priors.py
======================================================================
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import the training script logic
# We'll just execute the script via python subprocess to ensure clean env, 
# or import main. But imports might be messy if script expects __main__.
# Let's use subprocess for robustness.

import subprocess

def main():
    print("🚀 Starting Training Pipeline...")
    print("Training Best Model: Hybrid Mixture Prior (10k steps)")
    
    cmd = [
        sys.executable, 
        "scripts/advanced_priors.py", 
        "--idea", "2", 
        "--steps", "10000"
    ]
    
    subprocess.check_call(cmd)
    
    print("\n✅ Training Complete. Model saved to models/advanced_priors/")

if __name__ == "__main__":
    main()
