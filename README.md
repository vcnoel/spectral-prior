# Spectral Prior: Geometric Alignment for Tabular Data Generation

## Abstract

This repository contains the implementation of **Spectral Priors**, a method for generating synthetic tabular data with controllable spectral properties. We demonstrate that by aligning the spectral entropy ($H \approx 2.44$) and covariance structure of the prior to the intrinsic geometry of tabular data, a small **1.1M parameter Transformer** (NanoTabPFN) can achieve performance competitive with or exceeding commercial State-of-the-Art (NeuralK) on standard benchmarks.

This work investigates the "Geometric Goldilocks Zone"—the specific range of spectral complexity characteristic of organic tabular data—and proposes methods to synthesize data within this zone for effective In-Context Learning (ICL).

## Key Findings

We evaluated the model on the TabArena benchmark against a commercial SOTA baseline (NeuralK). Despite being significantly smaller and faster to train, our model demonstrates robust generalization.

| Metric | NeuralK (Commercial) | NanoTabPFN (Ours) | Difference |
|---|---|---|---|
| **Parameters** | >100M (Est.) | **1.1M** | -99% |
| **Training Time** | High Compute Cluster | **15 Minutes (1 GPU)** | Efficient |
| **Blood Transfusion** | 76.89% | **77.51%** | **+0.62%** |
| **Diabetes** | 76.80% | **77.14%** | **+0.34%** |
| **Australian** | 87.05% | **87.92%** | **+0.87%** |
| **Wine** | 100.00% | **100.00%** | = |
| **Breast Cancer** | 98.83% | **98.83%** | = |

*Note: Results report Mean accuracy over 5 random seeds.*

## Overview

This package provides data generators ("priors") that create synthetic datasets where the covariance structure of features is derived from random graph Laplacians or spectrally-initialized neural networks.

### Key Components

*   **`SpectralStudentTPrior`**: Generates data from a multivariate t-distribution with a covariance matrix constructed from a random graph Laplacian. This allows control over the "sharpness" (`nu`) and "connectivity" (`p`) of the feature correlations.
*   **`DeepSpectralPrior`**: Generates data by passing random noise through a non-linear neural network initialized with orthogonal weights, allowing control over spectral entropy via the network width (`hidden_dim`).
*   **`SpectralDAGPrior`**: Generates data based on a Structural Equation Model (SEM) over a random Directed Acyclic Graph (DAG), simulating causal relationships.

## Installation

It is recommended to use the `spectral-prior` conda environment.

```bash
# Option 1: Create from environment.yaml (Recommended)
conda env create -f environment.yaml
conda activate spectral-prior
pip install -e .

# Option 2: Create manually
conda create -n spectral-prior python=3.10
conda activate spectral-prior
pip install -r requirements.txt
pip install -e .
```

### Dependencies
*   `torch`
*   `numpy<2.0.0` (Required for binary compatibility)
*   `scikit-learn` (for evaluation)
*   `pandas` (for evaluation)
*   `tfmplayground` (included)

## Reproduction Instructions

### 1. Generate Synthetic Data
Run the demo script to generate random batches from our Spectral Priors and save them to CSV:

```bash
python scripts/demo_data_generation.py
```
*Outputs: `results/data/demo_spectral_student_t.csv`, `results/data/demo_deep_spectral.csv`*

### 2. Run the Benchmark
To reproduce the 5-seed statistics (Mean ± Std) reported in the Key Findings:

```bash
python scripts/tabarena_deep_benchmark.py
```
*Outputs: `results/logs/rigorous_benchmark.txt` with the full comparison table.*

### 3. Interactive Evaluation
To interactively reproduce the results on specific datasets (e.g., Blood Transfusion):
1.  Open `notebooks/reproduce_results.ipynb`
2.  Run all cells.

## Usage

### 1. Generating Data

You can use the priors directly to generate batches of synthetic data.

```python
from spectral_prior import SpectralStudentTPrior

# Initialize the prior
# nu: Degrees of freedom (lower = heavier tails)
# p: Probability of edge connection in the underlying graph (controls sparsity)
prior = SpectralStudentTPrior(nu=2.0, p=0.2, device='cpu')

# Generate a batch
# Returns a dictionary with 'x', 'y', 'target_y', 'single_eval_pos'
batch_size = 4
seq_len = 50
n_features = 10

batch = prior.get_batch(batch_size, seq_len, n_features)

print("Data shape:", batch['x'].shape) # (4, 50, 10)
print("Labels shape:", batch['y'].shape) # (4, 50)
```

### 2. Using Deep Spectral Prior

```python
from spectral_prior import DeepSpectralPrior

# hidden_dim controls the spectral entropy of the generated data
# typically 128 for high entropy, smaller values for lower entropy
prior = DeepSpectralPrior(hidden_dim=128, device='cpu')

batch = prior.get_batch(batch_size=4, seq_len=50, n_features=10)
```

## Parameters

### `SpectralStudentTPrior`
*   **`nu` (float, default=3.0)**: Degrees of freedom for the Student-t distribution.
    *   `nu` -> $\infty$ approaches a Gaussian distribution.
    *   Small `nu` (e.g., 2.0) creates heavy-tailed distributions.
*   **`p` (float, default=0.3)**: Probability of edge creation in the Erdos-Renyi graph used to build the Laplacian covariance matrix.
    *   Controls the sparsity and structure of feature correlations.
*   **`entropy` (float)**: Target spectral entropy (informational, not used in generation logic).

### `DeepSpectralPrior`
*   **`hidden_dim` (int, default=128)**: The width of the hidden layers in the generating network.
    *   Controls the rank/entropy of the output data manifold.

## Evaluation Scripts

The `scripts/` directory contains various experiments and evaluations:

*   **`scripts/train_spectral.py`**: Trains a NanoTabPFN model using the `SpectralStudentTPrior`.
*   **`scripts/ablation_study.py`**: Performs an ablation study on `nu` (latent distribution) and `hidden_dim` (spectral entropy), evaluating the trained model on the Wine dataset.
*   **`scripts/stress_test.py`**: Compares a model trained on `SpectralStudentTPrior` vs a Gaussian baseline on "disconnected" (stressed) data.
*   **`scripts/plot_spectral_match.py`**: Generates a plot comparing the singular value spectrum of the prior against real data geometry (saved to `results/figures/spectral_match.png`).

To run a script (ensure `tfmplayground` and `spectral_prior` are installed):

```bash
python scripts/train_spectral.py
```
