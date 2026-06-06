# Utilities Directory

This directory contains the modular helper functions for the VAE repository.

## Modules

1. **`activations.py`**
   - Contains mathematical activation functions and their derivatives.
   - `sigmoid(x)`: Stable sigmoid calculation with clipping to avoid overflow.
   - `sigmoid_derivative(x)`: Analytical derivative of sigmoid.
   - `relu(x)`: Rectified Linear Unit activation.
   - `relu_derivative(x)`: Derivative of ReLU.

2. **`losses.py`**
   - Contains VAE-specific loss calculation.
   - `vae_loss(...)`: Calculates the binary cross-entropy reconstruction loss and the analytical KL divergence loss, with scaling factor $\beta$ for KL annealing.

3. **`data.py`**
   - Handles data loading and acquisition.
   - `load_mnist(...)`: Reads local MNIST idx files.
   - `download_mnist(...)`: Fallback function to automatically download and extract MNIST dataset files from web mirrors if not found locally.

4. **`visualization.py`**
   - Handles plotting and display.
   - `plot_training_curves(...)`: Plots and saves loss curves.
   - `reconstruct_and_display(...)`: Displays and saves original vs reconstructed images.
   - `generate_and_display(...)`: Generates, displays, and saves digits from random noise.
