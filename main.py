import numpy as np
import os
from utils.data import load_mnist
from utils.activations import sigmoid, sigmoid_derivative, relu, relu_derivative
from utils.losses import vae_loss
from utils.visualization import plot_training_curves, generate_and_display, reconstruct_and_display

# ============================================
# VAE CLASS (with gradient clipping)
# ============================================

class VAE:
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=20):
        # Xavier/Glorot initialization
        self.W_e1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b_e1 = np.zeros((hidden_dim, 1))
        
        self.W_mu = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_mu = np.zeros((latent_dim, 1))
        
        self.W_logvar = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_logvar = np.zeros((latent_dim, 1))
        
        # DECODER
        self.W_d1 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b_d1 = np.zeros((hidden_dim, 1))
        
        self.W_out = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros((input_dim, 1))
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.gradient_norms = []
    
    def encode(self, x):
        self.x = x
        self.z1 = self.W_e1 @ x + self.b_e1
        self.h1 = relu(self.z1)
        self.mu = self.W_mu @ self.h1 + self.b_mu
        self.log_var = self.W_logvar @ self.h1 + self.b_logvar
        
        # Clip log_var to prevent extreme values
        self.log_var = np.clip(self.log_var, -10, 10)
        
        return self.mu, self.log_var
    
    def reparameterize(self, mu, log_var):
        self.mu = mu
        self.log_var = log_var
        self.std = np.exp(0.5 * log_var)
        self.eps = np.random.randn(*mu.shape)
        self.z = mu + self.std * self.eps
        return self.z
    
    def decode(self, z):
        self.z = z
        self.z2 = self.W_d1 @ z + self.b_d1
        self.h2 = relu(self.z2)
        self.z3 = self.W_out @ self.h2 + self.b_out
        self.x_recon = sigmoid(self.z3)
        return self.x_recon
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def backward(self, x, x_recon, mu, log_var, learning_rate, beta=1.0):
        # BCE reconstruction loss + Sigmoid gradient simplifies to:
        dL_dz3 = x_recon - x
        
        # Decoder output layer
        dL_dW_out = dL_dz3 @ self.h2.T
        dL_db_out = np.sum(dL_dz3, axis=1, keepdims=True)
        dL_dh2 = self.W_out.T @ dL_dz3
        
        # Decoder hidden layer
        dL_dz2 = dL_dh2 * relu_derivative(self.z2)
        dL_dz2 = np.clip(dL_dz2, -10, 10)
        
        dL_dW_d1 = dL_dz2 @ self.z.T
        dL_db_d1 = np.sum(dL_dz2, axis=1, keepdims=True)
        dL_dz = self.W_d1.T @ dL_dz2
        
        # Reparameterization gradients flow from reconstruction loss
        dL_dmu_reparam = dL_dz
        dL_dlog_var_reparam = dL_dz * self.eps * 0.5 * self.std
        dL_dlog_var_reparam = np.clip(dL_dlog_var_reparam, -10, 10)
        
        # KL divergence gradients
        # KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))
        # dKL/dmu = mu (fixed sign from original -mu)
        dKL_dmu = mu
        # dKL/dlog_var = -0.5 * (1 - exp(log_var))
        dKL_dlog_var = -0.5 * (1.0 - np.exp(np.clip(log_var, -10, 10)))
        
        # Total gradients with beta annealing factor
        dL_dmu_total = dL_dmu_reparam + beta * dKL_dmu
        dL_dlog_var_total = dL_dlog_var_reparam + beta * dKL_dlog_var
        
        # Encoder mu layer
        dL_dW_mu = dL_dmu_total @ self.h1.T
        dL_db_mu = np.sum(dL_dmu_total, axis=1, keepdims=True)
        dL_dh1_mu = self.W_mu.T @ dL_dmu_total
        
        # Encoder log_var layer
        dL_dW_logvar = dL_dlog_var_total @ self.h1.T
        dL_db_logvar = np.sum(dL_dlog_var_total, axis=1, keepdims=True)
        dL_dh1_logvar = self.W_logvar.T @ dL_dlog_var_total
        
        # Encoder hidden layer
        dL_dh1 = dL_dh1_mu + dL_dh1_logvar
        dL_dh1 = np.clip(dL_dh1, -10, 10)
        
        dL_dz1 = dL_dh1 * relu_derivative(self.z1)
        dL_dW_e1 = dL_dz1 @ self.x.T
        dL_db_e1 = np.sum(dL_dz1, axis=1, keepdims=True)
        
        # Gradient clipping to prevent exploding gradients
        grad_max = 5.0
        dL_dW_out = np.clip(dL_dW_out, -grad_max, grad_max)
        dL_dW_d1 = np.clip(dL_dW_d1, -grad_max, grad_max)
        dL_dW_mu = np.clip(dL_dW_mu, -grad_max, grad_max)
        dL_dW_logvar = np.clip(dL_dW_logvar, -grad_max, grad_max)
        dL_dW_e1 = np.clip(dL_dW_e1, -grad_max, grad_max)
        
        # Update weights
        self.W_out -= learning_rate * dL_dW_out
        self.b_out -= learning_rate * dL_db_out
        self.W_d1 -= learning_rate * dL_dW_d1
        self.b_d1 -= learning_rate * dL_db_d1
        self.W_mu -= learning_rate * dL_dW_mu
        self.b_mu -= learning_rate * dL_db_mu
        self.W_logvar -= learning_rate * dL_dW_logvar
        self.b_logvar -= learning_rate * dL_db_logvar
        self.W_e1 -= learning_rate * dL_dW_e1
        self.b_e1 -= learning_rate * dL_db_e1

# ============================================
# TRAINING FUNCTION
# ============================================

def train_vae(vae, X_images, epochs=10, batch_size=64, learning_rate=0.0005, warmup_ratio=0.5):
    n_samples = len(X_images)
    n_batches = n_samples // batch_size
    total_steps = epochs * n_batches
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    step = 0
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in range(n_batches):
            # Calculate linear annealing factor beta for the current step
            # beta increases from 0.0 to 1.0 during the first warmup_ratio of steps
            warmup_steps = int(total_steps * warmup_ratio)
            if warmup_steps > 0:
                beta = min(1.0, step / warmup_steps)
            else:
                beta = 1.0
                
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_loss = 0
            batch_recon = 0
            batch_kl = 0
            
            for idx in batch_indices:
                x = X_images[idx].reshape(784, 1)
                
                # Forward pass
                x_recon, mu, log_var = vae.forward(x)
                
                # Calculate loss
                loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, log_var, beta=beta)
                batch_loss += loss
                batch_recon += recon_loss
                batch_kl += kl_loss
                
                # Backward pass
                vae.backward(x, x_recon, mu, log_var, learning_rate, beta=beta)
            
            avg_batch_loss = batch_loss / batch_size
            avg_batch_recon = batch_recon / batch_size
            avg_batch_kl = batch_kl / batch_size
            
            epoch_loss += avg_batch_loss
            epoch_recon += avg_batch_recon
            epoch_kl += avg_batch_kl
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{n_batches}: Loss={avg_batch_loss:.2f}, Recon={avg_batch_recon:.2f}, KL={avg_batch_kl:.2f}, Beta={beta:.4f}")
            
            step += 1
        
        avg_epoch_loss = epoch_loss / n_batches
        avg_epoch_recon = epoch_recon / n_batches
        avg_epoch_kl = epoch_kl / n_batches
        
        losses.append(avg_epoch_loss)
        recon_losses.append(avg_epoch_recon)
        kl_losses.append(avg_epoch_kl)
        
        print(f"  >>> Epoch {epoch+1} Avg - Loss: {avg_epoch_loss:.2f}, Recon: {avg_epoch_recon:.2f}, KL: {avg_epoch_kl:.2f}")
        print()
    
    return losses, recon_losses, kl_losses

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("="*60)
    print("VAE FROM SCRATCH - STABLE VERSION")
    print("="*60)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading MNIST data...")
    # Try loading from Kaggle path first, falls back to local data/mnist
    X, y = load_mnist("/kaggle/input/datasets/hojjatk/mnist-dataset")
    
    # Normalize to 0-1
    X = X / 255.0
    
    # Use smaller subset for faster testing (increase for better results)
    n_samples = 2000
    X = X[:n_samples]
    y = y[:n_samples]
    
    print(f"\nUsing {len(X)} images for training")
    print(f"Image values range: {X.min():.2f} to {X.max():.2f}")
    print()
    
    # Step 2: Create VAE
    print("Step 2: Creating VAE...")
    vae = VAE(input_dim=784, hidden_dim=128, latent_dim=20)
    print(f"  Input dimension: {vae.input_dim}")
    print(f"  Hidden dimension: {vae.hidden_dim}")
    print(f"  Latent dimension: {vae.latent_dim}")
    print()
    
    # Step 3: Train VAE
    print("Step 3: Training VAE...")
    print("-" * 50)
    losses, recon_losses, kl_losses = train_vae(
        vae, 
        X, 
        epochs=5,
        batch_size=32,  # Smaller batch size
        learning_rate=0.0005,  # Smaller learning rate
        warmup_ratio=0.5  # Warmup KL loss over the first 50% of steps
    )
    
    # Step 4: Plot results
    print("\nStep 4: Visualizing results...")
    plot_training_curves(losses, recon_losses, kl_losses)
    
    # Step 5: Show reconstructions
    reconstruct_and_display(vae, X, n_images=5)
    
    # Step 6: Generate new images
    generate_and_display(vae, n_images=10)
    
    print("\n" + "="*60)
    print("VAE TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()