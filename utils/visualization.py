import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_curves(losses, recon_losses, kl_losses, save_path="training_curves.png"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total VAE Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")
    plt.show()

def generate_and_display(vae, n_images=10, save_path="generated_images.png"):
    """Generate new images from random noise"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    
    for i in range(n_images):
        z = np.random.randn(vae.latent_dim, 1)
        img = vae.decode(z).reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Gen {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Generated Images')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved generated images to {save_path}")
    plt.show()

def reconstruct_and_display(vae, X_images, n_images=5, save_path="reconstructed_images.png"):
    """Show original vs reconstructed images"""
    fig, axes = plt.subplots(2, n_images, figsize=(2*n_images, 4))
    
    for i in range(n_images):
        x = X_images[i].reshape(784, 1)
        x_recon, _, _ = vae.forward(x)
        
        axes[0, i].imshow(x.reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(x_recon.reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f'Recon {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Original vs Reconstructed')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved reconstructed images to {save_path}")
    plt.show()
