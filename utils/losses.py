import numpy as np

def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    epsilon = 1e-8
    
    # Reconstruction loss (clamp to prevent log(0))
    x_recon = np.clip(x_recon, epsilon, 1 - epsilon)
    recon_loss = -np.sum(
        x * np.log(x_recon) + 
        (1 - x) * np.log(1 - x_recon)
    )
    
    # KL loss (clip log_var)
    log_var = np.clip(log_var, -10, 10)
    kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
