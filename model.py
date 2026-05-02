"""
model.py — Variational Autoencoder (VAE) for SWaT Anomaly Detection
---------------------------------------------------------------------
ARCHITECTURE OVERVIEW:

    Input (42 sensors)
        ↓
    [Encoder]  — two fully-connected layers that compress the input
        ↓
    mu (mean vector)   +   log_var (log-variance vector)   ← latent space params
        ↓
    [Reparameterization Trick]  — sample z = mu + eps * std
        ↓
    [Decoder]  — two fully-connected layers that reconstruct the input
        ↓
    Reconstructed Input (42 sensors)

WHY PYTORCH?
    PyTorch lets you define neural networks as Python classes, with the
    forward pass written as regular Python code. This makes it very easy
    to understand what's happening at each step — you're not hiding anything
    behind a "model.fit()" black box.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Args:
        input_dim  : number of sensor features (columns in your CSV, after dropping label)
        hidden_dim : size of the intermediate hidden layers (controls model capacity)
        latent_dim : size of the compressed "bottleneck" representation
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        # Always call the parent class constructor first in PyTorch
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # ── ENCODER ───────────────────────────────────────────────────────────
        # Takes raw sensor data and compresses it into a hidden representation.
        # We use two layers to give the model enough capacity to learn
        # non-linear relationships between sensors.
        #
        # nn.Linear(in, out) : fully-connected layer  (y = xW + b)
        # nn.ReLU()          : activation function — adds non-linearity
        # nn.BatchNorm1d()   : normalizes activations during training,
        #                      stabilizes and speeds up training
        # nn.Dropout(0.1)    : randomly zeroes 10% of neurons during training
        #                      to prevent overfitting

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # The encoder ends by producing TWO separate vectors:
        # - fc_mu      : the mean (μ) of the latent distribution
        # - fc_log_var : the log-variance (log σ²) of the latent distribution
        #
        # We predict log_var instead of variance directly because:
        # 1. Log-variance can be any real number (no constraints)
        # 2. Variance must always be positive — log makes this automatic

        self.fc_mu      = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # ── DECODER ───────────────────────────────────────────────────────────
        # Takes the latent vector z and reconstructs the original sensor readings.
        # It's the "mirror image" of the encoder (narrow → wide).

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # No activation at the end — we want raw reconstructed values
            # (the input was standardized, so output can be any real number)
        )

    def encode(self, x):
        """
        Pass input through encoder, return mu and log_var.

        Args:
            x : tensor of shape (batch_size, input_dim)
        Returns:
            mu      : mean of latent distribution, shape (batch_size, latent_dim)
            log_var : log-variance of latent distribution, same shape
        """
        # Get the shared hidden representation
        h = self.encoder(x)

        # Split into mean and log-variance branches
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        The Reparameterization Trick — this is the core VAE innovation.

        THE PROBLEM: We want to sample z from N(mu, sigma²), but sampling
        is not differentiable — gradients can't flow through a random sample.

        THE TRICK: Instead of sampling z directly, we sample eps from N(0,1)
        (a standard normal) and compute:
            z = mu + eps * sigma

        Now mu and sigma are the only learnable parts — eps is just fixed
        random noise. Gradients flow through mu and sigma normally. ✅

        This is only applied during TRAINING. During inference, we just use mu.

        Args:
            mu      : mean vector
            log_var : log-variance vector

        Returns:
            z : sampled latent vector, same shape as mu
        """
        if self.training:
            # Convert log-variance to standard deviation
            # log_var → var → std:  std = exp(log_var / 2) = exp(0.5 * log_var)
            std = torch.exp(0.5 * log_var)

            # Sample epsilon from standard normal (same shape as std)
            eps = torch.randn_like(std)

            # Reparameterized sample
            return mu + eps * std
        else:
            # At inference time, just use the mean (no randomness needed)
            return mu

    def decode(self, z):
        """
        Reconstruct input from latent vector.

        Args:
            z : latent vector, shape (batch_size, latent_dim)
        Returns:
            x_reconstructed : shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass: encode → reparameterize → decode.

        Args:
            x : input tensor, shape (batch_size, input_dim)
        Returns:
            x_recon  : reconstructed input
            mu       : latent mean (used for KL loss)
            log_var  : latent log-variance (used for KL loss)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────

def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β × KL Divergence

    RECONSTRUCTION LOSS (MSE):
        How different is the reconstructed output from the original input?
        We use Mean Squared Error — large differences are penalized heavily.
        This forces the decoder to reconstruct the input accurately.

        Formula: mean( (x - x_recon)² )

    KL DIVERGENCE:
        How far is our learned latent distribution N(mu, sigma²) from
        a standard normal N(0, 1)?

        Intuition: Without this term, the encoder could collapse to tiny
        point distributions (like a regular autoencoder), losing the
        benefits of a smooth, continuous latent space.

        Formula (closed form for Gaussian):
            KL = -0.5 × sum(1 + log_var - mu² - exp(log_var))

        When mu=0 and log_var=0 (i.e., sigma=1), KL = 0. ✅

    β (beta):
        Controls the trade-off between the two losses.
        β=1 is standard VAE. β<1 emphasizes reconstruction (better for anomaly detection).
        β>1 emphasizes structure in latent space (better for generation).
        We default to β=0.5 for anomaly detection — reconstruction matters more.

    Args:
        x       : original input
        x_recon : reconstructed input
        mu      : latent mean
        log_var : latent log-variance
        beta    : KL weight (default 1.0, we'll override to 0.5 in train.py)

    Returns:
        total_loss  : combined loss (scalar)
        recon_loss  : reconstruction component (for logging)
        kl_loss     : KL component (for logging)
    """
    # Reconstruction loss: average over all features and batch items
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    # KL divergence: sum over latent dimensions, mean over batch
    # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )

    # Combine with beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
