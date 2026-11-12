from torch.nn import functional as F
from torch import torch


def vae_loss(x, reconstruction, z_mean, z_log_var):
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    # KL divergence with standard normal distribution
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    # without it, not able to generate from N(0,I) and latent space is not continuous
    return recon_loss + kl_loss, recon_loss, kl_loss


def cvae_loss(x, reconstruction, z_mean, z_log_var, yh):
    x = x.view(x.size(0), -1)
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    # KL(q(z|x) || N(yh, I)), yh is the learned per-class prior mean
    kl_loss = -0.5 * torch.sum(1 + z_log_var - (z_mean - yh).pow(2) - z_log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
