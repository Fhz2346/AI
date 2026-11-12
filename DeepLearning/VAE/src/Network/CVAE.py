from torch import nn
from torch.nn import functional as F
import torch
from Network.Module import Sampling


class Encoder(nn.Module):

    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(original_dim, intermediate_dim)
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc(x))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(self, latent_dim, intermediate_dim, original_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, intermediate_dim)
        self.out = nn.Linear(intermediate_dim, original_dim)

    def forward(self, z):
        h = F.relu(self.fc(z))
        return torch.sigmoid(self.out(h))


class CVAE(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.original_dim, cfg.intermediate_dim, cfg.latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(cfg.latent_dim, cfg.intermediate_dim, cfg.original_dim)
        self.class_mean = nn.Linear(cfg.num_classes, cfg.latent_dim)

    def forward(self, x, y):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        yh = self.class_mean(y)
        return reconstruction, z_mean, z_log_var, yh


def loss_fn(x, reconstruction, z_mean, z_log_var, yh):
    x = x.view(x.size(0), -1)
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    # KL(q(z|x) || N(yh, I)), yh is the learned per-class prior mean
    kl_loss = -0.5 * torch.sum(1 + z_log_var - (z_mean - yh).pow(2) - z_log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
