from torch import nn
from torch.nn import functional as F
from torch import torch
from Network.Module import Sampling


class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, 16)
        self.z_mean = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [bs, 32, H/2, W/2]
        x = F.relu(self.conv2(x)) # [bs, 64, H/4, W/4]
        x = self.flatten(x)       # [bs, 64 * 7 * 7]
        x = F.relu(self.fc(x))    # [bs, 16]
        
        z_mean = self.z_mean(x)       # [bs, latent_dim]
        z_log_var = self.z_log_var(x) # [bs, latent_dim]
        return z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 64)
        # ConvTranspose2d : learnable upsampling
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(cfg.latent_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var
