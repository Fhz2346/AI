import torch
from torch import nn
from .block import G_block, D_block


class Generator(nn.Module):

    def __init__(self, n_G, latent_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            G_block(cI=latent_dim, cO=n_G * 8, strides=1, padding=0),  # (n_G * 8, 4, 4)
            G_block(cI=n_G * 8, cO=n_G * 4),  # (64 * 4, 8, 8)
            G_block(cI=n_G * 4, cO=n_G * 2),  # (64 * 2, 16, 16)
            G_block(cI=n_G * 2, cO=n_G),  # (64, 32, 32)
            nn.ConvTranspose2d(n_G, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )  

    def forward(self, X):
        return self.net(X) # (3, 64, 64)


class Discriminator(nn.Module):

    def __init__(self, n_D):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            D_block(n_D),  # (64, 32, 32)
            D_block(cI=n_D, cO=n_D * 2),  # (64 * 2, 16, 16)
            D_block(cI=n_D * 2, cO=n_D * 4),  # (64 * 4, 8, 8)
            D_block(cI=n_D * 4, cO=n_D * 8),  # (64 * 8, 4, 4)
            nn.Conv2d(n_D * 8, 1, kernel_size=4, bias=False),
        )

    def forward(self, X):
        return self.net(X)
