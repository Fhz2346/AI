import torch
from torch import nn

class G_block(nn.Module):

    def __init__(self, cI, cO=3, kernel_size=4, strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cI, cO, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(cO),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.net(X)

class D_block(nn.Module):

    def __init__(
        self, cO, cI=3, kernel_size=4, strides=2, padding=1, alpha=0.2, **kwargs
    ):
        super(D_block, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(cI, cO, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(cO),
            nn.LeakyReLU(alpha, inplace=True),
        )

    def forward(self, X):
        return self.net(X)
