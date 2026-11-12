import torch
from torch import nn


class Sampling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps
