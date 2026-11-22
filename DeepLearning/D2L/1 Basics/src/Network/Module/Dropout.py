import torch
import torch.nn as nn


class Dropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, X):
        if not self.training:
            return X

        if self.p == 1:
            return torch.zeros_like(X)
        if self.p == 0:
            return X

        mask = (torch.rand_like(X) > self.p).float()
        return mask * X / (1.0 - self.p)
