import torch
from torch import nn


class BatchNorm(nn.Module):

    def __init__(self, n, len_shape, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        if len_shape == 2:  # Linear
            shape = (1, n)
        elif len_shape == 4:  # Conv
            shape = (1, n, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.mean_mov = torch.zeros(shape)
        self.var_mov = torch.ones(shape)

    def batch_norm(self, X: torch.Tensor):
        if not torch.is_grad_enabled():  # Eval
            X_hat = (X - self.mean_mov) / torch.sqrt(self.var_mov + self.eps)
        else:  # Train
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2: # Linear
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else: # Conv
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            # Update moving mean and variance
            self.mean_mov = self.momentum * self.mean_mov + (1.0 - self.momentum) * mean
            self.var_mov = self.momentum * self.var_mov + (1.0 - self.momentum) * var
        Y = self.gamma * X_hat + self.beta
        return Y

    def forward(self, X):
        device = X.device
        if self.mean_mov.device != device:
            self.mean_mov = self.mean_mov.to(device)
            self.var_mov = self.var_mov.to(device)
        Y = self.batch_norm(X)
        return Y
