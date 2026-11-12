import torch
import torch.nn as nn
import torch.nn.functional as F


class _RNN(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def normal(self, *shape):
        return nn.Parameter(torch.randn(*shape, device=self.device) * 0.01)

    def zeros(self, *shape):
        return nn.Parameter(torch.zeros(*shape, device=self.device))


class RNN(_RNN):

    def __init__(self, dI, dH, device):
        super().__init__(device)
        self.dI = dI
        self.dH = dH

    def begin_state(self, batch_size):
        pass

    def step(self, x, state):
        pass

    def forward(self, X, state):
        H_list = []
        for t in range(X.shape[0]):  # num_steps
            x = X[t]
            H, state = self.step(x, state)
            H_list.append(H)
        Hs = torch.stack(H_list, dim=0)
        return Hs, state
