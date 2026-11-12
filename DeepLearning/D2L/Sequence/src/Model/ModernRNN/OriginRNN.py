import torch
from Model.RNN import RNN


class OriginRNN(RNN):

    def __init__(self, dI, dH, device):
        super().__init__(dI, dH, device)
        self.W_xh = self.normal(self.dI, self.dH)
        self.W_hh = self.normal(self.dH, self.dH)
        self.b_h = self.zeros(self.dH)

    def begin_state(self, batch_size):
        H = torch.zeros(batch_size, self.dH, device=self.device)
        return H

    def step(self, x, state):
        H = state
        H = torch.tanh(x @ self.W_xh + H @ self.W_hh + self.b_h)
        return H, H
