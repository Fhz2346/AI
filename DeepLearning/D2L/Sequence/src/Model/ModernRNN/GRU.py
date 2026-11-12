import torch
from Model.RNN import RNN


class GRU(RNN):

    def __init__(self, dI, dH, device=None):
        super().__init__(dI, dH, device)
        # update gate z
        self.W_xz = self.normal(self.dI, self.dH)
        self.W_hz = self.normal(self.dH, self.dH)
        self.b_z = self.zeros(self.dH)
        # reset gate r
        self.W_xr = self.normal(self.dI, self.dH)
        self.W_hr = self.normal(self.dH, self.dH)
        self.b_r = self.zeros(self.dH)
        # candidate hidden state h_tilde
        self.W_xh = self.normal(self.dI, self.dH)
        self.W_hh = self.normal(self.dH, self.dH)
        self.b_h = self.zeros(self.dH)

    def begin_state(self, batch_size):
        H = torch.zeros(batch_size, self.dH, device=self.device)
        return H

    def step(self, x, state):
        H = state
        R = torch.sigmoid(x @ self.W_xr + H @ self.W_hr + self.b_r)
        Z = torch.sigmoid(x @ self.W_xz + H @ self.W_hz + self.b_z)
        H_tilde = torch.tanh(x @ self.W_xh + (R * H) @ self.W_hh + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        return H, H
