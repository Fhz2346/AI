import torch
from Model.RNN import RNN


class LSTM(RNN):

    def __init__(self, dI, dH, device=None):
        super().__init__(dI, dH, device)
        # input gate
        self.W_xi = self.normal(self.dI, self.dH)
        self.W_hi = self.normal(self.dH, self.dH)
        self.b_i = self.zeros(self.dH)
        # forget gate
        self.W_xf = self.normal(self.dI, self.dH)
        self.W_hf = self.normal(self.dH, self.dH)
        self.b_f = self.zeros(self.dH)
        # output gate
        self.W_xo = self.normal(self.dI, self.dH)
        self.W_ho = self.normal(self.dH, self.dH)
        self.b_o = self.zeros(self.dH)
        # candidate cell
        self.W_xc = self.normal(self.dI, self.dH)
        self.W_hc = self.normal(self.dH, self.dH)
        self.b_c = self.zeros(self.dH)

    def begin_state(self, batch_size):
        H = torch.zeros(batch_size, self.dH, device=self.device)
        C = torch.zeros(batch_size, self.dH, device=self.device)
        return H, C

    def step(self, x, state):
        H, C = state
        C_tilda = torch.tanh(x @ self.W_xc + H @ self.W_hc + self.b_c)
        I = torch.sigmoid(x @ self.W_xi + H @ self.W_hi + self.b_i)
        F_gate = torch.sigmoid(x @ self.W_xf + H @ self.W_hf + self.b_f)
        C = F_gate * C + I * C_tilda
        O = torch.sigmoid(x @ self.W_xo + H @ self.W_ho + self.b_o)
        H = O * torch.tanh(C)
        return H, (H, C)
