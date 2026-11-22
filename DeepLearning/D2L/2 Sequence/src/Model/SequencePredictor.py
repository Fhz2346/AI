import torch.nn as nn
from Model.RNN import RNN

class SequencePredictor(nn.Module):

    def __init__(self, rnn: RNN, vocab_size):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.W_hq = self.rnn.normal(self.rnn.dH, vocab_size)
        self.b_q = self.rnn.zeros(vocab_size)

    def forward(self, X, state):
        Hs, state = self.rnn(X, state)
        Y = Hs @ self.W_hq + self.b_q
        return Y, state
