import torch
import torch.nn as nn
import math

from .MaskedSoftmax import MaskedSoftmax


class Attention(nn.Module):

    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.masked_softmax = MaskedSoftmax()

    def scoring_function(self, queries: torch.Tensor, keys: torch.Tensor):
        raise NotImplementedError

    def forward(self, qkv, valid_lens=None):
        queries, keys, values = qkv                                  # (bs, n?, d?)
        nk, nv = keys.shape[1], values.shape[1]
        assert nk == nv  # n = nk = nv
        scores = self.scoring_function(queries, keys)                # (bs, nq, n)
        attention_weights = self.masked_softmax(scores, valid_lens)  # (bs, nq, n)
        outputs = torch.bmm(self.dropout(attention_weights), values) # (bs, nq, dv)
        return outputs


class DotProductAttention(Attention):

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__(dropout)

    def scoring_function(self, queries, keys):
        dq, dk = queries.shape[-1], keys.shape[-1]
        assert dq == dk
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dq)
        return scores


class AdditiveAttention(Attention):

    def __init__(self, dH, dropout):
        super(AdditiveAttention, self).__init__(dropout)
        self.Wq = nn.LazyLinear(dH, bias=False) # qs -> dH
        self.Wk = nn.LazyLinear(dH, bias=False) # ks -> dH
        self.wv = nn.LazyLinear(1, bias=False)

    def scoring_function(self, queries, keys):
        queries = self.Wq(queries).unsqueeze(2)       # (bs, nq, 1, dH)
        keys = self.Wk(keys).unsqueeze(1)             # (bs, 1, n, dH)
        add = queries + keys                          # (bs, nq, n, dH)
        scores = self.wv(torch.tanh(add)).squeeze(-1) # (bs, nq, n)
        return scores
