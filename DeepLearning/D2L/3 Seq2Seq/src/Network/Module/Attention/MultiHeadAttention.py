import torch
import torch.nn as nn
from .Attention import DotProductAttention


class _MultiHeadAttention(nn.Module):

    def __init__(self, n_head):
        super(_MultiHeadAttention, self).__init__()
        self.n_head = n_head

    def detach_head_one(self, X):
        bs, seq_len, dH = X.shape
        dH_per_head = dH // self.n_head
        X = X.reshape(bs, seq_len, self.n_head, dH_per_head) # (bs, seq_len, n_head, dH_per_head)
        X = X.permute(0, 2, 1, 3)                            # (bs, n_head, seq_len, dH_per_head)
        X = X.reshape(-1, seq_len, dH_per_head)              # (bs*n_head, seq_len, dH_per_head)
        return X

    def detach_head_qkv(self, queries, keys, values):
        queries = self.detach_head_one(queries) # (bs*n_head, nq, dH_per_head)
        keys = self.detach_head_one(keys)       # (bs*n_head, n, dH_per_head)
        values = self.detach_head_one(values)   # (bs*n_head, n, dH_per_head)
        return queries, keys, values

    def transpose_output(self, X: torch.Tensor):
        _, nq, dH_per_head = X.shape
        X = X.reshape(-1, self.n_head, nq, dH_per_head) # (bs, n_head, nq, dH_per_head)
        X = X.permute(0, 2, 1, 3)                       # (bs, nq, n_head, dH_per_head)
        X = X.reshape(X.shape[0], nq, -1)               # (bs, nq, dH)
        return X


class MultiHeadAttention(_MultiHeadAttention):
    """Do attention for n_head times and then concatenate the results"""

    def __init__(self, dH, n_head, dropout, bias=False):
        super(MultiHeadAttention, self).__init__(n_head)
        self.Wq = nn.LazyLinear(dH, bias=bias)
        self.Wk = nn.LazyLinear(dH, bias=bias)
        self.Wv = nn.LazyLinear(dH, bias=bias)
        
        self.attention = DotProductAttention(dropout)
        self.Wo = nn.Linear(dH, dH, bias=bias)
    
    def forward(self, qkv, valid_lens):
        queries, keys, values = qkv                                         # (bs, n?, d?)
        queries, keys, values = self.unify_size_qkv(queries, keys, values)  # (bs, n?, dH)
        queries, keys, values = self.detach_head_qkv(queries, keys, values) # (bs*n_head, n?, dH_per_head)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.n_head, dim=0) # (bs*n_head)
        out = self.attention((queries, keys, values), valid_lens) # (bs*n_head, nq, dH_per_head)
        out = self.transpose_output(out)                          # (bs, nq, dH)
        out = self.Wo(out)                                        # (bs, nq, dH)
        return out
    
    def unify_size_qkv(self, queries, keys, values):
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)
        return queries, keys, values
