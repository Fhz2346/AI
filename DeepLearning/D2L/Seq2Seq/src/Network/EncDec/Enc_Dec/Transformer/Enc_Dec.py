import torch
import torch.nn as nn
import math

from Network.Module.Attention.MultiHeadAttention import MultiHeadAttention
from .Basics import PositionWiseFFN, AddNorm, PositionalEncoding


class EncoderBlock(nn.Module):

    def __init__(self, cfg):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(cfg.dH, cfg.MultiHeadAttention.n_head, cfg.dropout, cfg.MultiHeadAttention.use_bias)
        self.addnorm1 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)
        self.ffn = PositionWiseFFN(cfg.dH, cfg.PositionWiseFFN.dH, cfg.dH)
        self.addnorm2 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)

    def forward(self, X, src_vl):
        X = self.addnorm1(X, self.attention((X, X, X), src_vl))
        X = self.addnorm2(X, self.ffn(X))
        return X


class TransformerEncoder(nn.Module):

    def __init__(self, src_vs, cfg):
        super(TransformerEncoder, self).__init__()
        self.dH = cfg.dH
        self.embedding = nn.Embedding(src_vs, cfg.dH)
        self.pos_encoding = PositionalEncoding(self.dH, cfg.dropout)

        self.blks = nn.Sequential()
        for _ in range(cfg.n_blk):
            blk = EncoderBlock(cfg)
            self.blks.append(blk)

    def forward(self, X, src_vl):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dH))
        for blk in self.blks:
            X = blk(X, src_vl)
        return X # (bs, seq_len, dH)


class DecoderBlock(nn.Module):

    def __init__(self, cfg):
        super(DecoderBlock, self).__init__()
        dH = cfg.dH
        self.attention1 = MultiHeadAttention(dH, cfg.MultiHeadAttention.n_head, cfg.dropout, cfg.MultiHeadAttention.use_bias)
        self.addnorm1 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)
        self.attention2 = MultiHeadAttention(dH, cfg.MultiHeadAttention.n_head, cfg.dropout, cfg.MultiHeadAttention.use_bias)
        self.addnorm2 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)
        self.ffn = PositionWiseFFN(dH, cfg.PositionWiseFFN.dH, dH)
        self.addnorm3 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)

    def forward(self, X, encO, src_vl, dec_vl):
        X = self.addnorm1(X, self.attention1((X, X, X), dec_vl))       # (bs, seq_len, dH)
        X = self.addnorm2(X, self.attention2((X, encO, encO), src_vl)) # (bs, seq_len, dH)
        X = self.addnorm3(X, self.ffn(X))                              # (bs, seq_len, dH)
        return X


class TransformerDecoder(nn.Module):

    def __init__(self, tgt_vs, cfg):
        super(TransformerDecoder, self).__init__()
        self.dH = cfg.dH

        self.embedding = nn.Embedding(tgt_vs, cfg.dH)
        self.pos_encoding = PositionalEncoding(cfg.dH, cfg.dropout)
        self.blks = nn.Sequential()
        for _ in range(cfg.n_blk):
            blk = DecoderBlock(cfg)
            self.blks.append(blk)

    def forward(self, X, encO, src_vl):

        def get_dec_vl(X):
            bs, seq_len = X.shape[0:2]
            dec_vl = torch.arange(1, seq_len + 1, device=X.device) # (seq_len,)
            dec_vl = dec_vl.repeat(bs, 1) # (bs, seq_len)
            return dec_vl 

        if self.training:
            dec_vl = get_dec_vl(X)
        else:
            dec_vl = None
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dH)) # (bs, seq_len, dH)
        for blk in self.blks:
            X = blk(X, encO, src_vl, dec_vl) # (bs, seq_len, dH)
        return X
