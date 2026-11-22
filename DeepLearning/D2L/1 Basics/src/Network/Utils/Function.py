import torch
from torch import nn


def softmax(X):
    X_exp = torch.exp(X)
    sum = X_exp.sum(1, keepdim=True)
    return X_exp / sum


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
