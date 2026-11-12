import torch


def squared_loss(y, y_hat):
    return (y_hat - y) ** 2 / 2


def cross_entropy(y_hat, y_t):
    return -torch.log(y_hat[range(len(y_hat)), y_t])
