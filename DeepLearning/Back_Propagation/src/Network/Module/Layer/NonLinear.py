import numpy as np
from ..Module import Module


class ReLU(Module):

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_y):
        return grad_y * self._mask


class Sigmoid(Module):

    def __init__(self):
        self._y = None

    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        self._y = y
        # (bs, dO)
        return y

    def backward(self, grad_y):
        y = self._y
        return grad_y * (y * (1.0 - y))


class Tanh(Module):

    def __init__(self):
        self._y = None

    def forward(self, x):
        y = np.tanh(x)
        self._y = y
        return y

    def backward(self, grad_y):
        y = self._y
        return grad_y * (1.0 - y * y)


act_dict = {
    "ReLU": ReLU,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
}
