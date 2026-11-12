import numpy as np
from ..Module import Module, Parameter


class Linear(Module):

    def __init__(self, dI, dO):
        scale = np.sqrt(1.0 / dI)
        W = np.random.randn(dO, dI) * scale
        b = np.zeros((dO,))
        self.W = Parameter(W)
        self.b = Parameter(b)
        self._x = None

    def forward(self, x):
        self._x = x
        # (bs,dI)
        y = x @ self.W.data.T + self.b.data
        # (bs,dO)
        return y

    def backward(self, grad_y):
        x = self._x
        self.W.grad += grad_y.T @ x
        self.b.grad += np.sum(grad_y, axis=0)
        grad_x = grad_y @ self.W.data
        return grad_x
