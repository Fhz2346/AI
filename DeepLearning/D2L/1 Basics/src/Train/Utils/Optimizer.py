import torch


class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad
                param.grad.zero_()
