import numpy as np


class Parameter:

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad[...] = 0.0


class Module:

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, Parameter):
                        params.append(it)
                    elif isinstance(it, Module):
                        params.extend(it.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_y):
        raise NotImplementedError
        # return grad_x 

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out
