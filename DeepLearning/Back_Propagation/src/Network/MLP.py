from .Module.Module import Module, Sequential
from .Module.Layer.Linear import Linear
from .Module.Layer.NonLinear import act_dict


class MLP(Module):

    def __init__(self, cfg):
        dims = cfg.dims
        assert len(dims) >= 2
        layers = []
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                NonLinear = act_dict[cfg.activation]
                layers.append(NonLinear())
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad_out):
        return self.net.backward(grad_out)
