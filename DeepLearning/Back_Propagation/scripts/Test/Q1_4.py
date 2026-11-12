import numpy as np
import hydra
from omegaconf import DictConfig

from Network.Module.Module import Module
from Network.Module.Layer.Linear import Linear
from Network.Module.Layer.NonLinear import Tanh
from Network.Module.Loss import MAE
from Train.Optimizer import SGD
from Config import cfgG


class MLP(Module):

    def __init__(self, cfg):
        self.linear1 = Linear(4, 3)
        self.linear1.W.data = np.array(cfg.W)
        self.linear1.b.data = np.array(cfg.b)
        self.tanh = Tanh()
        self.linear2 = Linear(3, 1)
        self.linear2.W.data = np.array(cfg.v).reshape(1, -1)
        self.linear2.b.data = np.array([cfg.c], dtype=np.float64)

    def forward(self, x):
        z = self.linear1(x)
        h = self.tanh(z)
        y = self.linear2(h)
        return y

    def backward(self, grad_y):
        grad_h = self.linear2.backward(grad_y)
        grad_z = self.tanh.backward(grad_h)
        grad_x = self.linear1.backward(grad_z)
        return grad_x


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="Q1_4", version_base=None)
def main(cfg: DictConfig):
    model = MLP(cfg.network)
    x = np.array([cfg.data.x])
    y = np.array([[cfg.data.y]], dtype=np.float64)
    criterion = MAE()
    optimizer = SGD(model.parameters(), lr=cfg.train.eta)
    
    pred = model(x)
    loss, grad_pred = criterion(pred, y)
    print(f"loss(before step): {loss:.6f}")
    
    optimizer.zero_grad()
    model.backward(grad_pred)
    optimizer.step()
    
    print("W:", model.linear1.W.data)
    print("b:", model.linear1.b.data)
    print("v:", model.linear2.W.data)
    print("c:", model.linear2.b.data)
    
    pred = model(x)
    loss, grad_pred = criterion(pred, y)
    print(f"loss(after step): {loss:.6f}")


if __name__ == "__main__":
    main()
