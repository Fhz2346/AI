import torch
from torch import nn
import hydra
from omegaconf import DictConfig

from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Train.CPU import train
from Network.Utils.Function import relu
from Config import cfgG

@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="MLP")
def main(cfg: DictConfig):
    train_iter, test_iter = load_data_fashion_mnist(cfg.training.batch_size)
    dI, dH, dO = cfg.network.dI, cfg.network.dH, cfg.network.dO

    W1 = nn.Parameter(torch.randn(dI, dH) * 0.01)
    b1 = nn.Parameter(torch.zeros(dH))
    W2 = nn.Parameter(torch.randn(dH, dO) * 0.01)
    b2 = nn.Parameter(torch.zeros(dO))
    params = [W1, b1, W2, b2]

    def net(X):
        X = X.reshape((-1, dI))
        H = relu(X @ W1 + b1)
        O = H @ W2 + b2
        return O

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = torch.optim.SGD(params, lr=cfg.training.lr)
    train(net, train_iter, test_iter, loss, cfg.training.num_epochs, updater)

if __name__ == "__main__":
    main()
