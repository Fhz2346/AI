import torch
from torch import nn
import hydra
from omegaconf import DictConfig

from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Train.CPU import train
from Network.Utils.Initialize import init_weights
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="MLP")
def main(cfg: DictConfig):
    train_iter, test_iter = load_data_fashion_mnist(cfg.training.batch_size)
    dI, dH, dO = cfg.network.dI, cfg.network.dH, cfg.network.dO

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dI, dH),
        nn.ReLU(),
        nn.Linear(dH, dO),
    )
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = torch.optim.SGD(net.parameters(), lr=cfg.training.lr)
    train(net, train_iter, test_iter, loss, cfg.training.num_epochs, updater)


if __name__ == "__main__":
    main()
