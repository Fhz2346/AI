import torch
import hydra
from omegaconf import DictConfig

from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Train.Utils.Loss import cross_entropy
from Network.Utils.Function import softmax
from Train.Utils.Optimizer import SGD
from Train.CPU import train
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="SoftMax")
def main(cfg: DictConfig):
    train_iter, test_iter = load_data_fashion_mnist(cfg.training.batch_size)
    dI, dO = cfg.network.dI, cfg.network.dO
    W = torch.normal(0, 0.01, size=(dI, dO), requires_grad=True)
    b = torch.zeros(dO, requires_grad=True)

    def net(X):
        X = X.reshape((-1, dI))
        return softmax(X @ W + b)

    updater = SGD([W, b], lr=cfg.training.lr)
    train(net, train_iter, test_iter, cross_entropy, cfg.training.num_epochs, updater)


if __name__ == "__main__":
    main()
