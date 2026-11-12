import torch
from torch import nn
import hydra
from omegaconf import DictConfig

from utils.Device import try_gpu
from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Network.CNN_dict import network_dict, init_weights
from Train.GPU import TrainPipeline
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config/CNN", config_name="AlexNet", version_base="1.2")
def main(cfg: DictConfig):
    device = try_gpu()
    print("training on", device)
    train_iter, test_iter = load_data_fashion_mnist(
        cfg.training.batch_size, resize=cfg.training.resize
    )
    net = network_dict[cfg.network.type](cfg.network)
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.training.lr)
    train_pip = TrainPipeline(net, loss, cfg.training.num_epochs, optimizer, device)
    train_pip.train(train_iter, test_iter)


if __name__ == "__main__":
    main()
