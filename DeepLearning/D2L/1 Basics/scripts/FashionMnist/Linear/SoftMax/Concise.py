import torch
from torch import nn

from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Network.Utils.Initialize import init_weights
from Train.CPU import train
from Config import cfgSM


if __name__ == "__main__":
    train_iter, test_iter = load_data_fashion_mnist(cfgSM.batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=cfgSM.lr)
    train(net, train_iter, test_iter, loss, cfgSM.num_epochs, trainer)
