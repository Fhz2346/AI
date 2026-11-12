import torch
from torch import nn

from utils.Device import try_gpu
from Data.Dataset.FashionMnist import load_data_fashion_mnist
from Train.GPU import TrainPipeline
from Network.Module.BatchNorm import BatchNorm


if __name__ == "__main__":
    device = try_gpu()
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)

    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(6, len_shape=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        BatchNorm(16, len_shape=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        BatchNorm(120, len_shape=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, len_shape=2),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
    train_pip = TrainPipeline(net, loss, 10, optimizer, device)
    train_pip.train(train_iter, test_iter)
