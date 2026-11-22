import torch
from utils.Auxiliary import Accumulator, Timer
from Evaluate import accuracy, evaluate_accuracy_gpu
from tqdm import tqdm


class TrainPipeline:

    def __init__(self, net, loss, num_epochs, optimizer, device):
        self.net = net.to(device)
        self.loss = loss
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device
        self.timer = Timer()
        self.metric = Accumulator(3)
    
    def train_epoch(self, train_iter):
        if isinstance(self.net, torch.nn.Module):
            self.net.train()
        for X, y in train_iter:
            self.timer.start()
            self.optimizer.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            l.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            self.timer.stop()
        train_loss = self.metric[0] / self.metric[2]
        train_acc = self.metric[1] / self.metric[2]
        return train_loss, train_acc

    def train(self, train_iter, test_iter):
        pbar = tqdm(range(self.num_epochs), desc="Training")
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch(train_iter)
            test_acc = evaluate_accuracy_gpu(self.net, test_iter, self.device)
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "test_acc": f"{test_acc:.4f}",
                }
            )


def train_epoch(net, train_iter, loss, optimizer, device, timer: Timer, metric: Accumulator):
    if isinstance(net, torch.nn.Module):
        net.train()
    for X, y in train_iter:
        timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        timer.stop()
    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    return train_loss, train_acc


def train(net, train_iter, test_iter, loss, num_epochs, updater, device):
    net.to(device)
    timer = Timer()
    metric = Accumulator(3)
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater, device, timer, metric)
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "test_acc": f"{test_acc:.4f}",
            }
        )
