import torch
import torchvision
from torchvision import transforms
from Config import cfgG


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=cfgG.data_dir, train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=cfgG.data_dir, train=False, transform=trans, download=False)
    return (
        torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]
