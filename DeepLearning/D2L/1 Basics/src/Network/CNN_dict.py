from Network.CNN import AlexNet, NiN, VGG, GoogLeNet, ResNet, DenseNet
from torch import nn

network_dict = {
    "AlexNet": AlexNet,
    "NiN": NiN,
    "VGG": VGG,
    "GoogLeNet": GoogLeNet,
    "ResNet": ResNet,
    "DenseNet": DenseNet,
}

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
