from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):

    def __init__(self, chn_in, chn_out, use_1x1=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            chn_in, chn_out, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(chn_out, chn_out, kernel_size=3, padding=1)
        if use_1x1:
            self.conv_ = nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=strides)
        else:
            self.conv_ = None
        self.bn1 = nn.BatchNorm2d(chn_out)
        self.bn2 = nn.BatchNorm2d(chn_out)

    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        if self.conv_:
            X = self.conv_(X)
        Y += X
        return F.relu(Y)


class ResNetBlock(nn.Module):

    def __init__(self, chn_in, chn_out, n_res, first_block=False):
        super().__init__()
        layers = []
        for i in range(n_res):
            if i == 0 and not first_block:
                layers.append(Residual(chn_in, chn_out, use_1x1=True, strides=2))
            else:
                layers.append(Residual(chn_out, chn_out))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = ResNetBlock(64, 64, 2, first_block=True)
        self.b3 = ResNetBlock(64, 128, 2)
        self.b4 = ResNetBlock(128, 256, 2)
        self.b5 = ResNetBlock(256, 512, 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, self.head)

    def forward(self, x):
        return self.net(x)
