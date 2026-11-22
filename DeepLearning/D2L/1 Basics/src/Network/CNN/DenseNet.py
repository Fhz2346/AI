from torch import nn
import torch


class ConvBlock(nn.Module):

    def __init__(self, chn_in, chn_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(chn_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(chn_in, chn_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):

    def __init__(self, n_conv, chn_in, growth_rate):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.append(ConvBlock(chn_in + growth_rate * i, growth_rate))
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


class TransitionBlock(nn.Module):

    def __init__(self, chn_in, chn_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(chn_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(chn_in, chn_out, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        chn, growth_rate = cfg.chn, cfg.growth_rate
        blks = []
        for i, n_conv in enumerate(cfg.n_conv_in_dense_blocks):
            blks.append(DenseBlock(n_conv, chn, growth_rate))
            chn += n_conv * growth_rate

            if i != len(cfg.n_conv_in_dense_blocks) - 1:
                blks.append(TransitionBlock(chn, chn // 2))
                chn = chn // 2

        self.features = nn.Sequential(*blks)
        self.head = nn.Sequential(
            nn.BatchNorm2d(chn),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(chn, 10)
        )
        self.net = nn.Sequential(self.b1, self.features, self.head)

    def forward(self, x):
        return self.net(x)
