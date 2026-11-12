from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, n_convs, chn_in, chn_out):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(chn_in, chn_out, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_convs - 1):
            layers.append(nn.Conv2d(chn_out, chn_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        conv_blks = []
        chn_in = 1

        for n_convs, chn_out in cfg.conv_arch:
            conv_blks.append(VGGBlock(n_convs, chn_in, chn_out))
            chn_in = chn_out

        self.features = nn.Sequential(*conv_blks)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chn_out * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
