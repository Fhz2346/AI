from torch import nn


def NiN_block(chn_in, chn_out, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(chn_in, chn_out, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(chn_out, chn_out, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(chn_out, chn_out, kernel_size=1),
        nn.ReLU(),
    )


class NiN(nn.Module):

    def __init__(self, cfg):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            NiN_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            NiN_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NiN_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NiN_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)
