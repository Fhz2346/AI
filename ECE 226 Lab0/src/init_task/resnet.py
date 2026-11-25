import torch.nn as nn


class RESNET(nn.Module):

    def __init__(self):
        super(RESNET, self).__init__()
        in_channel = 3
        num_classes = 10
        self.channel_1 = 64
        self.channel_2 = 128
        self.channel_3 = 256
        self.channel_4 = 512

        self.conv_1 = nn.Conv2d(in_channel, self.channel_1, (7, 7), padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.block_1 = nn.Sequential(
            nn.Conv2d(self.channel_1, self.channel_1, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.channel_1),
            nn.ReLU(),
            nn.Conv2d(self.channel_1, self.channel_1, (3, 3), padding=1, stride=1),
        )

        self.bn2 = nn.BatchNorm2d(self.channel_1)
        self.block_2 = nn.Sequential(
            nn.Conv2d(self.channel_1, self.channel_2, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.channel_2),
            nn.ReLU(),
            nn.Conv2d(self.channel_2, self.channel_2, (3, 3), padding=1, stride=1),
        )

        self.bn3 = nn.BatchNorm2d(self.channel_2)
        self.conv_2_skip = nn.Conv2d(self.channel_1, self.channel_2, (1, 1), stride=1)
        self.block_3 = nn.Sequential(
            nn.Conv2d(self.channel_2, self.channel_3, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.channel_3),
            nn.ReLU(),
            nn.Conv2d(self.channel_3, self.channel_3, (3, 3), padding=1, stride=1),
        )

        self.bn4 = nn.BatchNorm2d(self.channel_3)
        self.conv_3_skip = nn.Conv2d(self.channel_2, self.channel_3, (1, 1), stride=1)
        self.block_4 = nn.Sequential(
            nn.Conv2d(self.channel_3, self.channel_4, (3, 3), padding=1, stride=2),
            nn.BatchNorm2d(self.channel_4),
            nn.ReLU(),
            nn.Conv2d(self.channel_4, self.channel_4, (3, 3), padding=1, stride=1),
        )

        self.bn5 = nn.BatchNorm2d(self.channel_4)
        self.conv_4_skip = nn.Conv2d(self.channel_3, self.channel_4, (1, 1), stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((3, 3), padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(self.channel_4, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        maxpool_out = self.maxpool(x)
        x = self.block_1(maxpool_out)
        x = self.bn2(x)
        block1_out = x + maxpool_out
        x = self.block_2(block1_out)
        x = self.bn3(x)
        block2_out = x + self.conv_2_skip(block1_out)
        x = self.block_3(block2_out)
        x = self.bn4(x)
        block3_out = x + self.conv_3_skip(block2_out)
        x = self.block_4(block3_out)
        x = self.bn5(x)
        block4_out = x + self.conv_4_skip(block3_out)
        x = self.avgpool(block4_out)
        x = x.view(x.shape[0], -1)
        scores = self.fc(x)
        return scores
