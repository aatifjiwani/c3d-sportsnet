import torch.nn as nn
import torch
from torch import flatten
from torch import cat 

from typing import Union

class C3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C3D, self).__init__()

        self.conv_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc_6 = nn.Linear(8192, 4096)
        self.fc_7 = nn.Linear(4096, 4096)
        self.fc_8 = nn.Linear(4096, out_channels)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.pool_2(x)

        x = self.conv_3a(x)
        x = self.conv_3b(x)
        x = self.pool_3(x)

        x = self.conv_4a(x)
        x = self.conv_4b(x)
        x = self.pool_4(x)

        x = self.conv_5a(x)
        x = self.conv_5b(x)
        x = self.pool_5(x)

        x = x.view(-1, 8192)

        x = self.fc_6(x)
        x = self.fc_7(x)
        x = self.fc_8(x)

        out = self.softmax(x)

        return x




