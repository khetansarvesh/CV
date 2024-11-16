import torch
from torch import nn

class DepthwiseSepConvBlock(nn.Module):

    def __init__( self, in_channels: int, out_channels: int, stride: int = 1):

        super().__init__()

        # Depthwise conv => you just add an extra parameter called groups = input_channel
        self.dc = nn.Sequential[
                        nn.Conv2d( in_channels, in_channels, kernel_size = (3, 3), stride = stride, padding = 1, groups = in_channels),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU6()]

        # Pointwise Convolution
        self.pc = nn.Sequential[
                        nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1)),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU6()]

    def forward(self, x):

        x = self.dc(x)
        x = self.pc(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1),


            DepthwiseSepConvBlock(32, 64, 1),
            DepthwiseSepConvBlock(64, 128, 2),
            DepthwiseSepConvBlock(128, 128, 1),
            DepthwiseSepConvBlock(128, 256, 2),
            DepthwiseSepConvBlock(256, 256, 1),
            DepthwiseSepConvBlock(256, 512, 2),
            DepthwiseSepConvBlock(512, 512, 1),
            DepthwiseSepConvBlock(512, 512, 1),
            DepthwiseSepConvBlock(512, 512, 1),
            DepthwiseSepConvBlock(512, 512, 1),
            DepthwiseSepConvBlock(512, 512, 1),
            DepthwiseSepConvBlock(512, 1024, 2),
            DepthwiseSepConvBlock(1024, 1024, 1),


            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1000)
        )

    def forward(self, x):
        return self.model(x)