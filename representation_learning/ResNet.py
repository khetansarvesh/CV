import torch
import torch.nn as nn
from residual_block_utils.py import *

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d( in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False, ), nn.BatchNorm2d(intermediate_channels), nn.ReLU()
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False,), nn.BatchNorm2d(intermediate_channels), nn.ReLU()
            nn.Conv2d( intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1, padding=0, bias=False, ), nn.BatchNorm2d(intermediate_channels * 4)
        )

        self.identity_downsample = identity_downsample

    def forward(self, x):

        if self.identity_downsample is not None:
            return self.relu(self.block(x) + self.identity_downsample(identity))
        
        return self.relu(self.block(x) + x)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(3, intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(8, intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(36, intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(3, intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential( nn.Conv2d( self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False, ),nn.BatchNorm2d(intermediate_channels * 4))

        layers.append(ResidualBlock(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(ResidualBlock(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
