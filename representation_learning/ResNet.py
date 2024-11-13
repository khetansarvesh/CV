import torch
import torch.nn as nn
from residual_block_utils.py import ResidualBlock2

class ResNet(nn.Module):
    
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential( nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU() )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock2(64, 256, nn.Sequential( nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(256)), 1),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock2(256, 512, nn.Sequential( nn.Conv2d( self.in_channels, 512, 1, 2, bias=False),nn.BatchNorm2d(512)), 2),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512)
        )
        #self._make_layer(8, intermediate_channels=128, stride=2)
        
        self.layer3 = self._make_layer(36, intermediate_channels=256, stride=2)
        
        self.layer4 = self._make_layer(3, intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):

        # feature extraction
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
            
        # image flattening
        x = x.reshape(x.shape[0], -1)

        # classification
        x = self.fc(x)

        return x

    def _make_layer(self, num_residual_blocks = 3, intermediate_channels=64, stride=1):
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
                identity_downsample = nn.Sequential( nn.Conv2d( self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False, ),nn.BatchNorm2d(intermediate_channels * 4))
        else:
                identity_downsample = None

        layers.append(ResidualBlock2(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(ResidualBlock2(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
