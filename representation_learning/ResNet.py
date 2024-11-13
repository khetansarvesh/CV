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
                                        ResidualBlock2(64, 64, nn.Sequential( nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(256)), 1),
                                        ResidualBlock2(256, 64),
                                        ResidualBlock2(256, 64)
                                    )
        
        self.layer2 = nn.Sequential(
                                        ResidualBlock2(256, 128, nn.Sequential(nn.Conv2d(256, 512, 1, 2, bias=False), nn.BatchNorm2d(512)), 2), 
                                        ResidualBlock2(512, 128), 
                                        ResidualBlock2(512, 128), #1
                                        ResidualBlock2(512, 128), #2
                                        ResidualBlock2(512, 128), #3
                                        ResidualBlock2(512, 128), #4
                                        ResidualBlock2(512, 128), #5
                                        ResidualBlock2(512, 128)  #6
                                    )
        
        self.layer3 = nn.Sequential(
                                        ResidualBlock2(512, 256, nn.Sequential(nn.Conv2d(512, 1024, 1, 2, bias=False), nn.BatchNorm2d(1024)), 2),
                                        ResidualBlock2(1024, 256),
                                        
                                        ResidualBlock2(1024, 256), #1
                                        ResidualBlock2(1024, 256), #2
                                        ResidualBlock2(1024, 256), #3
                                        ResidualBlock2(1024, 256), #4
                                        ResidualBlock2(1024, 256), #5
                                        ResidualBlock2(1024, 256), #6
                            
                                        ResidualBlock2(1024, 256), #7
                                        ResidualBlock2(1024, 256), #8
                                        ResidualBlock2(1024, 256), #9
                                        ResidualBlock2(1024, 256), #10
                                        ResidualBlock2(1024, 256), #11
                                        ResidualBlock2(1024, 256), #12
                            
                                        ResidualBlock2(1024, 256), #13
                                        ResidualBlock2(1024, 256), #14
                                        ResidualBlock2(1024, 256), #15
                                        ResidualBlock2(1024, 256), #16
                                        ResidualBlock2(1024, 256), #17
                                        ResidualBlock2(1024, 256), #18
                            
                                        ResidualBlock2(1024, 256), #19
                                        ResidualBlock2(1024, 256), #20
                                        ResidualBlock2(1024, 256), #21
                                        ResidualBlock2(1024, 256), #22
                                        ResidualBlock2(1024, 256), #23
                                        ResidualBlock2(1024, 256), #24
                            
                                        ResidualBlock2(1024, 256), #25
                                        ResidualBlock2(1024, 256), #26
                                        ResidualBlock2(1024, 256), #27
                                        ResidualBlock2(1024, 256), #28
                                        ResidualBlock2(1024, 256), #29
                                        ResidualBlock2(1024, 256), #30
                            
                                        ResidualBlock2(1024, 256), #31
                                        ResidualBlock2(1024, 256), #32
                                        ResidualBlock2(1024, 256), #33
                                        ResidualBlock2(1024, 256), #34
                                    )
        
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

    def _make_layer(self, num_residual_blocks = 36, intermediate_channels=256, stride=2):
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
