import torch
import torch.nn as nn  

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # relu layer
        self.relu = nn.ReLU()

        # average pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1,padding=0)

        # FFNN layers
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):

        # conv layer 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # conv layer 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # conv layer 3
        x = self.relu(self.conv3(x))

        # flattening
        x = x.reshape(x.shape[0], -1)

        # linear layers
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x