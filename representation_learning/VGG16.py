import torch
import torch.nn as nn

class VGG_net(nn.Module):

    def __init__(self):

        super(VGG_net, self).__init__()

        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.fcs = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 4096),nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096, 4096),nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096, 1000),
                    )

    def forward(self, x):

        # passing through convolution layers
        x = self.conv_layers(x)

        # flattening
        x = x.reshape(x.shape[0], -1)

        # passing through linear layers
        x = self.fcs(x)

        return x

