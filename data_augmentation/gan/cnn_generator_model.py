import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.ConvTranspose2d(100, 512, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(256),nn.ReLU(),
                                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), nn.Identity(),nn.Tanh()
                                    )

    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, 100, 1, 1) # reshaping
        out = self.layers(out)
        return out.reshape(batch_size, 1, 28, 28) # reshaping