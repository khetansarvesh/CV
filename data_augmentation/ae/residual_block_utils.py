import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

residual_encoder = nn.Sequential(nn.Conv2d(1, 128, 3, 1, 1),

                              #ResidualBlock(128, 128),
                              #ResidualBlock(128, 128),
                              #nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),

                              #ResidualBlock(128, 128),
                              #ResidualBlock(128, 128),
                              #nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),

                              ResidualBlock(128, 256),
                              #ResidualBlock(256, 256),
                              nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),

                              #ResidualBlock(256, 256),
                              #ResidualBlock(256, 256),
                              nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),

                              ResidualBlock(256, 512),
                              #ResidualBlock(512, 512),

                              #ResidualBlock(512, 512),
                              ResidualBlock(512, 512),

                              nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(512, 256, 3, 1, 1)
                            )

residual_decoder = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                              ResidualBlock(512, 512),
                              #ResidualBlock(512, 512),

                              #ResidualBlock(512, 512),
                              #ResidualBlock(512, 512),
                              #ResidualBlock(512, 512),

                              ResidualBlock(512, 256),
                              #ResidualBlock(256, 256),
                              ResidualBlock(256, 256),
                              nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),

                              #ResidualBlock(256, 256),
                              #ResidualBlock(256, 256),
                              #ResidualBlock(256, 256),
                              #nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),

                              ResidualBlock(256, 128),
                              #ResidualBlock(128, 128),
                              ResidualBlock(128, 128),
                              nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),

                              #ResidualBlock(128, 128),
                              #ResidualBlock(128, 128),
                              #ResidualBlock(128, 128),
                              #nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),

                              nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6, affine=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 1, 3, 1, 1))
