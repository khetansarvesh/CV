import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups = 32):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # residual block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=1e-6, affine=True),nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=1e-6, affine=True), nn.ReLU()
        )
        
        # dimensionality matching block
        self.dim_matching_layer = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.dim_matching_layer(x) + self.block(x)
        else:
            return x + self.block(x)