import torch
import torch.nn as nn

input_dim, output_dim = 4, 2
conv_block = nn.Sequential(
            # 512x512x1 → 256x256x32
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 256x256x32 → 128x128x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 128x128x64 → 64x64x128
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 64x64x128 → 64x64x400
            nn.Conv2d(32, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

x = torch.rand((1, 4, 512, 512))
y = conv_block(x)
print(y.shape)