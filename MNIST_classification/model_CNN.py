import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*14*14, 10)

    def forward(self, x):
        # Nx1x28x28 -> Nx64x28x28
        out = self.conv(x)
        out = self.batch_norm(out)
        out = nn.functional.relu(out)

        # Nx64x28x28 -> Nx64x28x28
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = nn.functional.relu(out)

        # Nx64x28x28 -> Nx64x14x14
        out = nn.functional.max_pool2d(out, (2, 2))

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
