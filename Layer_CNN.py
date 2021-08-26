import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 384, (6, 2), (2, 1), 0),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.flc = nn.Linear(16128, 6)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)

        return x
