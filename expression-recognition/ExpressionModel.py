import torch
from torch import nn
import torch.nn.functional as F

class ExpressionModel(nn.Module):
    def __init__(self):
        super(ExpressionModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1)
        x = self.model(x)
        return x

#
#
# class ExpressionModel(nn.Module):
#     def __init__(self):
#         super(ExpressionModel, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#
#         self.fc1 = nn.Linear(400, 128)
#         self.fc2 = nn.Linear(128, 2)
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#
#         x = x.view(x.size(0), -1)  # Flatten feature maps
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x