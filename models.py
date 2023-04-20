import torch
import torch.nn as nn


class CNN1d(nn.Module):
    def __init__(self):
        super().__init__()
        # bz*310
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
            nn.AvgPool1d(3)
        )
        # bz*16*152
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3200, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 4, dtype=float),
            nn.Softmax(dim=1)

        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.fc(X.view(X.shape[0], -1))
        return X
