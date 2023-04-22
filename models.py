import torch.nn as nn
from typing import List, Dict


class CNN1d(nn.Module):
    def __init__(self):
        super().__init__()
        # bz*310
        self.features_dim = 128
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(5,), stride=(1, )),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(3)
        )
        # bz*16*152
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=(3, ), stride=(1, )),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3200, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 4),
            nn.Softmax(dim=1),
        )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]

    def forward(self, f):
        f = self.conv1(f)
        f = self.conv2(f)
        f = self.fc1(f.view(f.shape[0], -1))
        x = self.fc2(f)
        if self.training:
            return x, f
        else:
            return x
