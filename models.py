import torch.nn as nn
from dann import ReverseLayerF


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


class DANN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(3200, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 4, dtype=float),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(3200, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 2, dtype=float),
            nn.Softmax(dim=1)
        )

    def forward(self, x, alpha):
        feature = self.feature(x)
        feature = feature.view(feature.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output
