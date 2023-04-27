import torch
import torch.nn as nn


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

    def forward(self, f):
        f = self.conv1(f)
        f = self.conv2(f)
        f = self.fc1(f.view(f.shape[0], -1))
        x = self.fc2(f)
        if self.training:
            return x, f
        else:
            return x


# Define the CNN model
class EEG_CNN(nn.Module):
    def __init__(self, num_classes):
        super(EEG_CNN, self).__init__()
        self.expand_features = nn.Linear(62 * 5, 16 * 16)  # Expanding the features
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

        self.features_dim = 64

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.}]

    def forward(self, x):
        # Expanding the features using a fully connected layer
        x = x.view(-1, 62 * 5)
        x = self.relu(self.expand_features(x))
        x = x.view(-1, 1, 16, 16)

        # Apply the convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.bn3(x)

        # Flatten and apply the fully connected layers
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        f = self.dropout(x)
        x = self.fc2(f)
        x = self.softmax(x)
        if self.training:
            return x, f
        else:
            return x