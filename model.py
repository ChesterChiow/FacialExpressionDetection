import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.name = "cnn"

        # ==== Block 1 ====
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # ==== Block 2 ====
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout)

        # ==== Block 3 ====
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # ==== Block 4 ====
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(dropout)

        # ==== Block 5 ====
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 48 -> 24 -> 12 -> 6 -> 3 -> 1
        self.flatten_dim = 512 * 1 * 1

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pool1(self.bn1(F.relu(self.conv1(x))))

        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool3(self.bn3(F.relu(self.conv3(x))))

        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        x = self.drop4(x)

        x = self.pool5(self.bn5(F.relu(self.conv5(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)

        return x