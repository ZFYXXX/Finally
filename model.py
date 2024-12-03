# model.py
import torch.nn as nn
import torch.nn.functional as F

class QRCodeCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(QRCodeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 假设输入图像大小为32x32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x