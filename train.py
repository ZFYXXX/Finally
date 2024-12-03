# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import QRCodeDataset
from model import QRCodeCNN  # 确保您有定义 QRCodeCNN 模型
from torchvision import transforms

# 超参数
batch_size = 32
learning_rate = 0.001
num_epochs = 20
num_classes = 10  # 根据实际分类数量调整

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # 可选：添加数据增强
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# 加载数据集
json_folder = 'outputs/train_data'  # 根据实际路径调整
full_dataset = QRCodeDataset(json_folder=json_folder, transform=transform)

# 划分训练集和验证集
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QRCodeCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算验证集损失和准确率
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], '  
          f'Loss: {running_loss/len(train_loader):.4f}, '  
          f'Val Loss: {val_loss/len(val_loader):.4f}, '  
          f'Val Acc: {100 * correct / total:.2f}%')

# 保存模型权重
torch.save(model.state_dict(), 'qrcode_model.pt')