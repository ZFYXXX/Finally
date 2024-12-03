# export_onnx.py
import torch
from model import QRCodeCNN

# 初始化模型并加载权重
num_classes = 10  # 根据实际调整
model = QRCodeCNN(num_classes=num_classes)
model.load_state_dict(torch.load('qrcode_model.pt'))
model.eval()

# 创建一个示例输入
dummy_input = torch.randn(1, 3, 32, 32)

# 导出为ONNX
torch.onnx.export(model, dummy_input, 'qrcode_model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)