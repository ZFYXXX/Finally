# dataset.py
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class QRCodeDataset(Dataset):
    def __init__(self, json_folder, transform=None):
        """
        初始化数据集。

        参数:
            json_folder (str): 存放所有 JSON 文件的文件夹路径。
            transform (callable, optional): 对图像进行预处理的函数。
        """
        self.transform = transform
        self.data = []

        # 遍历指定文件夹中的所有 .json 文件
        for filename in os.listdir(json_folder):
            if filename.endswith('.json'):
                json_path = os.path.join(json_folder, filename)
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    self.data.append(json_data)  # 使用 append 而不是 extend

        print(f"Total samples loaded: {len(self.data)}")
        if self.data:
            print(f"First sample: {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        参数:
            idx (int): 数据样本的索引。

        返回:
            tuple: (image, label) 形式的样本。
        """
        data_item = self.data[idx]

        # 提取图像路径
        img_path = data_item.get('path', None)
        if img_path is None:
            raise KeyError(f"Sample {idx} missing 'path' key.")

        # 提取标签
        outputs = data_item.get('outputs', {})
        objects = outputs.get('object', [])
        if not objects:
            raise ValueError(f"Sample {idx} has no objects in 'outputs'.")

        # 假设每个图像只有一个标签（第一个对象的名称）
        label_str = objects[0].get('name', None)
        if label_str is None:
            raise KeyError(f"Sample {idx} object's 'name' key missing.")

        try:
            label = int(label_str)
        except ValueError:
            raise ValueError(f"Sample {idx} has non-integer label: {label_str}")

        # 检查图像文件是否存在
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label