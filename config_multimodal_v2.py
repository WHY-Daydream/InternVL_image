"""
多模态分类配置文件
"""

import json
import torch
from collections import Counter
import os


class Config:
    def __init__(self):
        self.data_path = "/mnt/workspace/.cache/modelscope/datasets/smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios/all.json"
        self.image_dir = "/mnt/workspace/image/dataset_images"

        # 文本编码器
        self.text_encoder_name = "hfl/chinese-macbert-base"
        self.text_encoder_path = "/mnt/workspace/.cache/modelscope/models/hfl/chinese-macbert-base"

        # 图像编码器 - ResNet50 with ImageNet pretrained weights
        import torchvision.models as models
        self.resnet_weights = 'IMAGENET1K_V2'

        # 训练参数
        self.num_epochs = 20
        self.batch_size = 32  # ResNet50 比 ViT 轻量，批次可以大
        self.learning_rate = 2e-5
        self.image_lr = 1e-4  # 图像编码器用稍大学习率（部分冻结）
        self.text_lr = 2e-5
        self.weight_decay = 0.01
        self.max_text_length = 512  # BERT上限，保留完整对话
        self.image_size = 224
        self.warmup_ratio = 0.1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 类别
        self.class_list = self._extract_classes()
        self.num_classes = len(self.class_list)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        self.save_dir = "/mnt/workspace/image/save_models"
        self.model_save_path = os.path.join(self.save_dir, "multimodal_v2_best.pt")
        self.prediction_path = "/mnt/workspace/image/predictions.csv"

        os.makedirs(self.save_dir, exist_ok=True)

    def _extract_classes(self):
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        return sorted(set(item["output"] for item in data))

    def get_label_weights(self):
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        counter = Counter(item["output"] for item in data)
        total = len(data)
        n = len(self.class_list)
        weights = [total / (n * counter[c]) for c in self.class_list]
        return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    conf = Config()
    print(f"类别数: {conf.num_classes}, 设备: {conf.device}")
