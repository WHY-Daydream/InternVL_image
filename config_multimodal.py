"""
多模态意图识别模型配置文件
融合文本(BERT)和图像(ViT)特征，用于电商场景图片分类 + 对话意图分类
"""

import torch
import json
from collections import Counter
import os


class Config:
    def __init__(self):
        # ========== 路径配置 ==========
        self.data_path = "/mnt/workspace/.cache/modelscope/datasets/smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios/all.json"
        self.image_dir = "/mnt/workspace/image/dataset_images"

        # ========== 模型配置 ==========
        # 文本编码器
        self.text_encoder_name = "hfl/chinese-macbert-base"
        self.text_encoder_path = "/mnt/workspace/.cache/modelscope/models/hfl/chinese-macbert-base"

        # 图像编码器
        self.image_encoder_name = "vit_base_patch16_224"
        self.image_feat_dim = 768  # ViT-B/16 输出维度

        # 融合层
        self.fusion_dim = 512
        self.dropout = 0.3

        # ========== 训练配置 ==========
        self.num_epochs = 10
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.text_lr = 2e-5
        self.image_lr = 1e-4
        self.fusion_lr = 1e-3
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_text_length = 512
        self.image_size = 224
        self.gradient_accumulation_steps = 2

        # ========== 设备 ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== 类别信息 ==========
        self.class_list = self._extract_classes()
        self.num_classes = len(self.class_list)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # ========== 保存路径 ==========
        self.save_dir = "/mnt/workspace/image/save_models"
        self.model_save_path = os.path.join(self.save_dir, "multimodal_best.pt")
        self.prediction_path = "/mnt/workspace/image/predictions.csv"

    def _extract_classes(self):
        """从数据中提取所有类别"""
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        classes = sorted(set(item["output"] for item in data))
        return classes

    def get_label_weights(self):
        """计算类别权重（用于加权损失）"""
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        counter = Counter(item["output"] for item in data)
        total = len(data)
        n_classes = len(self.class_list)
        weights = [total / (n_classes * counter[c]) for c in self.class_list]
        return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    conf = Config()
    print(f"设备: {conf.device}")
    print(f"类别数量: {conf.num_classes}")
    print(f"总数据量: 17442")
