"""
纯文本分类配置文件
"""

import torch
import json
from collections import Counter
import os


class Config:
    def __init__(self):
        self.data_path = "/mnt/workspace/.cache/modelscope/datasets/smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios/all.json"

        # 文本编码器（有预训练权重）
        self.text_encoder_name = "hfl/chinese-macbert-base"
        self.text_encoder_path = "/mnt/workspace/.cache/modelscope/models/hfl/chinese-macbert-base"

        # 训练参数
        self.num_epochs = 20
        self.batch_size = 24  # 减小以容纳更长文本
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.max_text_length = 256  # 平均文本长度356，保留大部分信息
        self.warmup_steps = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 类别
        self.class_list = self._extract_classes()
        self.num_classes = len(self.class_list)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        self.save_dir = "/mnt/workspace/image/save_models"
        self.model_save_path = os.path.join(self.save_dir, "text_bert_best.pt")
        self.prediction_path = "/mnt/workspace/image/predictions.csv"

    def _extract_classes(self):
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        return sorted(set(item["output"] for item in data))

    def get_label_weights(self):
        """计算类别权重（用于处理不平衡）"""
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        counter = Counter(item["output"] for item in data)
        total = len(data)
        n = len(self.class_list)
        weights = [total / (n * counter[c]) for c in self.class_list]
        return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    conf = Config()
    print(f"类别数: {conf.num_classes}")
    print(f"设备: {conf.device}")
