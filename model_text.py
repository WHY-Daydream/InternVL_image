"""
文本分类模型 - BERT 微调
用于电商对话意图 + 图片场景分类
"""

import torch
import torch.nn as nn
from transformers import BertModel


class TextClassifier(nn.Module):
    """纯文本BERT分类器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 加载预训练 BERT（已有权重！）
        self.bert = BertModel.from_pretrained(config.text_encoder_path)
        hidden_size = self.bert.config.hidden_size  # 768

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, config.num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    from config_text import Config
    conf = Config()
    model = TextClassifier(conf)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}, 可训练: {trainable:,} (全参数微调)")
