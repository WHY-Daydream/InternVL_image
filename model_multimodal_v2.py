"""
多模态分类模型：BERT(文本) + ResNet50(图像) 融合
用于电商图片场景分类 + 对话意图分类（48类）
"""

import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models


class MultimodalClassifierV2(nn.Module):
    """图文融合分类器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ===== 文本编码器 (BERT, 有预训练) =====
        self.text_encoder = BertModel.from_pretrained(config.text_encoder_path)
        self.text_dim = self.text_encoder.config.hidden_size  # 768

        # ===== 图像编码器 (ResNet50, ImageNet预训练) =====
        resnet = models.resnet50(weights=config.resnet_weights)
        self.image_dim = resnet.fc.in_features  # 2048
        # 去掉原始分类头，保留特征提取部分
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # 冻结所有BN层（保持预训练统计量）
        self._freeze_bn(resnet)

        # ===== 模态投影到统一维度 =====
        proj_dim = 512
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ===== 融合分类器 =====
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim // 2, config.num_classes),
        )

        # ===== 模态门控权重 =====
        self.text_gate = nn.Parameter(torch.tensor(0.5))
        self.image_gate = nn.Parameter(torch.tensor(0.5))

    def _freeze_bn(self, model):
        """冻结BN层的running stats"""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, images):
        # === 文本特征 ===
        text_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_feat = text_out.last_hidden_state[:, 0, :]  # [CLS]
        text_feat = self.text_proj(text_feat)

        # === 图像特征 ===
        image_feat = self.image_encoder(images)  # (B, 2048, 1, 1)
        image_feat = image_feat.flatten(1)  # (B, 2048)
        image_feat = self.image_proj(image_feat)

        # === 门控融合 ===
        tg = torch.sigmoid(self.text_gate)
        ig = torch.sigmoid(self.image_gate)
        fused = torch.cat([tg * text_feat, ig * image_feat], dim=-1)

        # === 分类 ===
        logits = self.classifier(fused)
        return logits


class TextOnlyV2(nn.Module):
    """纯文本基线"""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.text_encoder_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, images=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class ImageOnlyV2(nn.Module):
    """纯图像基线"""
    def __init__(self, config):
        super().__init__()
        resnet = models.resnet50(weights=config.resnet_weights)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, config.num_classes),
        )

    def forward(self, images, input_ids=None, attention_mask=None):
        feat = self.encoder(images).flatten(1)
        return self.classifier(feat)


if __name__ == "__main__":
    from config_multimodal_v2 import Config
    conf = Config()
    model = MultimodalClassifierV2(conf)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}, 可训练: {trainable:,}")
