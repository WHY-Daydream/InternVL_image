"""
多模态意图识别模型
融合 BERT (文本) + ViT (图像) 特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import timm


class MultimodalClassifier(nn.Module):
    """多模态分类器：文本 + 图像融合"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ========== 文本编码器 (BERT) ==========
        self.text_encoder = BertModel.from_pretrained(config.text_encoder_path)
        self.text_hidden_size = self.text_encoder.config.hidden_size  # 768

        # ========== 图像编码器 ==========
        # 由于网络不可达，使用pretrained=False（随机初始化）
        # 如有网络，可改为pretrained=True加载预训练权重
        try:
            self.image_encoder = timm.create_model(
                config.image_encoder_name,
                pretrained=False,
                num_classes=0,
            )
        except Exception:
            # 回退到 ResNet-18
            import torchvision.models as models
            self.image_encoder = models.resnet18(weights=None)
            # 去掉最后的全连接层
            self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
            config.image_feat_dim = 512
        self.image_hidden_size = config.image_feat_dim

        # ========== 特征投影层 ==========
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_size, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_hidden_size, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ========== 跨模态注意力融合 ==========
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # ========== 融合后处理 ==========
        self.fusion_ffn = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ========== 分类器 ==========
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, config.num_classes),
        )

        # ========== 可训练的模态权重 ==========
        self.text_weight = nn.Parameter(torch.tensor(0.7))
        self.image_weight = nn.Parameter(torch.tensor(0.3))

        self._init_weights()

    def _init_weights(self):
        """初始化投影层和分类器的权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids, attention_mask, images):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            images: (batch_size, 3, H, W)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.size(0)

        # ========== 文本编码 ==========
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_cls = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_feat = self.text_proj(text_cls)  # (B, fusion_dim)

        # ========== 图像编码 ==========
        image_feat = self.image_encoder(images)  # (B, feat_dim) or (B, feat_dim, 1, 1)
        if image_feat.dim() == 4:  # ResNet输出 (B, C, 1, 1)
            image_feat = image_feat.flatten(1)  # (B, C)
        image_feat = self.image_proj(image_feat)  # (B, fusion_dim)

        # ========== 跨模态融合 ==========
        # 文本作为query, 图像作为key/value
        text_feat_attn = text_feat.unsqueeze(1)  # (B, 1, D)
        image_feat_attn = image_feat.unsqueeze(1)  # (B, 1, D)

        # 交叉注意力: 文本关注图像
        fused, _ = self.cross_attn(
            query=text_feat_attn,
            key=image_feat_attn,
            value=image_feat_attn,
        )
        fused = fused.squeeze(1)  # (B, D)

        # 残差连接 + FFN
        # 权重融合: 文本特征 + 跨模态融合特征
        text_w = torch.sigmoid(self.text_weight)
        image_w = torch.sigmoid(self.image_weight)

        combined = torch.cat([
            text_w * text_feat + fused,
            image_w * image_feat,
        ], dim=-1)

        fused = self.fusion_ffn(combined)  # (B, fusion_dim)

        # ========== 分类 ==========
        logits = self.classifier(fused)

        return logits

    def get_features(self, input_ids, attention_mask, images):
        """提取特征向量（用于分析）"""
        with torch.no_grad():
            # 文本
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_cls = text_outputs.last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_cls)

            # 图像
            image_feat = self.image_encoder(images)
            image_feat = self.image_proj(image_feat)

            return text_feat, image_feat


class TextOnlyBaseline(nn.Module):
    """纯文本基线模型（用于对比）"""

    def __init__(self, config):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(config.text_encoder_path)
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size, config.num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits


class ImageOnlyBaseline(nn.Module):
    """纯图像基线模型（用于对比）"""

    def __init__(self, config):
        super().__init__()
        try:
            self.image_encoder = timm.create_model(
                config.image_encoder_name, pretrained=False, num_classes=0
            )
        except Exception:
            import torchvision.models as models
            self.image_encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.image_feat_dim, config.num_classes),
        )

    def forward(self, images):
        features = self.image_encoder(images)
        if features.dim() == 4:
            features = features.flatten(1)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    from config_multimodal import Config
    conf = Config()

    model = MultimodalClassifier(conf)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 512)),
        "attention_mask": torch.ones(2, 512),
        "image": torch.randn(2, 3, 224, 224),
    }
    logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
    print(f"输出形状: {logits.shape}")  # (2, 48)
