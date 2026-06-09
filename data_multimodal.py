"""
多模态数据加载器
支持文本 + 图片的联合加载
"""

import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer


class MultimodalDataset(Dataset):
    """多模态数据集：文本 + 图片 -> 分类标签"""

    def __init__(self, data, config, tokenizer, transform=None, is_train=True):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.is_train = is_train

        # 图像预处理
        if transform:
            self.transform = transform
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((config.image_size, config.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    normalize,
                ])

        # 统计有效数据（有图片的）
        self.valid_indices = []
        for i, item in enumerate(data):
            if item.get("image") and len(item["image"]) > 0:
                img_path = os.path.join(config.image_dir, item["image"][0])
                if os.path.exists(img_path):
                    self.valid_indices.append(i)

        print(f"数据集总样本: {len(data)}, 有图片的有效样本: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _build_text(self, item):
        """构建输入文本"""
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        # 拼接 instruction + input
        text = instruction
        if inp:
            text += "\n" + inp
        return text

    def _load_image(self, img_name):
        """加载并预处理图片"""
        img_path = os.path.join(self.config.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            # 如果图片加载失败，返回空白图
            print(f"图片加载失败: {img_path}, error: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.data[real_idx]

        # 1. 构建文本
        text = self._build_text(item)

        # 2. 文本编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.config.max_text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 3. 加载图片（取第一张）
        img_name = item["image"][0]
        image_tensor = self._load_image(img_name)

        # 4. 标签
        label = self.config.class_to_idx[item["output"]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "id": item["id"],
        }


def collate_fn(batch):
    """批量整理函数"""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
    }


def build_dataloaders(config):
    """构建训练/验证/测试数据加载器"""
    # 加载数据
    with open(config.data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.text_encoder_path)

    # 按 8:1:1 划分
    random.seed(42)
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]
    test_data = [all_data[i] for i in test_indices]

    print(f"数据划分: 训练 {len(train_data)}, 验证 {len(val_data)}, 测试 {len(test_data)}")

    # 创建数据集
    train_dataset = MultimodalDataset(train_data, config, tokenizer, is_train=True)
    val_dataset = MultimodalDataset(val_data, config, tokenizer, is_train=False)
    test_dataset = MultimodalDataset(test_data, config, tokenizer, is_train=False)

    # 创建加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config_multimodal import Config
    conf = Config()
    train_loader, val_loader, test_loader = build_dataloaders(conf)
    for batch in train_loader:
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"attention_mask: {batch['attention_mask'].shape}")
        print(f"image: {batch['image'].shape}")
        print(f"label: {batch['label'].shape}")
        print(f"id: {batch['id'][:3]}")
        break
