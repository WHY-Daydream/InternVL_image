"""
多模态数据加载器（V2版）
同时加载文本 + 图片
"""

import json
import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer


def build_text(item):
    """构建分类文本"""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")

    # 去掉标签列表（模型不需要看到这个，这是给人类的提示）
    if "以下是可以参考的分类标签为" in instruction:
        instruction = instruction.split("以下是可以参考的分类标签为")[0].strip()
    if "请直接只输出分类标签结果" in instruction:
        instruction = instruction.replace("请直接只输出分类标签结果，不需要其他多余的话。", "")

    text = instruction
    if inp:
        text += "\n" + inp
    return text


class MultimodalDatasetV2(Dataset):
    """图文数据集"""

    def __init__(self, data, config, tokenizer, is_train=True):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.is_train = is_train

        # 图像预处理
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

        # 收集有效索引
        self.valid_indices = []
        for i, item in enumerate(data):
            if item.get("image") and len(item["image"]) > 0:
                img_path = os.path.join(config.image_dir, item["image"][0])
                if os.path.exists(img_path):
                    self.valid_indices.append(i)

        print(f"  有效样本: {len(self.valid_indices)}/{len(data)}")

    def __len__(self):
        return len(self.valid_indices)

    def _load_image(self, img_name):
        img_path = os.path.join(self.config.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image)
        except:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.data[real_idx]

        # 文本
        text = build_text(item)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.config.max_text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 图片
        image = self._load_image(item["image"][0])
        label = self.config.class_to_idx[item["output"]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "id": item["id"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
    }


def build_dataloaders(config):
    with open(config.data_path, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"总数据: {len(all_data)}")

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder_path)

    random.seed(42)
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = [all_data[i] for i in indices[:train_end]]
    val_data = [all_data[i] for i in indices[train_end:val_end]]
    test_data = [all_data[i] for i in indices[val_end:]]

    print(f"划分: 训练 {len(train_data)}, 验证 {len(val_data)}, 测试 {len(test_data)}")

    train_dataset = MultimodalDatasetV2(train_data, config, tokenizer, is_train=True)
    val_dataset = MultimodalDatasetV2(val_data, config, tokenizer, is_train=False)
    test_dataset = MultimodalDatasetV2(test_data, config, tokenizer, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config_multimodal_v2 import Config
    conf = Config()
    train_loader, _, _ = build_dataloaders(conf)
    for batch in train_loader:
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"image: {batch['image'].shape}")
        print(f"label: {batch['label'].shape}")
        break
