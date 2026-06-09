"""
文本数据加载器
从 ModelScope 数据中提取 instruction + input 文本
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def build_text(item):
    """从数据项构建分类用的文本"""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")

    # 去掉 instruction 末尾的标签列表（模型不需要看标签列表）
    # 找到 "以下是可以参考的分类标签为" 并截断
    if "以下是可以参考的分类标签为" in instruction:
        instruction = instruction.split("以下是可以参考的分类标签为")[0].strip()
    if "请直接只输出分类标签结果" in instruction:
        instruction = instruction.replace("请直接只输出分类标签结果，不需要其他多余的话。", "")

    # 拼接
    text = instruction
    if inp:
        text += "\n" + inp

    return text


class TextDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer

        # 过滤无效数据
        self.valid = []
        for item in data:
            text = build_text(item)
            if text.strip():
                self.valid.append(item)

        print(f"  有效样本: {len(self.valid)}/{len(data)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        item = self.valid[idx]
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

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.config.class_to_idx[item["output"]], dtype=torch.long
            ),
            "id": item["id"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
    }


def build_dataloaders(config):
    """构建训练/验证/测试数据加载器"""
    with open(config.data_path, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"总数据: {len(all_data)}")

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder_path)

    # 8:1:1 划分
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

    train_dataset = TextDataset(train_data, config, tokenizer)
    val_dataset = TextDataset(val_data, config, tokenizer)
    test_dataset = TextDataset(test_data, config, tokenizer)

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
    from config_text import Config
    conf = Config()
    train_loader, val_loader, test_loader = build_dataloaders(conf)
    for batch in train_loader:
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"labels: {batch['label'].shape}")
        break
