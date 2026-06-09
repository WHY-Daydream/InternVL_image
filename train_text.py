"""
文本分类训练脚本
使用 BERT 微调进行意图识别
"""

import os
import time
import json
import csv
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

from config_text import Config
from data_text import build_dataloaders, TextDataset, collate_fn, build_text
from model_text import TextClassifier
from transformers import BertTokenizer


@torch.no_grad()
def evaluate(model, dataloader, device, config=None):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    report = None
    if config:
        report = classification_report(
            all_labels, all_preds, target_names=config.class_list,
            digits=4, zero_division=0,
        )

    return report, weighted_f1, accuracy, macro_f1


def train():
    conf = Config()
    os.makedirs(conf.save_dir, exist_ok=True)

    print(f"设备: {conf.device}")
    print(f"类别数量: {conf.num_classes}")

    # 数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = build_dataloaders(conf)

    # 模型
    print("\n构建模型...")
    model = TextClassifier(conf).to(conf.device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}, 可训练: {trainable:,}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    # 学习率预热
    total_steps = len(train_loader) * conf.num_epochs
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=conf.warmup_steps)

    # 损失函数（加权）
    label_weights = conf.get_label_weights().to(conf.device)
    criterion = nn.CrossEntropyLoss(weight=label_weights)

    # 训练
    print("\n开始训练...")
    best_f1 = 0.0
    no_improve = 0
    patience = 5

    for epoch in range(conf.num_epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.num_epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(conf.device)
            attention_mask = batch["attention_mask"].to(conf.device)
            labels = batch["label"].to(conf.device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证
        report, weighted_f1, accuracy, macro_f1 = evaluate(
            model, val_loader, conf.device, conf
        )

        elapsed = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | {elapsed:.0f}s | Loss: {avg_loss:.4f}")
        print(f"验证 -> F1(加权): {weighted_f1:.4f} | F1(宏): {macro_f1:.4f} | Acc: {accuracy:.4f}")

        # 保存最佳模型
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            torch.save(model.state_dict(), conf.model_save_path)
            print(f"✅ 保存最佳模型 (F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"F1未提升 ({no_improve}/{patience})")

        if no_improve >= patience:
            print("🛑 早停触发")
            break

    # 测试集评估
    print("\n" + "=" * 50)
    print("测试集评估...")
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    report, test_f1, test_acc, test_macro = evaluate(
        model, test_loader, conf.device, conf
    )
    print(f"测试 -> F1(加权): {test_f1:.4f} | F1(宏): {test_macro:.4f} | Acc: {test_acc:.4f}")
    print(f"\n报告:\n{report}")

    # 生成预测CSV
    print("\n生成预测结果...")
    with open(conf.data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(conf.text_encoder_path)
    all_dataset = TextDataset(all_data, conf, tokenizer)
    all_loader = DataLoader(
        all_dataset, batch_size=conf.batch_size * 2,
        shuffle=False, collate_fn=collate_fn, num_workers=4,
    )

    model.eval()
    predictions = []
    for batch in tqdm(all_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(conf.device)
        attention_mask = batch["attention_mask"].to(conf.device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        for i, p in enumerate(preds.cpu().numpy()):
            predictions.append({"id": batch["id"][i], "predict": conf.class_list[p]})

    with open(conf.prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "predict"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"结果保存至: {conf.prediction_path} ({len(predictions)} 条)")

    return test_f1


if __name__ == "__main__":
    train()
