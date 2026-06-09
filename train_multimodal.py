"""
多模态模型训练脚本
融合文本(BERT) + 图像(ViT) 进行电商意图识别
"""

import os
import sys
import time
import json
import csv
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer

from config_multimodal import Config
from data_multimodal import build_dataloaders, MultimodalDataset, collate_fn
from model_multimodal import MultimodalClassifier
from eval_utils import evaluate, get_predictions


def train():
    conf = Config()
    print(f"设备: {conf.device}")
    print(f"类别数量: {conf.num_classes}")
    if conf.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========== 构建数据加载器 ==========
    print("\n" + "=" * 60)
    print("加载数据...")
    train_loader, val_loader, test_loader = build_dataloaders(conf)

    # ========== 构建模型 ==========
    print("\n" + "=" * 60)
    print("构建模型...")
    model = MultimodalClassifier(conf).to(conf.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 打印显存占用
    if conf.device.type == "cuda":
        print(f"初始显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ========== 优化器 ==========
    # 分层学习率
    text_params = []
    image_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder" in name:
            text_params.append(param)
        elif "image_encoder" in name:
            image_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {"params": text_params, "lr": conf.text_lr},
        {"params": image_params, "lr": conf.image_lr},
        {"params": other_params, "lr": conf.fusion_lr},
    ], weight_decay=conf.weight_decay)

    # ========== 损失函数 ==========
    label_weights = conf.get_label_weights().to(conf.device)
    criterion = nn.CrossEntropyLoss(weight=label_weights)

    # ========== 学习率调度器 ==========
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 2, T_mult=2, eta_min=1e-6
    )

    # ========== 创建保存目录 ==========
    os.makedirs(conf.save_dir, exist_ok=True)

    # ========== 训练循环 ==========
    print("\n" + "=" * 60)
    print("开始训练...")

    best_f1 = 0.0
    best_epoch = 0
    global_step = 0
    no_improve_count = 0
    patience = 5  # 早停

    for epoch in range(conf.num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(conf.device)
            attention_mask = batch["attention_mask"].to(conf.device)
            images = batch["image"].to(conf.device)
            labels = batch["label"].to(conf.device)

            # 前向传播
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            batch_count += 1
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = total_loss / batch_count
        epoch_time = time.time() - epoch_start

        # ========== 验证 ==========
        report, weighted_f1, accuracy, macro_f1 = evaluate(
            model, val_loader, conf.device, conf
        )

        print(f"\nEpoch {epoch+1} | 耗时: {epoch_time:.0f}s | 损失: {avg_loss:.4f}")
        print(f"验证集 -> F1(加权): {weighted_f1:.4f} | F1(宏): {macro_f1:.4f} | 准确率: {accuracy:.4f}")

        if conf.device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"最大显存: {mem:.2f} GB")

        # 保存最佳模型
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), conf.model_save_path)
            print(f"✅ 保存最佳模型 (F1={best_f1:.4f}) -> {conf.model_save_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"F1未提升 (连续 {no_improve_count}/{patience})")

        # 早停
        if no_improve_count >= patience:
            print(f"🛑 早停触发，在第 {epoch+1} 轮停止")
            break

    # ========== 测试集评估 ==========
    print("\n" + "=" * 60)
    print(f"加载最佳模型 (Epoch {best_epoch}, F1={best_f1:.4f}) 评估测试集...")
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    report, test_f1, test_acc, test_macro = evaluate(
        model, test_loader, conf.device, conf
    )
    print(f"\n测试集结果:")
    print(f"  F1(加权): {test_f1:.4f}")
    print(f"  F1(宏): {test_macro:.4f}")
    print(f"  准确率: {test_acc:.4f}")
    print(f"\n详细分类报告:")
    print(report)

    # ========== 生成预测CSV ==========
    print("\n" + "=" * 60)
    print("生成预测结果...")

    # 使用全部数据预测
    all_dataset = MultimodalDataset(
        json.load(open(conf.data_path)), conf,
        BertTokenizer.from_pretrained(conf.text_encoder_path),
        is_train=False,
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=conf.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    predictions = get_predictions(model, all_loader, conf.device, conf)

    # 保存CSV
    import csv
    with open(conf.prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "predict"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"预测结果已保存到: {conf.prediction_path}")
    print(f"共 {len(predictions)} 条预测")

    print("\n" + "=" * 60)
    print("🎉 训练完成!")

    return best_f1


if __name__ == "__main__":
    train()
