"""
评估工具函数
计算准确率、精确率、召回率、F1
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device, config=None):
    """
    在给定数据加载器上评估模型
    Returns:
        report: 分类报告文本
        weighted_f1: 加权 F1 分数
        accuracy: 准确率
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, images)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # 如果提供了config，生成详细报告
    report = None
    if config:
        target_names = config.class_list
        report = classification_report(
            all_labels, all_preds,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )

    return report, weighted_f1, accuracy, macro_f1


def get_predictions(model, dataloader, device, config):
    """获取预测结果（用于提交）"""
    model.eval()
    predictions = []

    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        logits = model(input_ids, attention_mask, images)
        preds = torch.argmax(logits, dim=-1)

        for i, pred_idx in enumerate(preds.cpu().numpy()):
            predictions.append({
                "id": batch["id"][i],
                "predict": config.class_list[pred_idx],
            })

    return predictions


if __name__ == "__main__":
    print("评估工具模块加载成功")
