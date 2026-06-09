"""
预测脚本 - 生成比赛提交格式的 CSV 文件
输出: id, predict
"""

import os
import json
import csv
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_multimodal import Config
from model_multimodal import MultimodalClassifier
from data_multimodal import MultimodalDataset, collate_fn
from eval_utils import get_predictions


def predict(input_json=None, output_csv=None):
    """
    对数据进行预测并生成CSV提交文件
    Args:
        input_json: 输入数据路径 (默认使用配置中的data_path)
        output_csv: 输出CSV路径 (默认使用配置中的prediction_path)
    """
    conf = Config()

    if input_json is None:
        input_json = conf.data_path
    if output_csv is None:
        output_csv = conf.prediction_path

    device = conf.device
    print(f"设备: {device}")

    # ========== 加载模型 ==========
    print(f"加载模型: {conf.model_save_path}")
    if not os.path.exists(conf.model_save_path):
        print(f"❌ 模型文件不存在: {conf.model_save_path}")
        print("请先运行 train_multimodal.py 进行训练")
        return

    model = MultimodalClassifier(conf).to(device)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=device))
    model.eval()
    print(f"模型加载成功, 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 加载数据 ==========
    print(f"加载数据: {input_json}")
    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据总量: {len(data)}")

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(conf.text_encoder_path)

    dataset = MultimodalDataset(
        data, conf, tokenizer, is_train=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # ========== 预测 ==========
    print("开始预测...")
    predictions = get_predictions(model, dataloader, device, conf)

    # ========== 保存CSV ==========
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "predict"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\n✅ 预测完成!")
    print(f"结果已保存到: {output_csv}")
    print(f"共 {len(predictions)} 条预测")

    # 预览前5条
    print("\n预览 (前5条):")
    for p in predictions[:5]:
        print(f"  id={p['id']}, predict={p['predict']}")


if __name__ == "__main__":
    predict()
