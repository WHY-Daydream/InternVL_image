# InternVL_image — 多模态电商意图识别与图片分类

> BERT(文本) + ResNet50(图像) 融合模型，48 类电商场景分类

## 📋 项目概述

本项目实现了一个**多模态分类模型**，融合文本（BERT）和图像（ResNet50）特征，用于电商场景下的意图识别和图片分类（共 48 个类别）。支持三种推理模式：

- **多模态融合**（文本 + 图像）—— 默认推荐
- **纯文本**（仅 BERT）
- **纯图像**（仅 ResNet50）

## 🏗 项目结构

```
├── train_multimodal_v2.py       # V2 训练脚本（当前使用）
├── config_multimodal_v2.py      # V2 配置文件
├── data_multimodal_v2.py        # V2 数据加载器
├── model_multimodal_v2.py       # V2 模型定义
├── predict.py                   # 推理脚本（按置信度过滤 + CSV输出）
├── save_models/
│   └── multimodal_v2_best.pt    # V2 最佳模型权重
├── dataset_images/              # 图片数据集（.gitignore 排除）
├── InternVL/                    # OpenGVLab/InternVL 项目（参考/拓展用）
├── .gitignore
└── README.md
```

## 🚀 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（推荐，也可用 CPU）

### 安装依赖

```bash
pip install torch torchvision transformers tqdm scikit-learn pillow
```

## 🏋️ 训练

```bash
python train_multimodal_v2.py
```

训练参数见 `config_multimodal_v2.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | 20 | 训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 2e-5 | 融合层学习率 |
| `text_lr` | 2e-5 | BERT 学习率 |
| `image_lr` | 1e-4 | ResNet50 学习率 |
| `max_text_length` | 512 | 文本最大长度 |
| `patience` | 5 | 早停耐心值 |

训练过程会自动：
1. 按 8:1:1 划分训练/验证/测试集
2. 每轮评估验证集 F1，保存最佳模型
3. F1 连续 5 轮不提升则触发早停
4. 训练完成后输出测试集分类报告
5. 生成 `predictions.csv` 预测结果

## 📊 模型架构

```
文本 (BERT) ──→ [CLS] ──→ Linear(768→512) ──→ ┐
                                               ├──→ Concat ──→ MLP ──→ 48类
图像 (ResNet50) ──→ AvgPool ──→ Linear(2048→512) ─┘
```

- 文本编码器：`hfl/chinese-macbert-base`（冻结预训练权重）
- 图像编码器：ResNet50（ImageNet 预训练，BN 层冻结）
- 门控融合：可学习的 sigmoid 门控权重
- 分类头：3 层 MLP（512→256→48），含 Dropout

## 🔍 推理

```bash
python predict.py
```

功能：
- 加载训练好的最佳模型
- 对全部数据生成预测
- 可选择按置信度阈值过滤（默认 0.0）
- 输出 `predictions.csv`（包含 id、predict、confidence）

## 📈 日志

- `train_v2_output.log` — 最近一次 V2 训练的完整日志
- 包含每轮的 Loss、验证 F1、早停记录
- 可自行查看训练过程中的指标变化

## 📁 数据

数据集来自 ModelScope：
`smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios`

包含 17,442 条电商图文数据，涵盖 48 个意图/场景类别。

## 🧪 评估指标

- **F1 (加权)** — 主要优化目标
- **F1 (宏)** — 各类别平均
- **Accuracy** — 准确率
- 训练完成自动输出详细分类报告

## 📜 许可证

本项目基于 MIT 许可证。
