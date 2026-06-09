# InternVL_image — 多模态电商意图识别与图片分类

> BERT(文本) + ResNet50(图像) 融合模型，48 类电商场景分类

---

## 📋 项目概述

本项目实现了一个**多模态分类模型**，融合文本（BERT）和图像（ResNet50）特征，用于电商场景下的意图识别和图片分类（共 48 个类别）。支持三种推理模式：

- **多模态融合**（文本 + 图像）—— 默认推荐
- **纯文本**（仅 BERT）
- **纯图像**（仅 ResNet50）

---

## 🏗 项目结构

```
├── train_multimodal_v2.py       # V2 训练脚本（当前使用）
├── config_multimodal_v2.py      # V2 配置文件
├── data_multimodal_v2.py        # V2 数据加载器
├── model_multimodal_v2.py       # V2 模型定义
├── predict.py                   # 推理脚本（按置信度过滤 + CSV输出）
├── save_models/
│   └── multimodal_v2_best.pt    # V2 最佳模型权重（Git LFS）
├── dataset_images/              # 图片数据集（.gitignore 排除）
├── InternVL/                    # OpenGVLab/InternVL 项目（参考/拓展用）
├── .gitignore
└── README.md
```

---

## 🚀 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（推荐，也可用 CPU）

### 安装依赖

```bash
pip install torch torchvision transformers tqdm scikit-learn pillow
```

---

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

---

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

---

## 🔍 推理

```bash
python predict.py
```

功能：
- 加载训练好的最佳模型
- 对全部数据生成预测
- 可选择按置信度阈值过滤（默认 0.0）
- 输出 `predictions.csv`（包含 id、predict、confidence）

---

## 🧪 当前性能

最近一次 V2 训练结果（验证集最优）：

| 指标 | 值 |
|------|-----|
| F1 (加权) | **0.7948** |
| F1 (宏) | 0.7243 |
| Accuracy | 0.7993 |

---

# 🔬 深度诊断与优化方案

> 以下是对当前项目代码、数据和模型设计的全面诊断，以及提升准确率的优化路径。

## 一、当前问题分析

### 🐛 Bug：BatchNorm 冻结无效

**文件：** `model_multimodal_v2.py:27-29`

```python
self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
self._freeze_bn(resnet)  # ❌ 冻结的是原始 resnet，不是 image_encoder！
```

`nn.Sequential` 创建了一个**新对象**，调用 `_freeze_bn(resnet)` 冻结的是原始的 `resnet` 变量，而不是实际使用的 `self.image_encoder`。后果：

- ResNet50 的所有 BN 层在训练中**持续更新** running mean/var
- batch_size=32 时 BN 统计量不稳定，尤其尾部小类别 batch 方差大
- 训练后期 loss 震荡，难以收敛到更优值

**修复方式：** `self._freeze_bn(self.image_encoder)` 即可。

---

### 📊 数据严重不平衡

从数据集统计（17,442 条）：

```
最多: 下单过程中出现异常  1550条
最少: 个人信息页面            3条
比值: 516:1
```

| 指标 | 值 |
|------|-----|
| 类别数 | 48 |
| 平均每类 | 363 条 |
| 中位数 | 370 条 |
| 最少类 | 3 条（个人信息页面） |
| 尾部的 10 个类 | 均 < 100 条 |

**影响：** 当前使用 weighted CE Loss，但尾部类样本极少，模型几乎学不到有效特征，宏 F1（0.7243）远低于加权 F1（0.7948），说明尾部类严重拖后腿。

**优化方向：** 改用 **Focal Loss** 或 **Class-Balanced Loss**，让模型更关注难分/尾部样本。

---

### 🧹 文本噪音未清洗

文本中包含大量占位符，对分类无帮助：

```
instruction [610字]: 你是一个电商客服专家，请根据用户与客服的多轮对话判断用户的意图分类标签。
<用户与客服的对话 START>
用户: <image>          ← 占位符，不是有效文本
客服: ...
用户: <http>           ← 链接占位符
客服: ...
```

当前 `build_text()` 只去掉了标签列表，但 **没有清理** `<image>`、`<http>`、`<用户与客服的对话 START>` 等模板噪音。这些 tokens 占用 BERT 的 max_length，稀释了有效信息。

---

### 🔧 模型架构短板

**1. ResNet50 视觉编码能力有限**

- ResNet50 是 2015 年的架构，对 UI 截图、商品局部特写等**细粒度分类**场景特征提取能力不足
- 当前通过 `-[:-1]` 截断分类头做特征提取，但全量微调 ResNet50 的收益有限
- **替代方案：** ConvNeXt-Tiny、EfficientNet-V2、或 SWIN-Tiny，特征维度更高，细粒度分类更强

**2. 融合方式过于简单**

```
text_feat (512) ──┐
                  ├── Concat (1024) ──→ MLP ──→ 48类
image_feat (512) ─┘
```

简单拼接 + MLP 没有跨模态交互。改进方向：

- **Cross-Attention**：文本特征作为 Q，图像特征作为 K/V，让两者相互关注
- **Adaptive Gate**：当前门控是两个全局标量，改为逐样本的 SE-like 门控

**3. 无学习率调度**

全程 `lr=2e-5` 恒等，没有 cosine decay / warmup，后期无法精细收敛。

---

### 📏 max_length 截断问题

```
文本长度: 平均=426, p50=365, p90=587, p95=712
当前 max_length=512 → 约 5-10% 样本被截断
```

应增加到 **768** 或 **1024**（macbert-base 支持的 max 长度）。

---

## 二、优化路线图

按 **预期效果 / 改动量** 排序：

| 优先级 | 优化项 | 预期提升 | 改动文件 | 改动量 |
|--------|--------|----------|----------|--------|
| 🔴 P0 | **修复 BN 冻结 bug** | 训练稳定性↑ | `model_multimodal_v2.py:29` | 1行 |
| 🔴 P0 | **Focal Loss 替换 CrossEntropy** | 尾部类 F1↑↑ | `train_multimodal_v2.py` | +5行 |
| 🟠 P1 | **Cosine 余弦退火 LR 调度** | 收敛精度↑ | `train_multimodal_v2.py` | +3行 |
| 🟠 P1 | **清洗文本噪音**（去掉 \<image\> 等） | 文本质量↑ | `data_multimodal_v2.py` | +2行 |
| 🟠 P1 | **RandAugment / AutoAugment 增强** | 泛化能力↑ | `data_multimodal_v2.py` | +5行 |
| 🟠 P1 | **max_length 加大到 768** | 减少截断 | `config_multimodal_v2.py` | 1行 |
| 🟡 P2 | **ConvNeXt / EfficientNet 替换 ResNet50** | 图像特征↑↑ | `model_multimodal_v2.py` | +15行 |
| 🟡 P2 | **Cross-Attention 跨模态融合** | 模态交互↑↑ | `model_multimodal_v2.py` | +30行 |
| 🟡 P2 | **逐样本自适应门控**（SE-like） | 融合质量↑ | `model_multimodal_v2.py` | +10行 |
| 🟢 P3 | **Label Smoothing** | 泛化↑ | `train_multimodal_v2.py` | +1行 |
| 🟢 P3 | **梯度累积（accumulation）** | 大 batch 等效 | `train_multimodal_v2.py` | +2行 |
| 🟢 P3 | **MixUp / CutMix 增强** | 泛化↑ | `train_multimodal_v2.py` | +10行 |

### 预期目标

| 阶段 | 预期 F1 (加权) | 说明 |
|------|---------------|------|
| 当前 | ~0.79 | Baseline |
| P0 修复后 | ~0.80-0.81 | Bug 修复 + Focal Loss |
| P1 增强后 | ~0.82-0.84 | 文本清洗 + 增强 + LR 调度 |
| P2 架构升级 | ~0.85-0.88 | ConvNeXt + Cross-Attention |
| P3 精调 | ~0.86-0.89 | 全部优化叠加 |

---

## 📁 数据

数据集来自 ModelScope：
`smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios`

包含 17,442 条电商图文数据，涵盖 48 个意图/场景类别。

### 数据分布要点

- **两种任务混合：** 多轮对话意图分类（10,300 条）+ 纯图片场景分类（7,142 条）
- **instruction 格式不同：** 对话类包含 `<用户与客服的对话>` 模板，图片类包含标签列表
- **所有样本都有 1 张图片**，平均 1.06 张
- **input 字段全为空**

---

## 📈 日志

训练日志见 `train_v2_output.log`，包含每轮 Loss、验证 F1、早停记录。

---

## 📜 许可证

本项目基于 MIT 许可证。
