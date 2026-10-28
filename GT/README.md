# GT：GlycanGT 对比实验

本文件夹是**对比实验**用的，使用 [GlycanGT](https://huggingface.co/)（基于 TokenGT 的糖类图模型）提取嵌入，与本项目的 PUG（代码中称 W0）模型做公平比较。

> 🌐 English version: [README.en.md](./README.en.md)

> **命名说明**：本项目模型最初称 **W0**，后改名 **PUG**，代码仍沿用 W0 称呼。本文件夹的 `glycangt_classifier.py` 与 PUG 的 `glycanW0_classifier.py` 流程一致，仅数据来源不同。

## 目录结构

```
GT/
├── get_embeddings.py          # 用 GlycanGT 提取嵌入
└── glycangt_classifier.py     # 分类训练（与 PUG 的 glycanW0_classifier.py 一致，仅种子 42）
```

## 使用方法

### 1. 准备 GlycanGT 环境

GlycanGT 较新，本项目**未使用 GlycanGT 论文指定的包版本**，而是用本项目自身的依赖配置运行（已验证可正常运行 GlycanGT）。克隆 GlycanGT 仓库后在其根目录执行：

```bash
pip install .
```

### 2. 提取嵌入

将本文件夹的 `get_embeddings.py` 和 `split_files/`（按层级拆分的 WURCS CSV，来自 PUG 的 `MASK/94/tax(ML)/split_files/`；注意**不要用 `taxall/`**，其含不确定结构，GlycanGT 无法处理）一起放入 **GlycanGT 仓库根目录**，然后运行：

```bash
# 默认读取 ./split_files，输出到 ./split_files_processed
py -3.13 get_embeddings.py --mode split_files
```

运行后产生 `split_files_processed/`（processed 文件夹），内含追加 embedding 列的 CSV。

### 3. 分类训练

`glycangt_classifier.py` 的训练流程（CB Focal Loss、AdamW、MultiStageReduceLROnPlateau、EMA 早停、逐 epoch 阈值优化）与 PUG 的 `glycanW0_classifier.py` **完全一致**，用于公平比较。为避免重复，此处**只放了种子 42 的代码**：

```bash
py -3.13 glycangt_classifier.py --work_dir ./split_files_processed
```
