# imm(ML) — 免疫原性（immunogenicity）下游任务

基于主文件夹预训练得到的**第 94 轮最优 checkpoint**，对糖类免疫原性数据进行嵌入提取与分类训练。

> 🌐 English version: [README.en.md](./README.en.md)

## 目录结构

```
imm(ML)/
├── masktrain_windows_313.py        # ⬅ 从主文件夹移入（外部）
├── wurcs_parser_reverse_edge.py     # ⬅ 从主文件夹移入（外部）
├── get_embeddings_W0.py            # 提取 1027 维嵌入
├── glycanW0_classifier.py          # 免疫原性分类训练
├── embedding_vocab.json            # ⬅ 从主文件夹移入（外部）
├── checkpoints/
│   └── best_model_mlm.pt          # ⬅ 从主文件夹移入（94 轮预训练权重，外部）
├── train/
│   ├── imm_stata.csv               # ⬅ 外部移入：免疫原性原始数据
│   ├── imm_stata_with_embeddings.csv  # 由 get_embeddings_W0.py 生成
│   ├── classification_results/     # 由 glycanW0_classifier.py 生成
│   └── 数量统计.py                  # 统计 CSV 行数与标签数
```

> ⬅ 标记的文件为**从外部移入**：`masktrain_windows_313.py`、`wurcs_parser_reverse_edge.py`、`embedding_vocab.json` 与 `checkpoints/best_model_mlm.pt` 来自主文件夹 `MASK/`，`train/imm_stata.csv` 为来自 `downstream/imme/` 的免疫原性数据集。这些文件均已**随仓库上传**（最大为约 34 MB 的预训练权重，低于 GitHub 100 MB 单文件限制），克隆后可直接运行，无需手动放入。

## 数据说明

`train/imm_stata.csv` 每行一条糖结构，含 `glycan_id`、`glycan`（IUPAC）、`species`、`immunogenicity`（0/1）、`wurcs` 等列。

## 运行顺序

```bash
# 前置：确保 embedding_vocab.json 与 checkpoints/best_model_mlm.pt 已放入本目录

# 1. 提取嵌入 → 生成 train/imm_stata_with_embeddings.csv
py -3.13 get_embeddings_W0.py \
    --input_dir ./train \
    --output_dir ./train \
    --vocab_path embedding_vocab.json \
    --checkpoint_path checkpoints/best_model_mlm.pt

# 2. 分类训练 → 产出 train/classification_results/（training_log.txt、混淆矩阵、模型等）
#    默认读取 train/imm_stata_with_embeddings.csv，标签列 immunogenicity
py -3.13 glycanW0_classifier.py
```
