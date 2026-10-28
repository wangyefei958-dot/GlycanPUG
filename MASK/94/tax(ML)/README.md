# tax(ML) — 物种分类（taxonomy）下游任务

基于主文件夹预训练得到的**第 94 轮最优 checkpoint**，对糖类按物种分类学 8 个层级（domain → species）进行嵌入提取与多种子（1–10）分类训练。

> 🌐 English version: [README.en.md](./README.en.md)

## 目录结构

```
tax(ML)/
├── masktrain_windows_313.py        # ⬅ 从主文件夹移入（外部）
├── wurcs_parser_reverse_edge.py     # ⬅ 从主文件夹移入（外部）
├── get_embeddings_W0.py            # 提取 1027 维嵌入
├── glycanW0_classifier.py          # 物种分类训练（顶层副本，finaltrain 内各种子有自己的副本）
├── 总结.py                          # 汇总多 seed 的 F1/AUC 指标
├── embedding_vocab.json            # ⬅ 从主文件夹移入（外部）
├── GlycanW0_metrics.csv            # 由 总结.py 生成
├── checkpoints/
│   └── best_model_mlm.pt          # ⬅ 从主文件夹移入（94 轮预训练权重，外部）
├── split_files/                     # ⬅ 外部移入：按层级拆分的 WURCS CSV
│   ├── domain.csv / kingdom.csv / phylum.csv / class.csv
│   ├── order.csv / family.csv / genus.csv / species.csv
│   └── 数量统计.py                  # 统计各 CSV 行数与标签数
├── split_files_processed_wurcs/     # 由 get_embeddings_W0.py 生成（嵌入 CSV）
├── finaltrain/                      # 多种子最终训练（种子 1–10）
│   ├── 新建 文本文档.py             # 批量运行各种子目录下的 glycanW0_classifier.py
│   ├── 1/                           # 种子 1
│   │   ├── glycanW0_classifier.py   # 与顶层相同，仅 SEED 不同
│   │   └── split_files_processed_wurcs/  # ⬅ 手动移入（从顶层复制）
│   │       └── training_results_<level>/  # 各层级训练产物
│   ├── 2/ … 10/                     # 种子 2–10（结构同上）
│   └── __pycache__/
```

> ⬅ 标记的文件为**从外部移入**：`masktrain_windows_313.py`、`wurcs_parser_reverse_edge.py`、`embedding_vocab.json` 与 `checkpoints/best_model_mlm.pt` 来自主文件夹 `MASK/`，`split_files/*.csv` 为按物种层级拆分的外部数据。`finaltrain/<seed>/split_files_processed_wurcs/` 由顶层手动复制移入。这些文件体积大或为生成产物，**上传 GitHub 时不传**，运行前需手动放入。

## 数据说明

`split_files/` 下每个 CSV 对应一个分类层级，含 `glytoucan_id`、`label`、`WURCS` 等列，按 train/valid/test 划分。8 个层级：domain、kingdom、phylum、class、order、family、genus、species。

## 运行顺序

```bash
# 前置：确保 embedding_vocab.json 与 checkpoints/best_model_mlm.pt 已放入本目录

# 1. 提取嵌入 → 生成 split_files_processed_wurcs/*.csv
py -3.13 get_embeddings_W0.py \
    --input_dir ./split_files \
    --output_dir ./split_files_processed_wurcs \
    --vocab_path embedding_vocab.json \
    --checkpoint_path checkpoints/best_model_mlm.pt

# 2. 将 split_files_processed_wurcs/ 手动复制到 finaltrain 下每个种子文件夹内
#    每个种子文件夹内已有自己的 glycanW0_classifier.py（代码相同，种子不同）

# 3. 批量运行各种子的分类训练（递归查找并执行所有 glycanW0_classifier.py）
cd finaltrain
py -3.13 "新建 文本文档.py"

# 4. 回到上层，汇总多 seed 指标 → 生成 GlycanW0_metrics.csv
cd ..
py -3.13 总结.py
```
