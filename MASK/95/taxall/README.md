# taxall — 物种分类「全物种」配置

基于主文件夹预训练得到的**第 95 轮最优 checkpoint**，对糖类按物种分类学全部层级（含 superkingdom，共 9 层）进行嵌入提取与多种子分类训练。本目录是 `95轮tax` 的「全物种」扩展版本，`finaltrain/` 内保存最终训练结果与图结构不确定性分析。

> 🌐 English version: [README.en.md](./README.en.md)

## 目录结构

```
taxall/
├── masktrain_windows_313.py        # ⬅ 从主文件夹移入（外部）
├── wurcs_parser_reverse_edge.py     # ⬅ 从主文件夹移入（外部）
├── get_embeddings_W0.py            # 提取 1027 维嵌入
├── 总结.py                          # 汇总多 seed 的 F1/AUC 指标
├── embedding_vocab.json            # ⬅ 从主文件夹移入（外部）
├── GlycanW0_metrics.csv            # 由 总结.py 生成
├── checkpoints/
│   └── best_model_mlm.pt          # ⬅ 从主文件夹移入（95 轮预训练权重，外部）
├── split_files/                     # ⬅ 外部移入：按层级拆分的 WURCS CSV（含 superkingdom）
│   ├── superkingdom.csv / kingdom.csv / phylum.csv / class.csv
│   ├── order.csv / family.csv / genus.csv / species.csv
│   ├── 数量统计.py                  # 统计各 CSV 行数与标签数
│   └── result.txt                   # 统计结果输出
├── split_files_processed_wurcs/     # 由 get_embeddings_W0.py 生成（嵌入 CSV）
├── finaltrain/                      # 多种子最终训练 + 不确定性分析
│   ├── 新建 文本文档.py             # 批量运行各种子目录下的 glycanW0_classifier.py
│   ├── analyze_graph_uncertainty.py  # 图结构不确定性 vs 预测对错分析
│   ├── wurcs_parser_reverse_edge.py   # ⬅ 从主文件夹移入（外部，供不确定性分析用）
│   ├── embedding_vocab.json          # ⬅ 从主文件夹移入（外部）
│   ├── 42/                          # 种子 42
│   │   ├── glycanW0_classifier.py   # 与其他种子相同，仅种子参数不同
│   │   └── split_files_processed_wurcs/  # ⬅ 手动移入（从顶层复制）
│   │       └── training_results_<level>/  # 各层级训练产物
│   ├── 58/                          # 种子 58（结构同上）
│   ├── 518/                         # 种子 518
│   ├── 5188/                        # 种子 5188
│   ├── 51888/                       # 种子 51888
│   ├── split_files/                 # 供不确定性分析使用的数据副本
│   └── uncertainty_graph/           # 不确定性分析产物（图表 + summary_table.json）
```

> ⬅ 标记的文件为**从外部移入**：`masktrain_windows_313.py`、`wurcs_parser_reverse_edge.py`、`embedding_vocab.json` 与 `checkpoints/best_model_mlm.pt` 来自主文件夹 `MASK/`，`split_files/*.csv` 为按物种层级拆分的外部数据。`finaltrain/<seed>/split_files_processed_wurcs/` 由顶层手动复制移入。这些文件体积大或为生成产物，**上传 GitHub 时不传**，运行前需手动放入。

## 数据说明

`split_files/` 下每个 CSV 对应一个分类层级，含 `glytoucan_id`、`label`、`WURCS` 等列。共 9 个层级：superkingdom、kingdom、phylum、class、order、family、genus、species（比 `95轮tax` 多 superkingdom）。

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

# 5. （可选）图结构不确定性分析
cd finaltrain
py -3.13 analyze_graph_uncertainty.py
```

## finaltrain 说明

`finaltrain/` 保存多种子最终训练结果与不确定性分析。`新建 文本文档.py` 递归查找并批量执行所有 `glycanW0_classifier.py`。`analyze_graph_uncertainty.py` 比较「分对 vs 分错」样本的图结构不确定特征（doubt 边占比、平均边概率、anomer_x、stereo_x 等），多进程解析 WURCS，5 个种子各自统计后画 2x2 柱状图，并做 Mann-Whitney U 检验。产物在 `uncertainty_graph/` 下。
