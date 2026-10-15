# MASK补充（MASK_i）：消融实验

> 🌐 English version: [README.en.md](./README.en.md)

> **命名说明**：本项目最初将模型称为 **W0**，后改名为 **PUG**，但代码中仍沿用了 W0 的称呼（如 `glycanW0_classifier.py`、`get_embeddings_W0.py` 等）。

> **代码与论文命名对照**：代码沿用了编写时的命名，与论文略有不同——`big_nodes` 论文称 glycosyl residue nodes，`small_nodes` 称 substituent nodes，`big_edges` 称 inter-residue glycosidic edges，`small_edges` 称 substituent attachment edges，`stereo_type` 特征称 carbon descriptor。其余基本一致。

本文件夹是主项目 `MASK/` 的**消融实验（ablation study）版本**，目录结构与脚本与 `MASK/` 完全一致，**使用方法相同**，详见 [MASK 主 README](../MASK/README.md)。

## 与 MASK 的区别

消融实验在预训练时第 **124 轮（epoch）的 checkpoint 为最优模型**（MASK 主项目为 95 轮），因此下游任务文件夹统一命名为 `124`：

```
MASK补充/
├── wurcs_parser_reverse_edge.py   # WURCS 解析器（同 MASK）
├── masktrain_windows_313.py       # MLM 预训练（同 MASK）
├── deldownstata.py                # 数据泄漏清理工具（同 MASK）
├── dic.csv                         # 主数据字典
├── checkpoints/
│   └── best_model_mlm.pt          # 124 轮预训练权重
├── downstreamstata/                # 下游统计/泄漏检查用 CSV
└── 124/                            # 基于「124 轮最优 checkpoint」的下游任务
    └── taxall/                     # 物种分类「全物种」配置（含 superkingdom，9 层级）
        ├── get_embeddings_W0.py
        ├── glycanW0_classifier.py
        ├── 总结.py
        ├── embedding_vocab.json
        ├── GlycanW0_metrics.csv
        ├── split_files/             # 按层级拆分的 WURCS CSV
        └── finaltrain/              # 多种子最终训练（42 / 58 / 518 / 5188 / 51888）
```

> **注**：本文件夹内的 `dic.csv` 已运行过 `deldownstata.py`（已删除泄漏行）；从 GlyTouCan 下载的原始版本在 `downstream/` 文件夹内。

## 运行方式

预训练与下游任务的运行方式与 `MASK/` 完全相同，参见 [MASK 主 README 的「运行顺序」](../MASK/README.md#运行顺序主文件夹) 与 [「关于 95 文件夹」](../MASK/README.md#关于-95-文件夹) 章节，区别仅在于下游任务在 `124/` 下展开，且本消融实验目前只跑了 `taxall` 任务。
