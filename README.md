# glycanPUG：糖类结构图神经网络预训练与下游任务

> 🌐 English version: [README.en.md](./README.en.md)

本仓库把糖类的 WURCS 结构串解析为异构图（big/small 节点 + big/small 边），以 MLM 方式预训练异构 GAT（模型最初称 **W0**，后改名 **PUG**），再提取 1027 维图级嵌入，在物种分类、糖连接类型、免疫原性等下游任务上训练分类头。根目录下各文件夹的用途如下：

| 文件夹 | 说明 | README |
|--------|------|--------|
| `MASK/` | **主项目**：PUG 糖结构图神经网络预训练与下游任务，基于第 **94 轮**最优 checkpoint | [README](./MASK/README.md) |
| `GT/` | **对比实验**：用 GlycanGT 提取嵌入，与 PUG 公平比较 | [README](./GT/README.md) |
| `downstream/` | **下游任务原始数据与构建脚本**（各任务的 CSV 数据来源，详见下表） | — |

> `MASK_i/`（消融实验版本，结构与 MASK 一致）仍在运行中，完成后补充上传。

## downstream/ 内部结构

| 子文件夹 | 说明 | README |
|----------|------|--------|
| `tax(ML)/` | 物种分类数据（**干净版，不含不确定结构**，来自 GlycanML）：`allstata/` 为原始表补 WURCS，`3split.py` 按 8 个层级拆分到 `split_files/`，供 `MASK/94/tax(ML)/` 与 GT 对比实验使用 | [README](./downstream/tax(ML)/README.md) |
| `taxall/` | 物种分类「全物种」数据（含**不确定结构**，GlycanGT 无法处理）：`preprocessall.py` 按 superkingdom → species 生成 8 层 CSV | [README](./downstream/taxall/README.md) |
| `imme/` | 免疫原性数据构建（数据来自 GlycanML，经清洗/合并产出 `imm_stata.csv`） | [README](./downstream/imme/README.md) |
| `linkpred/` | GlyConnect 连接类型数据构建（爬取/处理，产出 `glyconnect_structures.csv`） | [README](./downstream/linkpred/README.md) |
| `dic.csv` | 从 GlyTouCan 下载的**原始全量**糖字典；`MASK/` 内的 `dic.csv` 是运行过 `deldownstata.py` 删除泄漏行后的版本 | — |
| `downstream_task_stats.py` / `downstream_task_stats.csv` / `downstream_task_stats.png` | 下游各任务的数据规模统计（样本数、标签数）及其绘图 | — |
| `report_second_round.py` / `second_round_report.md` | 第二轮 WURCS 转换失败原因分析（免疫原性、分类任务） | — |

> `tax(ML)/` 与 `imme/` 的**初始 CSV 均来自 GlycanML 基准**（Xu et al., [*GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning*](https://arxiv.org/abs/2405.16206), ICLR 2025）；`taxall/` 的数据则由 `preprocessall.py` 从 Glycosmos 抓取构建。

## 命名说明

本项目模型最初称为 **W0**，后改名 **PUG**，代码中仍沿用 W0 称呼（如 `glycanW0_classifier.py`、`get_embeddings_W0.py`）。图组成（节点/边）的代码—论文命名对照见 [MASK/README.md](./MASK/README.md)。
