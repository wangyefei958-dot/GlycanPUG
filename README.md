# glycanPUG 工作区总览

> 🌐 English version: [README.en.md](./README.en.md)

本仓库根目录下各文件夹的用途如下：

| 文件夹 | 说明 | README |
|--------|------|--------|
| `MASK/` | **主项目**：PUG（代码中称 W0）糖结构图神经网络预训练与下游任务。基于第 **95 轮**最优 checkpoint | [README](./MASK/README.md) |
| `MASK_i/` | **消融实验**版本，结构与 MASK 一致，基于第 **124 轮**最优 checkpoint，只跑了 taxall 任务 | [README](./MASK_i/README.md) |
| `GT/` | **对比实验**：用 GlycanGT 提取嵌入，与 PUG 公平比较 | [README](./GT/README.md) |
| `GlycanPUG/` | 整理后**上传 GitHub 的正式版本** | [README](./GlycanPUG/README.md) |
| `downstream/` | **下游任务原始数据与构建脚本**（各任务的 CSV 数据来源，详见下表） | — |

## downstream/ 内部结构

| 子文件夹 | 说明 |
|----------|------|
| `tax(ML)/` | 物种分类数据（**干净版，不含不确定结构**，来自 GlycanML），`分拆.py` 按层级拆分后供 `MASK/95/95轮tax/` 与 GT 对比实验使用 |
| `taxall/` | 物种分类「全物种」数据（含**不确定结构**，GlycanGT 无法处理），按 superkingdom → species 分 8 层 |
| `imme/` | 免疫原性数据构建（数据来自 GlycanML，经清洗/合并产出 `imm_stata.csv`） |
| `linkpred/` | GlyConnect 连接类型数据构建（爬取/处理，产出 `glyconnect_structures.csv`） |
| `dic.csv` | 从 GlyTouCan 下载的**原始全量**糖字典；`MASK/` 与 `MASK_i/` 内的 `dic.csv` 是运行过 `deldownstata.py` 删除泄漏行后的版本 |

## 命名说明

本项目模型最初称为 **W0**，后改名 **PUG**，代码中仍沿用 W0 称呼（如 `glycanW0_classifier.py`、`get_embeddings_W0.py`）。图组成与特征名的代码—论文命名对照见 [MASK/README.md](./MASK/README.md)。
