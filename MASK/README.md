# MASK：糖类结构图神经网络预训练与下游物种分类

> 🌐 English version: [README.en.md](./README.en.md)

本项目实现了一条完整的「WURCS 糖类结构 → 图神经网络掩码预训练 → 嵌入提取 → 物种分类层级」流水线。核心思路是把糖类的 WURCS 格式串解析为异构图（big/small 节点 + big/small 边），用 MLM（Masked Language Modeling）方式预训练一个异构 GAT，再用预训练模型提取 1027 维图级嵌入，最后在 8 个分类学层级（domain → species）上训练分类头。

## 目录结构

```
MASK/
├── wurcs_parser_reverse_edge.py   # WURCS 解析器（可独立运行，产出 json/txt/vocab 等）
├── masktrain_windows_313.py       # MLM 预训练（HeteroGAT）
├── deldownstata.py                # 数据泄漏清理工具
├── training_metrics_log.txt        # 预训练每轮指标日志（由预训练脚本生成）
├── dic.csv                         # 数据字典（**已由 deldownstata.py 删除下游任务样本**；原始未清理版见 `downstream/dic.csv`）
├── output_all.json / output_all.txt # 解析后的全量图数据（由解析器生成）
├── embedding_vocab.json           # 特征词表（由解析器生成）
├── vocab_freq_stats.txt           # 词表频次统计
├── parse_errors.csv               # 解析失败的 WURCS 记录
├── checkpoints/                    # 模型权重
│   └── best_model_mlm.pt
├── downstreamstata/                # 下游统计/泄漏检查用 CSV
├── error_analysis/                 # 训练日志与误分析
└── 94/                             # 基于「第 94 轮最优 checkpoint」的下游任务（见下文「关于 94 文件夹」）
    ├── tax(ML)/                    # 物种分类下游任务（8 层级，domain → species）
    ├── link/                       # 糖连接下游任务
    ├── imm(ML)/                    # 免疫原性下游任务
    └── taxall/                     # 物种分类「全物种」配置
```

## 各脚本说明

### 1. `wurcs_parser_reverse_edge.py`
WURCS 格式解析器。既可被其他脚本作为模块导入，也**可直接运行**：自动读取脚本目录里唯一的 `dic.csv`，把每条 `WURCS=...` 串解析为 GNN 输入的异构图结构（节点/边的代码命名见下文「代码—论文命名对照」），并做两遍扫描（先收集词表、再过滤低频/不确定项）。直接运行后产出：
- `output_all.json`（解析后的全量图数据，供预训练读取）
- `output_all.txt`（人类可读的图结构文本版）
- `embedding_vocab.json`（特征词表，预训练/嵌入提取都要用）
- `vocab_freq_stats.txt`（各特征类型的频次统计）
- `parse_errors.csv`（解析失败的 WURCS 记录）

### 2. `masktrain_windows_313.py`
MLM 预训练主脚本。读取上一步生成的 `output_all.json` 与 `embedding_vocab.json`，构建 `GlycanDataset`，对节点/边特征做随机掩码，训练一个异构 GAT（`HeteroConv + GATConv`）预测被掩码的 `anomeric_config`、`substituent`、`big_edge`、`small_edge_stereo`、`ring_bond` 等特征。使用 CB Focal Loss、AdamW、ReduceLROnPlateau、混合精度训练。产物：`checkpoints/best_model_mlm.pt`（预训练模型权重）与 `error_analysis/training_log.txt`。

### 3. `deldownstata.py`
数据泄漏清理工具。扫描 `downstreamstata/` 下所有 CSV 收集出现的 WURCS，再从 `dic.csv` 中删除这些行，避免下游评估样本泄漏进预训练/训练集。仓库内的 `MASK/dic.csv` 已是清理后的版本，原始未清理字典在 `downstream/dic.csv`。

## 代码—论文命名对照

图组成（节点/边）在代码与论文中的叫法：

| 代码中的名称 | 论文中的名称 | 含义 |
|--------------|--------------|------|
| `big_nodes` | residue | 糖残基节点 |
| `small_nodes` | substituents | 取代基节点 |
| `big_edges` | inter-residue glycosidic bonds | 残基间糖苷键 |
| `small_edges` | substituent attachment bonds | 取代基连接键 |

## 产物清单与大小

> 以下产物均可由对应脚本重新生成，**上传 GitHub 时一律不传，仅保留源代码**。列表供运行时参考。

| 脚本 | 产物 | 典型大小 |
|------|------|---------|
| `wurcs_parser_reverse_edge.py` | `output_all.json`（解析后的全量图数据） | ~2.3 GB |
| `wurcs_parser_reverse_edge.py` | `output_all.txt`（人类可读图结构文本） | ~945 MB |
| `wurcs_parser_reverse_edge.py` | `embedding_vocab.json`（特征词表） | < 1 MB |
| `wurcs_parser_reverse_edge.py` | `vocab_freq_stats.txt`（频次统计） | 小 |
| `wurcs_parser_reverse_edge.py` | `parse_errors.csv`（解析失败记录） | 小 |
| `masktrain_windows_313.py` | `checkpoints/best_model_mlm.pt`（预训练权重） | ~34 MB |
| `masktrain_windows_313.py` | `error_analysis/training_log.txt`（训练日志） | 小 |
| `masktrain_windows_313.py` | `training_metrics_log.txt`（每轮指标日志） | 小 |
| `94/tax(ML)/总结.py`、`94/taxall/总结.py` | `GlycanW0_metrics.csv`（指标汇总） | 小 |
| `deldownstata.py` | 原地改写 `dic.csv`（删除泄漏行；原始未清理版为 `downstream/dic.csv`） | `dic.csv` ~50 MB |

## 运行顺序（主文件夹）

> 以下命令假设工作目录为 `MASK/`，Python 版本 3.13（脚本名中的 `_313` 即指 3.13）。Windows 下用 `py -3.13`，其他平台用 `python`。

```bash
# Step 0. 仓库内 dic.csv 已是删除下游任务样本后的版本，可直接从 Step 2 开始；
#         若要从原始字典重跑：cp ../downstream/dic.csv ./dic.csv

# Step 1. 清理数据泄漏：从 dic.csv 删除出现在 downstreamstata/ 中的 WURCS（原地改写 dic.csv）
py -3.13 deldownstata.py

# Step 2. 解析 WURCS → 产出 output_all.json / output_all.txt / embedding_vocab.json / vocab_freq_stats.txt / parse_errors.csv
py -3.13 wurcs_parser_reverse_edge.py

# Step 3. MLM 预训练 → 产出 checkpoints/best_model_mlm.pt（读取 Step 2 的 output_all.json 与 embedding_vocab.json）
py -3.13 masktrain_windows_313.py
```

预训练完成后，下游任务（嵌入提取、分类训练、指标汇总等）在 `94/` 文件夹内各自展开，详见下文「关于 94 文件夹」。

## 关于 94 文件夹

预训练时第 **94 轮（epoch 94）的 checkpoint 是最优模型**，因此所有下游任务都基于这个权重展开，相关文件夹统一以 `94` 命名。`94/` 下按下游任务类型分子目录，**每个子目录各有自己的 README**：

| 子目录 | 任务 | README |
|--------|------|--------|
| `94/tax(ML)/` | 物种分类（taxonomy，8 层级） | [README](./94/tax(ML)/README.md) |
| `94/link/` | 糖连接类型（glycan-linking） | [README](./94/link/README.md) |
| `94/imm(ML)/` | 免疫原性（immunogenicity） | [README](./94/imm(ML)/README.md) |
| `94/taxall/` | 物种分类「全物种」配置（含 superkingdom，8 层级） | [README](./94/taxall/README.md) |

> **注意**：各子目录中的 `embedding_vocab.json` 和 `checkpoints/best_model_mlm.pt` 是**从主文件夹移入**的外部文件（94 轮预训练产物），`split_files/*.csv` 或 `train/*.csv` 为外部任务数据。这些文件体积大或为生成产物，上传 GitHub 时不传，运行前需手动放入。详见各子目录 README。当前各种子目录的随机种子为 **1–10**；多 seed 指标汇总用各任务目录下的 `总结.py`（`94/tax(ML)/` 与 `94/taxall/` 各一份，会自动识别 `finaltrain/` 下以数字命名的种子目录），汇总 macro F1 与 macro AUC，输出 `GlycanW0_metrics.csv`。

## 依赖
- Python 3.13
- PyTorch（建议带 CUDA）
- torch-geometric（GATConv / HeteroConv / HeteroData）
- scikit-learn、scipy、iterstrat（多标签分层划分）
- pandas、numpy、matplotlib、seaborn、tqdm
