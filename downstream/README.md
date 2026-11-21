# downstream — 下游任务数据与构建脚本

> 🌐 English version: [README.en.md](./README.en.md)

本文件夹存放各下游任务的原始数据、构建脚本与数据字典。各子任务的详细说明见各自 README：

| 子文件夹 | 说明 | README |
|----------|------|--------|
| `tax(ML)/` | 物种分类数据（**干净版，不含不确定结构**，来自 GlycanML），`3split.py` 按层级拆分出 `split_files/` | [README](./tax(ML)/README.md) |
| `taxall/` | 物种分类「全物种」数据（**来自 Glycosmos**，含**不确定结构**，GlycanGT 无法处理），按 superkingdom → species 分 8 层 | [README](./taxall/README.md) |
| `imme/` | 免疫原性数据构建（数据来自 GlycanML），产出 `imm_stata.csv` | [README](./imme/README_CN.md) |
| `linkpred/` | GlyConnect 连接类型数据构建（爬取/处理），产出 `glyconnect_structures.csv` | [README](./linkpred/README_CN.md) |

顶层文件：

- `dic.csv` — 从 GlyTouCan 下载的**原始全量**糖字典（未删除泄漏行）；`MASK/` 与 `MASK_i/` 内的 `dic.csv` 是运行过 `deldownstata.py` 后的版本
- `downstream_task_stats.csv / .png`、`plot_downstream_task_stats.py`、`report_second_round.py`、`second_round_report.md` — 下游任务规模统计与报告脚本
- `split_log.txt` — 数据拆分日志
