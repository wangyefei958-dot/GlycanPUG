# Glycan Taxonomy Pipeline（糖结构-物种分类流水线）

按 NCBI 分类阶层（superkingdom → species 共 8 级）构建多标签糖结构数据集的流水线。从 [Glycosmos](https://glycosmos.org/organisms) 抓取每个阶层下的物种及其糖链列表，用 GlyTouCan ID 关联到 WURCS 结构，清洗异常结构后按 WURCS 去重合并，跨多个物种的标签以下划线 `_` 连接。

## 目录结构

```
taxall/
├── preprocessall.py              # 主预处理脚本（遍历每个阶层文件夹）
├── 1superkingdom/
│   ├── dic.csv                   # GlyTouCan ID → WURCS 字典（~57 MB）
│   ├── superkingdom.csv          # 最终合并输出
│   └── data/
│       ├── glycosmos_crawler.py  # 该阶层专用爬虫
│       └── <taxon>.csv ...       # 每个分类单元一个 CSV（GlyTouCan ID 列）
├── 2phylum/        …  结构同上，输出 phylum.csv
├── 3class/         …  class.csv
├── 4order/         …  order.csv
├── 5family/        …  family.csv
├── 6genus/         …  genus.csv
├── 7kingdom/       …  kingdom.csv
├── 8species/       …  species.csv
├── split_files/    # 手动创建并手动移动文件，存放 8 个最终 CSV 的副本
│   ├── *.csv
│   ├── 数量统计.py    # 行数 / 标签数统计
│   └── result.txt
└── __pycache__/     # 残留字节码：remove_tilde.py / 合并_带标签.py / creature_to_graph.py（源文件已删除，前两者已重构进 preprocessall.py）
```

每个 `<N><rank>/` 文件夹是自包含的：同一份 `dic.csv`，自己的 `data/` 子目录（含该阶层的爬虫与每个分类单元的原始 CSV），以及该阶层的最终输出 CSV。

## 数据流

### 1. 抓取 — `data/glycosmos_crawler.py`

每个阶层文件夹内有一份针对该阶层定制的爬虫。它做的事情：

- 分页请求 `https://glycosmos.org/organisms/index?q[rank_in]=<RANK>&q[s]=glycan_count+desc`。
- 收集 `glycans_count > 20` 的物种（superkingdom 用 `> 30`）。
- 列表按 glycan 数降序，遇到第一个 `glycans_count <= 20` 的物种即停止抓取。
- 对每个物种，从 `https://glycosmos.org/organisms/<id>/glycans.csv` 下载其糖链 CSV，只保留 `GlyTouCan ID` 列，保存为 `data/<organism_name>.csv`。

### 2. 清洗 + 合并 — `preprocessall.py`

从项目根目录运行。遍历每一个 `<N><rank>/` 子文件夹，对每个文件夹：

1. 读 `dic.csv`，构建 `GlyTouCan ID → WURCS` 映射。
2. 对 `data/` 下每个 CSV：
   - 读取后用 GlyTouCan ID 关联，**添加 `WURCS` 列**（原地写回）。
   - **清洗规则（硬约束）**：
     - 删除任意列包含 `~` 的行（`~` 是 WURCS 的占位符，表示结构不完整）。
     - 删除 `GlyTouCan ID == G18907LN` 的行（硬编码排除的异常条目）。
     - 清洗后剩余行数 **少于 30** 的文件整文件丢弃（样本量不足）。
   - 加临时列 `_temp_label`，值为该文件名 stem（即分类单元名）。
3. 拼接所有保留的文件。
4. **按 `WURCS` 分组合并**（去重）；若同一 WURCS 来自多个文件，把对应的分类单元名排序后用 `_` 连接成多标签字符串。
5. 丢弃 `_temp_label` 列，规范 `WURCS` 列名，去掉 unnamed 列。
6. 标签列名取自阶层名去掉前缀数字（如 `1superkingdom` → `superkingdom`）。
7. 在文件夹根目录写出最终合并文件 `<rank>.csv`（如 `1superkingdom/superkingdom.csv`）。

### 3. 拆分目录 — `split_files/`

手动创建、手动移动文件的目录。8 个 `<rank>.csv` 被手工拷贝到这里方便下游统一处理。没有任何脚本生成这个文件夹。

### 4. 统计 — `split_files/数量统计.py`

扫描自身所在目录下所有 CSV，对每个文件统计：

- 有效数据行数（跳过表头，要求至少 3 列）。
- 独立标签数（第三列按下划线 `_` 分割后去重；总数为跨文件去重）。

结果同时打印到控制台并写入 `result.txt`。

## 输出

8 个最终 CSV（每个阶层一个），列为：`GlyTouCan ID, WURCS, <rank>`，其中 `<rank>` 列存放一个或多个分类单元名，多个时用 `_` 连接。

### 数量统计（来自 `split_files/result.txt`）

| 文件               | 行数    | 独立标签数 |
|--------------------|---------|------------|
| superkingdom.csv   | 15,431  | 4          |
| kingdom.csv        | 11,720  | 5          |
| phylum.csv         | 15,271  | 19         |
| class.csv          | 15,089  | 31         |
| order.csv          | 14,571  | 65         |
| family.csv         | 14,054  | 76         |
| genus.csv          | 13,179  | 85         |
| species.csv        | 11,877  | 79         |
| **合计**           | 111,192 | 364（去重）|

## 依赖

- Python 3.13（已验证）
- `pandas`、`numpy`
- `requests`、`beautifulsoup4`（仅爬虫需要）

## 注意

- **爬虫具有时效性**：[Glycosmos](https://glycosmos.org/organisms) 网站可能随时更新（列表页 URL 结构、HTML 元素 class、糖链 CSV 导出格式等都可能变），若重新抓取时爬虫报错或抓到空数据，需要根据当前网站结构调整 `data/glycosmos_crawler.py` 中的 URL、CSS 选择器或列名。
- 8 份 `dic.csv` 是同一份 GlyTouCan WURCS 字典的字节级相同副本（各 ~57 MB）。
- `__pycache__` 下的 `remove_tilde.pyc`、`合并_带标签.pyc`、`creature_to_graph.pyc` 是旧脚本残留字节码，对应 `.py` 源文件已不存在；前两者的功能已合并进 `preprocessall.py`。
- 所有清洗操作都**原地修改** `data/` 下的 CSV 文件，删除不可逆；首次运行前请备份原始数据。
