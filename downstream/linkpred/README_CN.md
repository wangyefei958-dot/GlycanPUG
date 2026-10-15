# GlyConnect 糖结构数据处理流水线

从 [GlyConnect](https://glyconnect.expasy.org/) 数据库爬取糖结构数据，补全 WURCS 格式标识，生成用于下游任务的标准化糖数据集。

## 处理流程

```
GlyConnect 网站
      │
      ▼  0glyconnect_crawler.py
glyconnect_structures.csv  (3列: GlyTouCan ID, glycantype, IUPAC Condensed)
      │
      ▼  1process_glyconnect.py  +  dic.csv
glyconnect_structures.csv  (4列: + WURCS)
      │
      ▼  2remove_tilde.py
creature_0wave.csv  (过滤掉聚合物结构后的版本)
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `0glyconnect_crawler.py` | 爬取 GlyConnect 网站所有糖结构，提取 GlyTouCan ID、糖类型和 IUPAC 结构式 |
| `1process_glyconnect.py` | 用 `dic.csv` 中的映射补上 WURCS 列，清洗空行和无效行 |
| `2remove_tilde.py` | 过滤掉 WURCS 中含 `~`（聚合物重复单元）的结构及个别异常条目 |
| `dic.csv` | GlyTouCan ID → WURCS 的映射字典（外部来源） |
| `glyconnect_structures.csv` | 爬取 + 合并后的最终数据集，包含 4 列 |

## 依赖

- Python 3.10+
- `requests`
- `beautifulsoup4`
- `pandas`

```bash
pip install requests beautifulsoup4 pandas
```

## 使用方法

按编号顺序依次运行：

```bash
# 第 1 步：爬取 GlyConnect 网站（约 57 页，耗时数分钟）
python 0glyconnect_crawler.py

# 第 2 步：合并 WURCS 标识（依赖 dic.csv 已存在）
python 1process_glyconnect.py

# 第 3 步：过滤聚合物结构
python 2remove_tilde.py
```

每一步都会**覆盖**上一步的 `glyconnect_structures.csv`，最终步骤输出 `creature_0wave.csv`。如需保留中间结果，请手动备份。

## 输出数据格式

`glyconnect_structures.csv` 包含以下 4 列：

| 列名 | 示例 | 说明 |
| --- | --- | --- |
| `GlyTouCan ID` | `G51562OR` | [GlyTouCan](https://glytoucan.org/) 糖结构唯一标识 |
| `glycantype` | `N-Linked`, `O-Linked` 等 | 糖的连接类型 |
| `IUPAC Condensed` | `Gal(b1-4)GlcNAc(b1-3)...` | IUPAC 紧缩格式的糖结构式 |
| `WURCS` | `WURCS=2.0/5,18,17/...` | [WURCS 2.0](https://www.wurcs.org/) 格式的糖结构描述 |
