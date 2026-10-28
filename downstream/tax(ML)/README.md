# tax(ML) 数据处理流水线复现指南

本目录负责将原始糖基化分类表转换为每个分类级独立的训练数据，供下游 GNN 训练使用。

## 目录结构总览

```
tax(ML)/
├── allstata/                     ← Stage 0：原始数据补 WURCS
│   ├── glycan_classification.csv     原始输入（无 wurcs 列）
│   ├── 0add_wurcs_to_csv.py          API 批量转 WURCS
│   ├── test_single_glycan.py         工具函数库
│   ├── iupac_tree.py                 树结构认证（解析成残基树 + 同构比较）
│   ├── 1batch_process_errors.py      修复 API 失败行
│   ├── 2merge_fixed.py               回填修复结果
│   └── ...中间产物...
├── allstata/glycan_classification_final.csv  Stage 0 最终产物（3split.py 直接读这份）
├── 3split.py                     ← Stage 1：按 8 个分类级拆分
├── split_log.txt                 ← Stage 1 产出的低频标签清单
└── split_files/                  ← Stage 1 最终产出
    ├── species.csv
    ├── genus.csv
    └── ...（共 8 个分类级 CSV）
```

## 依赖

| 阶段 | 第三方包 | 说明 |
|---|---|---|
| Stage 0（allstata） | `requests` | 调 glycosmos API |
| Stage 1（分拆） | `pandas` | 唯一的分析依赖 |

全部使用 Python 3.13 运行。

---

## 复现步骤

### Stage 0：补全 WURCS（`allstata/` 内操作）

**输入**：`allstata/glycan_classification.csv`  
**原始表头**：`target, species, genus, family, order, class, phylum, kingdom, domain`

#### 0.1 调 API 转 WURCS

```bash
cd allstata
python 0add_wurcs_to_csv.py
```

- API：`https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs`
- 6 线程并发，每 0.1s 一次请求，失败重试 2 次
- **产出**：
  - `glycan_classification_with_wurcs.csv`（成功行，新增 `wurcs` 列）
  - `glycan_classification_errors.csv`（失败行，`wurcs=ERROR` + `error_reason` 列）
  - `wurcs_progress.json`（断点续传进度）

#### 0.2 修复失败行

```bash
python 1batch_process_errors.py
```

- 从 `test_single_glycan.py` 导入工具函数（命名规范化、修饰基替换、IUPAC↔WURCS 互转）
- **树认证（最后一步）**：由 `iupac_tree.py` 完成，取代原先的字符频次比对。把 `modified_iupac`（期望结构）和 `verified_iupac`（API 回转结构）分别解析成残基树，用规范化签名做同构比较，逐项核对残基类型、连接端点、分支位置、立体化学（α/β）和修饰；字符计数相同但结构不同的会被判为 MISMATCH
- **送 API 前拦截**：括号不配对等非法输入标记 `INVALID_INPUT`；糖醛酸 A 与 OAc 共存（共用 `?F` 占位符）标记 `COLLISION`，两者都不调用 API
- 断点续传：已处理行会跳过
- **产出**：`glycan_classification_errors_processed.csv`（含 `status` / `modified_iupac` / `verified_iupac` 列）

#### 0.3 回填修复结果

```bash
python 2merge_fixed.py
```

- 用 `(target, species)` 作 key，把修复的 WURCS 回填到原 `ERROR` 行
- **产出**：`allstata/glycan_classification_final.csv`
- **最终表头**：`glycan, species, genus, family, order, class, phylum, kingdom, domain, wurcs`  
  （注意第一列从 `target` 改名为 `glycan`）

#### 0.4 无需复制文件

`3split.py` 已改为绝对路径直接读取 `allstata/glycan_classification_final.csv`，与运行时的当前目录无关，Stage 0 结束后可直接执行 Stage 1。旧的 `glycan_classification_merged.csv` 是历史产物，勿再使用。

### Stage 1：按分类级拆分（根目录操作）

```bash
cd ..
python 3split.py
```

- 读 `allstata/glycan_classification_final.csv`（绝对路径，与运行目录无关）
- 8 个分类级独立处理：`species, genus, family, order, class, phylum, kingdom, domain`
- 每个分类级：去空 → 去低频标签（出现次数 < 30）→ 按 WURCS 聚合
- **产出**：
  - `split_files/{species,genus,...domain}.csv`（表头：`wurcs, {分类级}, glycan`）
  - `split_log.txt`（被删的低频标签清单）

> `split_files/` 当前已经存在且包含完整产物。如果不重跑 Stage 0，可以直接查看该目录；如果想从头跑一遍，依次执行 Stage 0 → Stage 1 即可。

---

## 常见问题

**Q: API 接口超时或限流？**  
接口来自 [glycosmos.org](https://api.glycosmos.org/glycanformatconverter/)，当前脚本已设 30s 超时、2 次重试、0.1s 间隔。如果仍失败，降低 `MAX_WORKERS` 或增加 `REQUEST_INTERVAL`。

**Q: 为什么 allstata/ 里也有一份 glycan_classification_merged.csv？**  
那是历史版本 2merge 脚本的产物，现已过期。当前 2merge 输出 `glycan_classification_final.csv`，`3split.py` 直接读它，无需再复制到根目录。
