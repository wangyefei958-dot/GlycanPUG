# tax(ML) 数据处理流水线复现指南

本目录负责将原始糖基化分类表转换为每个分类级独立的训练数据，供下游 GNN 训练使用。

## 目录结构总览

```
tax(ML)/
├── allstata/                     ← Stage 0：原始数据补 WURCS
│   ├── glycan_classification.csv     原始输入（无 wurcs 列）
│   ├── 0add_wurcs_to_csv.py          API 批量转 WURCS
│   ├── test_single_glycan.py         工具函数库
│   ├── 1batch_process_errors.py      修复 API 失败行
│   ├── 2merge_fixed.py               回填修复结果
│   └── ...中间产物...
├── glycan_classification_merged.csv  Stage 0 最终产物（根目录的分拆.py 读这份）
├── 分拆.py                       ← Stage 1：按 8 个分类级拆分
└── split_files/                  ← Stage 1 最终产出
    ├── species.csv
    ├── genus.csv
    └── ...（共 8 个分类级 CSV + split_log.txt）
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
python add_wurcs_to_csv.py
```

- API：`https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs`
- 6 线程并发，每 0.1s 一次请求，失败重试 2 次
- **产出**：
  - `glycan_classification_with_wurcs.csv`（成功行，新增 `wurcs` 列）
  - `glycan_classification_errors.csv`（失败行，`wurcs=ERROR` + `error_reason` 列）
  - `wurcs_progress.json`（断点续传进度）

#### 0.2 修复失败行

```bash
python batch_process_errors.py
```

- 从 `test_single_glycan.py` 导入工具函数（命名规范化、修饰基替换、IUPAC↔WURCS 互转）
- 断点续传：已处理行会跳过
- **产出**：`glycan_classification_errors_processed.csv`

#### 0.3 回填修复结果

```bash
python merge_fixed.py
```

- 用 `(target, species)` 作 key，把修复的 WURCS 回填到原 `ERROR` 行
- **产出**：`allstata/glycan_classification_merged.csv`
- **最终表头**：`glycan, species, genus, family, order, class, phylum, kingdom, domain, wurcs`  
  （注意第一列从 `target` 改名为 `glycan`）

#### 0.4 复制 merged.csv 到根目录（Stage 1 依赖）

```powershell
Copy-Item "allstata\glycan_classification_merged.csv" "."
```

根目录的 `分拆.py` 用相对路径读同目录的 `glycan_classification_merged.csv`，所以 Stage 0 结束后必须把这份文件拷到根目录。

### Stage 1：按分类级拆分（根目录操作）

```bash
cd ..
python 分拆.py
```

- 读同目录下的 `glycan_classification_merged.csv`
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
`allstata/merge_fixed.py` 的输出就在 `allstata/` 里。根目录的 `分拆.py` 读的是同目录那份，所以 Stage 0 做完后需要手动复制一次（Stage 0.4）。allstata/ 里的那份可以留着也可以删，不影响。
