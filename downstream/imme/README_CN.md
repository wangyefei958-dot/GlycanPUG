# 糖链免疫原性数据集 — WURCS 转换流水线

本项目将糖链免疫原性数据集中的 **IUPAC Condensed** 格式糖链字符串转换为 **WURCS** 格式，为下游糖链建模提供计算机可读的标准化表示。

## 数据来源

- **原始文件**: `glycan_immunogenicity.csv`（1320 条糖链记录）
- **输出文件**: `imm_stata.csv`（过滤后 1106 条可用记录）

### 原始数据列说明

| 列名 | 含义 |
|------|------|
| `glycan_id` | 糖链编号 |
| `glycan` | IUPAC Condensed 格式的糖链字符串（转换输入） |
| `species` | 来源物种 |
| `immunogenicity` | 免疫原性标签（0.0 / 1.0） |
| `inferred_origin` | 推断来源 |
| `link` | 注释列 |
| `train` / `valid` / `test` | 数据集划分标记 |

---

## 流水线执行顺序

脚本文件名前的数字即执行顺序，**必须按序运行**：

```
0add_wurcs_to_csv.py      ← 第 1 步
        │
        ▼
1batch_process_errors.py  ← 第 2 步（处理第 1 步的失败记录）
        │
        ▼
2merge_fixed.py           ← 第 3 步（合并修复结果）
        │
        ▼
3create_imm_stata.py      ← 第 4 步（生成最终产物）
```

### 各步骤详解

#### Step 1 — 首轮转换：`0add_wurcs_to_csv.py`

- **输入**: `glycan_immunogenicity.csv`
- **API**: `https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs`
- **策略**: 6 线程并发，直接调用 Glycosmos API 将 IUPAC Condensed 转为 WURCS
- **输出**:
  - `glycan_immunogenicity_with_wurcs.csv` — 全部记录（成功的带 WURCS，失败的标记 `wurcs=ERROR`）
  - `glycan_immunogenicity_errors.csv` — 仅失败记录（~691 条）
  - `wurcs_immunogenicity_progress.json` — 断点续传进度文件

#### Step 2 — 错误修复：`1batch_process_errors.py`

- **输入**: `glycan_immunogenicity_errors.csv`
- **核心逻辑**: 从 `test_single_glycan.py` 导入工具函数，使用**占位符替换策略**绕过 API 对罕见修饰的识别限制：

  | 原始修饰 | 占位符 | 说明 |
  |----------|--------|------|
  | OMe      | `?Me`  | 甲氧基 |
  | 糖名+A   | `?F`   | 糖醛酸（如 GalA） |
  | OP       | `?P`   | 磷酸化 |
  | OS       | `?S`   | 硫酸化 |
  | OAc      | `?F` → WURCS 中再替换为 `_?*OCC/3=O` | 乙酰基 |
  | NeuNAc   | Neu5Ac | 神经氨酸重命名 |

- **验证流程**: 占位符替换 → API 转 WURCS → WURCS 去除占位符 → 反转回 IUPAC → 字符频次比对验证
- **输出**: `glycan_immunogenicity_errors_processed.csv`（包含 SUCCESS / MISMATCH / ERROR 三种状态）

#### Step 3 — 合并修复：`2merge_fixed.py`

- **输入**:
  - `glycan_immunogenicity_with_wurcs.csv`（首轮结果）
  - `glycan_immunogenicity_errors_processed.csv`（修复结果）
- **策略**: 以 `(glycan, species)` 为 key，将修复成功的 WURCS 回填到首轮结果中
- **输出**: `glycan_immunogenicity_final.csv`（仍有部分 ERROR 残留）

#### Step 4 — 最终过滤：`3create_imm_stata.py`

- **输入**: `glycan_immunogenicity_final.csv`
- **策略**: 移除所有 `wurcs == "ERROR"` 的行
- **输出**: `imm_stata.csv`（最终产物，供下游建模使用）

---

## 文件总览

| 文件 | 类型 | 说明 |
|------|------|------|
| `glycan_immunogenicity.csv` | 输入 | 原始糖链免疫原性数据 |
| `glycan_immunogenicity_with_wurcs.csv` | 中间产物 | 首轮 API 转换结果 |
| `glycan_immunogenicity_errors.csv` | 中间产物 | 首轮转换失败记录 |
| `glycan_immunogenicity_errors_processed.csv` | 中间产物 | 错误修复结果 |
| `glycan_immunogenicity_final.csv` | 中间产物 | 合并修复后的数据 |
| `imm_stata.csv` | **最终产物** | 过滤干净、含 WURCS 的可用数据集 |
| `test_single_glycan.py` | 工具 | 占位符替换 + 双向转换 + 验证的核心函数库（Step 2 使用） |

---

## 依赖

```
requests    # HTTP 请求（调用 Glycosmos API）
```

Python 标准库：`csv`, `os`, `re`, `json`, `threading`, `concurrent.futures`, `collections`

## 注意事项

- **API 调用**: Glycosmos 公共 API 无认证要求，但需控制并发（本项目 MAX_WORKERS=6 / 28，REQUEST_INTERVAL=0.1~0.5s）以避免被限流
- **断点续传**: Step 1 使用 `wurcs_immunogenicity_progress.json` 保存进度，中途中断可恢复
- **占位符策略**: Step 2 的替换逻辑基于经验整理，并非所有修饰都能被 Glycosmos API 正确识别，失败的会保留 ERROR 标记
