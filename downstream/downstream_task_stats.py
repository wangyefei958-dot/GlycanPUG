# -*- coding: utf-8 -*-
"""下游任务样本统计与总览图（重建版）

统计口径：
  - 分类任务（Glycosmos / GlycanML）：每个分类级的 split 文件行数 = 样本数，
    该级标签列按 "_" 拆分去重 = 标签数
  - 单任务（Glycosylation / Immunogenicity）：WURCS 有效（非空且非 ERROR）的行数 = 样本数
  - 总览的平均样本数 = 8 个分类级样本数的算术平均

输出：
  - 控制台表格
  - downstream_task_stats.csv（数值明细）
  - downstream_task_stats.png（调用 plot_downstream_task_stats.py 生图）

用法：
  py -3.13 downstream_task_stats.py
"""

import csv
from pathlib import Path

from plot_downstream_task_stats import render

BASE_DIR = Path(__file__).resolve().parent

GLYCOSMOS_DIR = BASE_DIR / "taxall" / "split_files"
GLYCANML_DIR = BASE_DIR / "tax(ML)" / "split_files"
IMMUNE_CSV = BASE_DIR / "imme" / "glycan_immunogenicity_final.csv"
GLYCOSYLATION_CSV = BASE_DIR / "linkpred" / "glyconnect_structures.csv"

# 图中从上到下的展示顺序
GLYCOSMOS_LEVELS = ["superkingdom", "kingdom", "phylum", "class",
                    "order", "family", "genus", "species"]
GLYCANML_LEVELS = ["domain", "kingdom", "phylum", "class",
                   "order", "family", "genus", "species"]


def split_stats(path, level):
    """返回 (样本数, 标签数)。标签列按表头匹配 level。"""
    with open(path, encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        try:
            idx = [h.strip().lower() for h in header].index(level)
        except ValueError:
            idx = len(header) - 1  # 兜底：标签在最后一列
        rows = 0
        labels = set()
        for row in reader:
            if len(row) <= idx:
                continue
            rows += 1
            for lbl in row[idx].split("_"):
                lbl = lbl.strip()
                if lbl:
                    labels.add(lbl)
    return rows, len(labels)


def taxonomy_stats(directory, levels):
    out = []
    for level in levels:
        path = directory / f"{level}.csv"
        if not path.exists():
            print(f"  [警告] 缺少 {path}")
            continue
        rows, labels = split_stats(path, level)
        out.append((level, rows, labels))
    return out


def count_valid_rows(path, wurcs_col):
    with open(path, encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        return sum(1 for r in reader
                   if (r.get(wurcs_col) or "").strip() not in ("", "ERROR"))


def print_table(name, stats):
    print(f"\n=== {name} ===")
    print(f"{'level':<16}{'samples':>10}{'labels':>10}")
    for level, rows, labels in stats:
        print(f"{level:<16}{rows:>10}{labels:>10}")


def save_csv(glycosmos, glycanml, glycosylation, immunogenicity, g_avg, m_avg):
    out = BASE_DIR / "downstream_task_stats.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "level", "samples", "labels"])
        for level, rows, labels in glycosmos:
            w.writerow(["Glycosmos", level, rows, labels])
        for level, rows, labels in glycanml:
            w.writerow(["GlycanML", level, rows, labels])
        w.writerow(["Glycosylation", "total", glycosylation, ""])
        w.writerow(["Immunogenicity", "total", immunogenicity, ""])
        w.writerow(["Glycosmos", "average", g_avg, ""])
        w.writerow(["GlycanML", "average", m_avg, ""])
    print(f"数值明细已保存: {out}")


def main():
    glycosmos = taxonomy_stats(GLYCOSMOS_DIR, GLYCOSMOS_LEVELS)
    glycanml = taxonomy_stats(GLYCANML_DIR, GLYCANML_LEVELS)
    glycosylation = count_valid_rows(GLYCOSYLATION_CSV, "WURCS")
    immunogenicity = count_valid_rows(IMMUNE_CSV, "wurcs")

    print_table("Glycosmos Taxonomy", glycosmos)
    print_table("GlycanML Taxonomy", glycanml)
    print("\n=== 单任务 ===")
    print(f"Glycosylation  samples={glycosylation:,}")
    print(f"Immunogenicity samples={immunogenicity:,}")

    g_avg = round(sum(r for _, r, _ in glycosmos) / len(glycosmos))
    m_avg = round(sum(r for _, r, _ in glycanml) / len(glycanml))
    print("\n=== 平均样本数（8 级） ===")
    print(f"Glycosmos avg={g_avg:,}   GlycanML avg={m_avg:,}")

    save_csv(glycosmos, glycanml, glycosylation, immunogenicity, g_avg, m_avg)
    print()
    render(glycosmos, glycanml,
           {"Glycosylation": glycosylation, "Immunogenicity": immunogenicity},
           {"Glycosmos": g_avg, "GlycanML": m_avg},
           BASE_DIR / "downstream_task_stats.png")


if __name__ == "__main__":
    main()
