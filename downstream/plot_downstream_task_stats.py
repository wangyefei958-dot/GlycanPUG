# -*- coding: utf-8 -*-
"""下游任务样本统计 —— 生图代码（独立版）

只负责绘图，不负责统计。数据来源两种方式：
  1) 直接运行：读取同目录的 downstream_task_stats.csv 并输出 PNG
  2) 被 import：调用 render(glycosmos, glycanml, totals, averages, out_path)

图布局（与原图一致）：
  A. Downstream Task Overview   （单任务总数 + 两个分类任务 8 级平均）
  B. Glycosmos: Label Count     （横向条形，x 轴反向）
  C. Glycosmos: Sample Count
  D. GlycanML: Label Count      （横向条形，x 轴反向）
  E. GlycanML: Sample Count

用法：
  py -3.13 plot_downstream_task_stats.py
  py -3.13 plot_downstream_task_stats.py --csv downstream_task_stats.csv --out my.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

COLOR_GREEN = "#5fa869"
COLOR_ORANGE = "#dd7a52"
COLOR_NAVY = "#1f3b73"
COLOR_BLUE = "#4c72b0"


def load_stats(csv_path):
    """从 downstream_task_stats.csv 还原绘图所需结构。"""
    glycosmos, glycanml = [], []
    totals, averages = {}, {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            dataset, level = row["dataset"], row["level"]
            samples = int(row["samples"])
            if level == "average":
                averages[dataset] = samples
                continue
            if level == "total":
                totals[dataset] = samples
                continue
            item = (level, samples, int(row["labels"]))
            if dataset == "Glycosmos":
                glycosmos.append(item)
            elif dataset == "GlycanML":
                glycanml.append(item)
    return glycosmos, glycanml, totals, averages


def _bar_panel(axis, stats, title, color, invert=False,
               show_ylabels=True, label_side="left"):
    levels = [s[0] for s in stats]
    values = [s[1] if "Sample" in title else s[2] for s in stats]
    ypos = list(range(len(levels)))[::-1]
    axis.barh(ypos, values, color=color, height=0.62)
    for yi, v in zip(ypos, values):
        # 反向坐标轴时，ha="right" 才能把标签放到柱子外侧
        axis.text(v + max(values) * 0.02, yi, f"{v:,}", va="center",
                  ha="right" if invert else "left", fontsize=9)
    axis.set_yticks(ypos)
    axis.set_yticklabels(levels, fontsize=10)
    if not show_ylabels:
        axis.tick_params(axis="y", left=False, labelleft=False)
    elif label_side == "right":
        axis.yaxis.set_ticks_position("right")
        axis.tick_params(axis="y", pad=6)
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Number of Samples (Structures)" if "Sample" in title
                    else "Number of Labels (Classes)", fontsize=9)
    axis.set_xlim(0, max(values) * 1.16)
    if invert:
        axis.invert_xaxis()


def _mirrored_pair(fig, sub, stats, left_title, left_color,
                   right_title, right_color):
    """左右两图共用纵坐标，分类标签放在中间的窄缝里。"""
    left = fig.add_subplot(sub[0, 0])
    gap = fig.add_subplot(sub[0, 1], sharey=left)
    right = fig.add_subplot(sub[0, 2], sharey=left)

    _bar_panel(left, stats, left_title, left_color, invert=True,
               show_ylabels=False)
    _bar_panel(right, stats, right_title, right_color, show_ylabels=False)

    gap.axis("off")
    ypos = list(range(len(stats)))[::-1]
    for yi, (level, _, _) in zip(ypos, stats):
        gap.text(0.5, yi, level, transform=gap.get_yaxis_transform(),
                 ha="center", va="center", fontsize=10)


def render(glycosmos, glycanml, totals, averages, out_path):
    """绘制 A~E 五个面板并保存为 out_path。"""
    glycosylation = totals["Glycosylation"]
    immunogenicity = totals["Immunogenicity"]
    g_avg = averages.get("Glycosmos",
                         round(sum(r for _, r, _ in glycosmos) / len(glycosmos)))
    m_avg = averages.get("GlycanML",
                         round(sum(r for _, r, _ in glycanml) / len(glycanml)))

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1],
                            hspace=0.42, wspace=0.32)

    # ---- A. 总览 ----
    ax = fig.add_subplot(grid[0, :])
    tasks = [
        ("Glycosylation", glycosylation, COLOR_GREEN),
        ("Immunogenicity", immunogenicity, COLOR_ORANGE),
        ("GlycanML Taxonomy\n(8 levels, avg)", int(m_avg), COLOR_NAVY),
        ("Glycosmos Taxonomy\n(8 levels, avg)", int(g_avg), COLOR_BLUE),
    ]
    y = list(range(len(tasks)))[::-1]  # 第一项在最上方
    values = [t[1] for t in tasks]
    ax.barh(y, values, color=[t[2] for t in tasks], height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, f"{v:,}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in tasks], fontsize=10)
    ax.set_xlabel("Average Samples per Level / Total Samples", fontsize=10)
    ax.set_title("A. Downstream Task Overview", fontsize=13)
    ax.set_xlim(0, max(values) * 1.18)

    # 每行左右两图共用纵坐标，分类标签放在两图中间的窄缝里
    _mirrored_pair(
        fig, grid[1, :].subgridspec(1, 3, width_ratios=[1, 0.26, 1], wspace=0.0),
        glycosmos,
        "B. Glycosmos: Label Count", COLOR_ORANGE,
        "C. Glycosmos: Sample Count", COLOR_BLUE)
    _mirrored_pair(
        fig, grid[2, :].subgridspec(1, 3, width_ratios=[1, 0.26, 1], wspace=0.0),
        glycanml,
        "D. GlycanML: Label Count", COLOR_ORANGE,
        "E. GlycanML: Sample Count", COLOR_NAVY)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"图已保存: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="下游任务样本统计生图")
    parser.add_argument("--csv", default=str(BASE_DIR / "downstream_task_stats.csv"),
                        help="数值明细 CSV 路径")
    parser.add_argument("--out", default=str(BASE_DIR / "downstream_task_stats.png"),
                        help="输出 PNG 路径")
    args = parser.parse_args()

    glycosmos, glycanml, totals, averages = load_stats(args.csv)
    render(glycosmos, glycanml, totals, averages, args.out)


if __name__ == "__main__":
    main()
