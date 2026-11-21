# -*- coding: utf-8 -*-
"""
统计与脚本同目录下的所有 CSV 文件：
- 有效数据行数（去掉表头，且至少有三列）
- 独立标签数（第三列按下划线分割后去重）
输出结果同时保存为 result.txt
"""

import csv
import sys
from pathlib import Path

# 强制 stdout 使用 UTF-8，防止中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def main():
    # 脚本所在的目录（绝对路径）
    script_dir = Path(__file__).resolve().parent
    print(f"正在扫描目录: {script_dir}\n")

    # 找同目录下所有 .csv 文件
    csv_files = sorted(script_dir.glob("*.csv"))
    if not csv_files:
        print("[提示] 当前目录下没有找到任何 .csv 文件！")
        input("按回车键退出...")
        return

    results = []
    all_unique_labels = set()

    for csv_path in csv_files:
        row_count = 0
        unique_labels = set()

        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # 跳过表头
                except StopIteration:
                    continue

                for row in reader:
                    if len(row) >= 3:  # 确保有第三列
                        row_count += 1
                        label_col = row[2].strip()  # 取第三列
                        if label_col:
                            labels = [lbl.strip() for lbl in label_col.split("_") if lbl.strip()]
                            unique_labels.update(labels)

            all_unique_labels.update(unique_labels)
            results.append((csv_path.name, row_count, len(unique_labels)))

        except Exception as e:
            results.append((csv_path.name, -1, 0))
            print(f"  [错误] 处理 {csv_path.name} 时出错: {e}")

    # 打印表格
    header = f"{'文件名':<30} {'有效数据行数':>12} {'独立标签数':>10}"
    print(header)
    print("-" * 55)
    for name, rows, labels in results:
        if rows == -1:
            print(f"{name:<30} {'读取失败':>12} {'--':>10}")
        else:
            print(f"{name:<30} {rows:>12,} {labels:>10,}")

    print("-" * 55)
    total_rows = sum(r for n, r, l in results if r > 0)
    print(f"{'合计（跨文件去重标签）':<30} {total_rows:>12,} {len(all_unique_labels):>10,}")
    print("=" * 55)
    print(f"共处理 {len(csv_files)} 个 CSV 文件")

    # 保存结果到同目录下的 result.txt
    result_path = script_dir / "result.txt"
    with result_path.open("w", encoding="utf-8") as out:
        out.write(header + "\n")
        out.write("-" * 55 + "\n")
        for name, rows, labels in results:
            if rows == -1:
                out.write(f"{name:<30} {'读取失败':>12} {'--':>10}\n")
            else:
                out.write(f"{name:<30} {rows:>12,} {labels:>10,}\n")
        out.write("-" * 55 + "\n")
        out.write(f"{'合计（跨文件去重标签）':<30} {total_rows:>12,} {len(all_unique_labels):>10,}\n")

    print(f"\n结果已保存到: {result_path}")
    input("按回车键退出...")

if __name__ == "__main__":
    main()