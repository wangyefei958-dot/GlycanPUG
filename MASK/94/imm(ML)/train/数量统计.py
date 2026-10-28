# -*- coding: utf-8 -*-
"""
统计本脚本所在目录下（不含子文件夹）所有 CSV 文件的数据行数。
行数 = 文件总行数 - 1（去掉表头）。Python 3.13 可直接运行。

用法:
    python count_csv_rows.py
    python count_csv_rows.py > result.txt          # 结果存文件
    python count_csv_rows.py -d "D:/data"          # 指定其他目录
    python count_csv_rows.py --csv                 # 精确模式（按 CSV 记录计数，可处理字段内换行）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def count_rows(path: Path, precise: bool = False) -> int:
    """返回数据行数（已减掉表头）。文件为空或无数据行时返回 0。"""
    if precise:
        try:
            with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
                return max(sum(1 for row in csv.reader(f) if row) - 1, 0)
        except OSError:
            return -1

    try:
        # errors="replace" 只影响显示，不影响行数；utf-8-sig 自动吃掉 BOM
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            return max(sum(1 for line in f if line.strip()) - 1, 0)
    except UnicodeDecodeError:
        # 极端情况：非 UTF-8 编码，退回二进制按换行符计数，结果一致
        data = path.read_bytes()
        return max(data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1) - 1, 0)
    except OSError:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="统计目录下 CSV 文件的数据行数（不含表头）")
    parser.add_argument("-d", "--dir", default=None, help="目标目录，默认使用本脚本所在目录")
    parser.add_argument("--csv", action="store_true", help="精确模式：按 CSV 记录计数（处理字段内换行）")
    args = parser.parse_args()

    base = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent

    if not base.is_dir():
        print(f"[错误] 目录不存在: {base}")
        return

    # 只看当前目录，不递归子文件夹
    files = sorted(
        (p for p in base.glob("*.csv") if p.is_file()),
        key=lambda p: p.name.lower(),
    )

    if not files:
        print(f"[提示] {base} 下没有找到 .csv 文件")
        return

    rows: list[tuple[str, int]] = []
    for p in files:
        rows.append((p.stem, count_rows(p, precise=args.csv)))

    name_w = max(len(n) for n, _ in rows)
    name_w = max(name_w, len("文件名"))

    print("=" * (name_w + 14))
    print(f"目录: {base}")
    print(f"模式: {'精确 (csv 记录)' if args.csv else '快速 (行计数)'}")
    print("=" * (name_w + 14))
    print(f"{'文件名':<{name_w}}  {'数据行数':>10}")
    print("-" * (name_w + 14))

    for name, n in rows:
        shown = "读取失败" if n < 0 else f"{n:,}"
        print(f"{name:<{name_w}}  {shown:>10}")

    print("-" * (name_w + 14))
    total = sum(n for _, n in rows if n > 0)
    failed = sum(1 for _, n in rows if n < 0)
    print(f"{'合计':<{name_w}}  {total:>10,}")
    print(f"{'文件数':<{name_w}}  {len(rows):>10}")
    if failed:
        print(f"（{failed} 个文件读取失败，未计入合计）")
    print("=" * (name_w + 14))


if __name__ == "__main__":
    main()