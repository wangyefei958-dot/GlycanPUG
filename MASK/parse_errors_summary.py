"""汇总 parse_errors.csv 中的错误类型（按归一化后的错误信息分类）。"""

import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "parse_errors.csv")

PAT_INT_PLUS = re.compile(r"invalid literal for int\(\) with base 10: '(\d+)\+'")
PAT_INT_OTHER = re.compile(r"invalid literal for int\(\) with base 10: '([^']*)'")
PAT_LINKAGE = re.compile(r"Invalid linkage format: cannot parse")
PAT_LINKAGE_VALUE = re.compile(r"cannot parse '([^']*)'")
PAT_KEY = re.compile(r"KeyError: (.*)")


def pad(text, width):
    """按显示宽度补空格（中文等全角字符按 2 列算）。"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)


def classify(msg):
    """把原始错误信息归一化为错误类型，返回 (类型名, 细节)。"""
    m = PAT_INT_PLUS.match(msg)
    if m:
        if m.group(1) == "0":
            return "linkage 计数未知（头部为 0+，残基模板含 x/? 等不确定信息）", "'0+'"
        return "linkage 计数带 '+'（如 11+，计数为下限/不精确）", f"'{m.group(1)}+'"
    m = PAT_INT_OTHER.match(msg)
    if m:
        return "int() 无法解析的字段值", f"'{m.group(1)}'"
    if PAT_LINKAGE.match(msg):
        return "linkage 字符串无法解析（aXnY 语法不支持）", ""
    m = PAT_KEY.match(msg)
    if m:
        return "KeyError（键不存在）", m.group(1).strip()
    return msg.split(":", 1)[0].strip()[:80], ""


def classify_linkage(link_str):
    """把无法解析的 linkage 串按语法不支持的子类型归类。"""
    if "~" in link_str:
        return "含 ~（重复单元）"
    if "n" in link_str:
        return "n：连接位置未知（如 a3n2-b1n1）"
    if re.search(r"[du](?=-|$)", link_str):
        return "d/u：位置构型未定（如 a11d-b1、a1u-b4d）"
    if re.search(r"x(?=-|$)", link_str):
        return "x：位置构型未知（如 b1x-f3、a2x-b2x）"
    return "其他"


def main():
    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    type_counts = Counter()
    detail_counts = Counter()
    samples = defaultdict(list)
    examples = defaultdict(list)

    for r in rows:
        ctype, detail = classify(r.get("error", "") or "")
        type_counts[ctype] += 1
        if detail:
            detail_counts[(ctype, detail)] += 1
        gid = r.get("glytoucan_id") or ""
        if len(samples[ctype]) < 5 and gid not in samples[ctype]:
            samples[ctype].append(gid)
        if len(examples[ctype]) < 3:
            wurcs = (r.get("WURCS") or "").strip()
            examples[ctype].append((gid, wurcs[:100] + ("…" if len(wurcs) > 100 else "")))

    print(f"总错误行数: {total}")
    print(f"归一化后错误类型: {len(type_counts)}\n")
    print(f"{pad('类型', 52)}{'数量':>8}{'占比':>9}  示例ID")
    print("-" * 100)
    for ctype, n in type_counts.most_common():
        print(f"{pad(ctype, 52)}{n:>8}{n / total * 100:>8.2f}%  {', '.join(samples[ctype])}")

    print("\n--- 示例 ---")
    for ctype, _ in type_counts.most_common():
        print(f"[{ctype}]")
        for gid, wurcs in examples[ctype]:
            print(f"  {gid}: {wurcs}")

    link_strs = []
    for r in rows:
        m = PAT_LINKAGE_VALUE.search(r.get("error", "") or "")
        if m:
            link_strs.append(m.group(1))

    if link_strs:
        print("\n--- linkage 语法不支持细分 ---")
        print(f"共 {len(link_strs)} 条，{len(set(link_strs))} 种不同的 linkage 串")
        print(f"其中含 '~'（重复单元）: {sum('~' in s for s in link_strs)} 条"
              "（解析时在 wurcs_parser_reverse_edge.py 里直接跳过，不写入本文件）")
        linkage_counts = Counter(classify_linkage(s) for s in link_strs)
        samples_by_sub = defaultdict(list)
        for s in link_strs:
            sub = classify_linkage(s)
            if s not in samples_by_sub[sub] and len(samples_by_sub[sub]) < 2:
                samples_by_sub[sub].append(s)
        for sub, n in linkage_counts.most_common():
            print(f"{pad(sub, 44)}{n:>5}  {'; '.join(samples_by_sub[sub])}")

        n_group = [s for s in link_strs if classify_linkage(s).startswith("n：")]
        if n_group:
            print(f"  └ n 类中同时含 取代基 *: {sum('*' in s for s in n_group)}，"
                  f"环闭合 $: {sum('$' in s for s in n_group)}，"
                  f"分支括号（: {sum('(' in s for s in n_group)}，"
                  f"双键 ^: {sum('^' in s for s in n_group)}")

    print("\n--- 细分（仅列出出现 >= 5 次的细节）---")
    for (ctype, detail), n in detail_counts.most_common():
        if n >= 5:
            print(f"{ctype} | {detail} | {n}")


if __name__ == "__main__":
    main()
