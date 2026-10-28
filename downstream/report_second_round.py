# -*- coding: utf-8 -*-
"""第二轮转换失败原因统计报告

第二轮 = 第一轮转换失败、进入 errors.csv 的那批结构。
统计每个失败原因：
  - 占第二轮比例 = 该原因数量 / errors.csv 行数
  - 占全部数据比例 = 该原因数量 / 原始数据集行数

用法：
  python report_second_round.py              # 统计 imme 和 tax(ML) 两份
  python report_second_round.py imme         # 只统计指定数据集
"""

import csv
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    {
        'name': 'imme (免疫原性)',
        'dir': os.path.join(BASE_DIR, 'imme'),
        'total': 'glycan_immunogenicity.csv',
        'errors': 'glycan_immunogenicity_errors.csv',
        'processed': 'glycan_immunogenicity_errors_processed.csv',
        'key': 'glycan',
    },
    {
        'name': 'tax(ML) (分类)',
        'dir': os.path.join(BASE_DIR, 'tax(ML)', 'allstata'),
        'total': 'glycan_classification.csv',
        'errors': 'glycan_classification_errors.csv',
        'processed': 'glycan_classification_errors_processed.csv',
        'key': 'target',
    },
]


def read_rows(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def get_status(row):
    """兼容新旧列名"""
    return (row.get('status') or row.get('error_reason') or '').strip()


def classify(status):
    """把一条status归类为(原因类别, 细分原因)。成功返回None"""
    s = status.strip()
    if not s or s == 'SUCCESS':
        return None

    if s.startswith('A+OAc collision'):
        return ('占位符碰撞', 'A与OAc共用?F占位符')
    if s.startswith('INVALID_INPUT'):
        return ('输入非法（本地拦截）', '括号/语法错误')
    if s.startswith('MISMATCH'):
        return ('结构不一致', '树同构比较失败')
    if s.startswith('PARSE_FAIL'):
        return ('本地解析失败', '回转结果无法建树')
    if 'Timeout' in s or 'timed out' in s.lower():
        return ('网络异常', '请求超时')

    if s.startswith('ERROR'):
        if 'HTTP 500' in s:
            return ('API服务端错误', 'HTTP 500')
        if 'No WURCS found' in s:
            low = s.lower()
            if 'monosaccharide size could not' in low:
                return ('IUPAC→WURCS失败', '残基尺寸无法识别')
            if 'only one modification is allowed' in low:
                return ('IUPAC→WURCS失败', '同一碳位存在多个修饰')
            if 'no ring making anomer' in low:
                return ('IUPAC→WURCS失败', '缺少环化异头信息')
            if 'could not completely parsed' in low:
                return ('IUPAC→WURCS失败', '罕见取代基/修饰无法解析')
            return ('IUPAC→WURCS失败', '其他解析错误')
        if 'No IUPAC found' in s:
            low = s.lower()
            if 'ketose' in low:
                return ('WURCS→IUPAC回转失败', '酮糖异头态不明')
            if 'multiple root' in low:
                return ('WURCS→IUPAC回转失败', '多根结构')
            return ('WURCS→IUPAC回转失败', '其他回转错误')
        return ('API异常', '未归类错误')

    return ('未分类', s[:60])


def collect_rare_tokens(statuses):
    """从'罕见取代基无法解析'的报错中提取具体的取代基名"""
    tokens = Counter()
    for s in statuses:
        m = re.search(r'could not completely parsed\s*:\s*(\S+)', s)
        if m:
            tokens[m.group(1)] += 1
    return tokens


def report_dataset(ds):
    total_path = os.path.join(ds['dir'], ds['total'])
    errors_path = os.path.join(ds['dir'], ds['errors'])
    processed_path = os.path.join(ds['dir'], ds['processed'])

    if not os.path.exists(total_path) or not os.path.exists(errors_path):
        return None, f"缺少输入文件: {ds['total']} 或 {ds['errors']}"

    n_total = len(read_rows(total_path))
    n_second = len(read_rows(errors_path))

    if not os.path.exists(processed_path):
        return None, (f"第二轮尚未处理（缺少 {ds['processed']}），"
                      f"当前只能确认：全部 {n_total} 行中有 {n_second} 行第一轮失败")

    processed = read_rows(processed_path)
    statuses = [get_status(r) for r in processed]
    n_processed = len(processed)

    buckets = Counter()
    for s in statuses:
        c = classify(s)
        if c:
            buckets[c] += 1

    n_success = sum(1 for s in statuses if s == 'SUCCESS')
    n_fail = n_processed - n_success
    rare = collect_rare_tokens(statuses)

    lines = []
    lines.append(f"## {ds['name']}")
    lines.append('')
    lines.append(f"- 全部数据：{n_total} 行")
    lines.append(f"- 第一轮直接成功：{n_total - n_second} 行"
                 f"（{(n_total - n_second) / n_total:.2%}）")
    lines.append(f"- 第二轮输入（第一轮失败）：{n_second} 行（占全部 {n_second / n_total:.2%}）")
    lines.append(f"- 第二轮成功：{n_success} 行（占第二轮 {n_success / n_second:.2%}，"
                 f"占全部 {n_success / n_total:.2%}）")
    lines.append(f"- 第二轮仍失败：{n_fail} 行（占第二轮 {n_fail / n_second:.2%}，"
                 f"占全部 {n_fail / n_total:.2%}）")
    if n_processed != n_second:
        lines.append(f"- 注意：processed 文件 {n_processed} 行 ≠ 第二轮输入 {n_second} 行，"
                     f"可能未跑完，比例以第二轮输入数 {n_second} 为准")
    lines.append('')
    lines.append('| 原因类别 | 细分原因 | 数量 | 占第二轮 | 占全部数据 |')
    lines.append('|---|---|---:|---:|---:|')

    for (cat, sub), n in sorted(buckets.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {cat} | {sub} | {n} | {n / n_second:.2%} | {n / n_total:.2%} |")
    lines.append(f"| **合计** | | **{n_fail}** | **{n_fail / n_second:.2%}** | "
                 f"**{n_fail / n_total:.2%}** |")
    lines.append('')

    if rare:
        top = '、'.join(f"{t}×{n}" for t, n in rare.most_common(12))
        lines.append(f"罕见取代基明细（前12）：{top}")
        lines.append('')

    return lines, None


def run(selected=None):
    out = ['# 第二轮转换失败原因报告', '']
    if selected:
        todo = [ds for ds in DATASETS if selected.lower() in ds['name'].lower()
                or selected.lower() in ds['dir'].lower()]
        if not todo:
            print(f"未找到数据集: {selected}")
            return
    else:
        todo = DATASETS

    for ds in todo:
        lines, msg = report_dataset(ds)
        if lines is None:
            print(f"[{ds['name']}] {msg}")
            out.extend([f"## {ds['name']}", '', msg, ''])
        else:
            for ln in lines:
                if ln.startswith('- ') or ln.startswith('|'):
                    print(ln)
                elif ln.startswith('## '):
                    print('\n' + ln)
            out.extend(lines)

    report_path = os.path.join(BASE_DIR, 'second_round_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"\n报告已保存: {report_path}")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
