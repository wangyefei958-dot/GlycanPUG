#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理 v12_df_species_errors.csv
- 断点续传（基于已有行数）
- 实时保存CSV
- 多线程处理
"""

import csv
import time
import requests
import re
import os
import sys
import threading
from queue import Queue

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 导入test_single_glycan的函数
from test_single_glycan import (
    normalize_naming,
    count_replacements,
    replace_OMe_with_placeholder,
    replace_A_with_placeholder,
    replace_OP_with_placeholder,
    replace_OS_with_placeholder,
    replace_OAc_with_placeholder,
    replace_NeuNAc,
    process_wurcs_remove_f,
    convert_iupac_to_wurcs,
    convert_wurcs_to_iupac,
    compare_iupac_by_composition,
    API_IUPAC2WURCS,
    API_WURCS2IUPAC,
    REQUEST_TIMEOUT
)

INPUT_CSV = os.path.join(SCRIPT_DIR, "glycan_immunogenicity_errors.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "glycan_immunogenicity_errors_processed.csv")
MAX_WORKERS = 28  # 并行线程数
REQUEST_INTERVAL = 0.5  # API请求间隔


def process_single_glycan(glycan):
    """处理单个glycan"""
    result = {
        'status': 'ERROR',
        'wurcs': '',
        'modified_iupac': '',
        'verified_iupac': '',
        'error_reason': ''
    }

    try:
        normalized_iupac = normalize_naming(glycan)
        has_A, has_OMe, has_OP, has_OS, has_P, has_OAc, has_NeuNAc, type_count = count_replacements(normalized_iupac)

        modified_iupac = replace_OMe_with_placeholder(normalized_iupac)
        if has_OAc:
            modified_iupac = replace_OAc_with_placeholder(modified_iupac)
        if has_OP:
            modified_iupac = replace_OP_with_placeholder(modified_iupac)
        if has_OS:
            modified_iupac = replace_OS_with_placeholder(modified_iupac)
        if has_NeuNAc:
            modified_iupac = replace_NeuNAc(modified_iupac)
        modified_iupac = normalize_naming(modified_iupac)
        result['modified_iupac'] = modified_iupac

        input_iupac = normalized_iupac
        input_iupac = replace_OMe_with_placeholder(input_iupac)
        if has_A:
            input_iupac = replace_A_with_placeholder(input_iupac)
        if has_OP:
            input_iupac = replace_OP_with_placeholder(input_iupac)
        if has_OS:
            input_iupac = replace_OS_with_placeholder(input_iupac)
        if has_OAc:
            input_iupac = input_iupac.replace("OAc", "?F")
        if has_NeuNAc:
            input_iupac = replace_NeuNAc(input_iupac)

        time.sleep(REQUEST_INTERVAL)
        wurcs = convert_iupac_to_wurcs(input_iupac)

        if not wurcs or not wurcs.startswith("WURCS="):
            result['error_reason'] = wurcs if wurcs else "No WURCS returned"
            return result

        processed_wurcs = process_wurcs_remove_f(wurcs, has_A)

        if has_OAc:
            processed_wurcs = processed_wurcs.replace('_?*F', '_?*OCC/3=O')

        result['wurcs'] = processed_wurcs

        time.sleep(REQUEST_INTERVAL)
        verified_iupac = convert_wurcs_to_iupac(processed_wurcs)
        verified_iupac_normalized = normalize_naming(verified_iupac)
        result['verified_iupac'] = verified_iupac_normalized

        # 使用组分比较
        match, _, _ = compare_iupac_by_composition(modified_iupac, verified_iupac_normalized)
        if match:
            result['status'] = 'SUCCESS'
        else:
            result['status'] = 'MISMATCH'
            # MISMATCH保留原来的error_reason（如果有的话）

    except Exception as e:
        result['error_reason'] = f"{type(e).__name__}: {str(e)}"

    return result


def get_existing_row_count():
    """获取已成功处理的行数（从最后一个成功行之后开始续传）"""
    if not os.path.exists(OUTPUT_CSV):
        return 0
    
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        last_ok_row = 0
        for i, row in enumerate(reader):
            # 成功的标志：wurcs字段不为空且以WURCS=开头
            wurcs = row.get('wurcs', '').strip()
            if wurcs and wurcs.startswith('WURCS='):
                last_ok_row = i + 1  # 记录成功的行号（从1开始）
        return last_ok_row


def load_existing_rows():
    """加载已存在的行"""
    existing = []
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
    return existing


def worker(task_queue, result_queue):
    """工作线程函数"""
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, glycan = task
        result = process_single_glycan(glycan)
        result_queue.put((idx, result))


def main():
    print(f"Start batch processing: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print("-" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found {INPUT_CSV}")
        sys.exit(1)

    input_rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            input_rows.append(row)

    total_count = len(input_rows)
    print(f"Total records: {total_count}")

    existing_count = get_existing_row_count()
    print(f"Already processed: {existing_count} records")

    if existing_count >= total_count:
        print("All records already processed!")
        return

    existing_rows = load_existing_rows()
    print("-" * 60)

    # 创建队列
    task_queue = Queue()
    result_queue = Queue()

    # 启动工作线程
    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker, args=(task_queue, result_queue))
        t.start()
        threads.append(t)

    # 添加任务（从断点开始）
    for idx in range(existing_count, total_count):
        glycan = input_rows[idx]['glycan']
        task_queue.put((idx, glycan))

    # 添加终止信号
    for _ in range(MAX_WORKERS):
        task_queue.put(None)

    # 收集结果
    processed_count = existing_count
    stats = {'SUCCESS': 0, 'MISMATCH': 0, 'ERROR': 0}
    last_save_time = time.time()
    SAVE_INTERVAL = 2

    all_rows = existing_rows.copy() if existing_rows else []

    while processed_count < total_count:
        try:
            idx, result = result_queue.get(timeout=5)
        except:
            continue

        row = input_rows[idx].copy()

        if result['status'] == 'SUCCESS':
            row['wurcs'] = result['wurcs']
            row['error_reason'] = ''
            stats['SUCCESS'] += 1
        elif result['status'] == 'MISMATCH':
            stats['MISMATCH'] += 1
        else:
            # ERROR: 保持原来的值
            stats['ERROR'] += 1

        all_rows.append(row)
        processed_count += 1

        print(f"[{processed_count}/{total_count}] {result['status']}")

        if time.time() - last_save_time > SAVE_INTERVAL or processed_count == total_count:
            with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            last_save_time = time.time()

    for t in threads:
        t.join()

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("-" * 60)
    print(f"Done! SUCCESS: {stats['SUCCESS']}, MISMATCH: {stats['MISMATCH']}, ERROR: {stats['ERROR']}")
    print(f"Result saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()