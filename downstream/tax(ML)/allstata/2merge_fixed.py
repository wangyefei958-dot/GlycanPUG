# -*- coding: utf-8 -*-
import csv
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 读取原始数据
original_data = []
with open(os.path.join(SCRIPT_DIR, 'glycan_classification_with_wurcs.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_data.append(row)

# 读取修复后的数据，建立target+species的查找字典
fix_dict = {}
with open(os.path.join(SCRIPT_DIR, 'glycan_classification_errors_processed.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('wurcs', '').startswith('WURCS='):
            key = (row['target'], row['species'].strip())
            fix_dict[key] = row['wurcs']

print(f"原始数据总行数: {len(original_data)}")
print(f"修复字典条目数: {len(fix_dict)}")

# 合并：用修复的WURCS替换ERROR
merged = 0
for row in original_data:
    if row.get('wurcs', '') == 'ERROR':
        key = (row['target'], row['species'].strip())
        if key in fix_dict:
            row['wurcs'] = fix_dict[key]
            merged += 1

print(f"成功修复并合并的行数: {merged}")

# 统计剩余ERROR
remaining_error = sum(1 for row in original_data if row.get('wurcs', '') == 'ERROR')
print(f"剩余ERROR行数: {remaining_error}")

# 统计成功数
success_count = sum(1 for row in original_data if row.get('wurcs', '').startswith('WURCS='))
print(f"成功转换行数: {success_count}")

# 写入新文件，同时把target改成glycan
fieldnames = ['glycan', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'domain', 'wurcs']
with open(os.path.join(SCRIPT_DIR, 'glycan_classification_final.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in original_data:
        # 把target改成glycan
        new_row = {
            'glycan': row['target'],
            'species': row['species'],
            'genus': row['genus'],
            'family': row['family'],
            'order': row['order'],
            'class': row['class'],
            'phylum': row['phylum'],
            'kingdom': row['kingdom'],
            'domain': row['domain'],
            'wurcs': row.get('wurcs', '')
        }
        writer.writerow(new_row)

print(f"已生成最终文件: {os.path.join(SCRIPT_DIR, 'glycan_classification_final.csv')}")