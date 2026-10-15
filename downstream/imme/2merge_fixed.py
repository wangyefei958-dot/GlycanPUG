# -*- coding: utf-8 -*-
import csv
import os
import codecs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

original_data = []
with codecs.open(os.path.join(SCRIPT_DIR, 'glycan_immunogenicity_with_wurcs.csv'), 'r', 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_data.append(row)

fix_dict = {}
with codecs.open(os.path.join(SCRIPT_DIR, 'glycan_immunogenicity_errors_processed.csv'), 'r', 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('wurcs', '').startswith('WURCS='):
            key = (row['glycan'], row['species'].strip())
            fix_dict[key] = row['wurcs']

print("Original data rows: %d" % len(original_data))
print("Fix dict entries: %d" % len(fix_dict))

merged = 0
for row in original_data:
    if row.get('wurcs', '') == 'ERROR':
        key = (row['glycan'], row['species'].strip())
        if key in fix_dict:
            row['wurcs'] = fix_dict[key]
            merged += 1

print("Merged rows: %d" % merged)

remaining_error = sum(1 for row in original_data if row.get('wurcs', '') == 'ERROR')
print("Remaining ERROR rows: %d" % remaining_error)

success_count = sum(1 for row in original_data if row.get('wurcs', '').startswith('WURCS='))
print("Success rows: %d" % success_count)

fieldnames = ['glycan_id', 'glycan', 'species', 'immunogenicity', 'inferred_origin', 'link', 'train', 'valid', 'test', 'wurcs']
output_path = os.path.join(SCRIPT_DIR, 'glycan_immunogenicity_final.csv')
with codecs.open(output_path, 'w', 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(original_data)

print("Final file saved to: %s" % output_path)