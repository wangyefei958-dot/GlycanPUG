# -*- coding: utf-8 -*-
import csv
import os
import codecs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(SCRIPT_DIR, 'glycan_immunogenicity_final.csv')
output_file = os.path.join(SCRIPT_DIR, 'imm_stata.csv')

with codecs.open(input_file, 'r', 'utf-8') as f_in:
    reader = csv.DictReader(f_in)
    rows = list(reader)

print("Total rows: %d" % len(rows))

filtered_rows = [row for row in rows if row.get('wurcs', '').strip() != 'ERROR']
print("Rows after filtering (wurcs != ERROR): %d" % len(filtered_rows))
print("Rows removed: %d" % (len(rows) - len(filtered_rows)))

with codecs.open(output_file, 'w', 'utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print("File saved to: %s" % output_file)