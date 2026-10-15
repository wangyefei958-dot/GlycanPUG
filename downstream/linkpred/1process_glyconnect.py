
import csv
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 文件路径
glyconnect_file = os.path.join(script_dir, 'glyconnect_structures.csv')
dic_file = os.path.join(script_dir, 'dic.csv')
output_file = os.path.join(script_dir, 'glyconnect_structures.csv')

# 1. 读取 dic.csv，构建 GlyTouCan ID -> WURCS 的映射
dic_map = {}
with open(dic_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header_dic = next(reader)  # 跳过表头
    for row in reader:
        if len(row) >= 2:
            dic_map[row[0]] = row[1]

# 2. 读取 glyconnect_structures.csv
rows = []
with open(glyconnect_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        # 删掉第一列或第二列为空的行，或第一列是"-"的行
        if len(row) < 2 or not row[0].strip() or not row[1].strip() or row[0].strip() == '-':
            continue
        # 新增 WURCS 列
        glytoucan_id = row[0].strip()
        wurcs = dic_map.get(glytoucan_id, '')
        row.append(wurcs)
        rows.append(row)

# 3. 写入文件
# 表头去掉引号
new_header = [col.replace('"', '').strip() for col in header]
new_header.append('WURCS')


def needs_quote(val):
    """判断值是否需要加引号"""
    return ',' in val or '"' in val or '\n' in val


def format_cell(val, is_first_col):
    """格式化单元格：第一列不加引号，其余列按需加引号"""
    if is_first_col:
        return val.replace('"', '').strip()
    if needs_quote(val):
        # 转义内部引号
        escaped = val.replace('"', '""')
        return f'"{escaped}"'
    return val


with open(output_file, 'w', encoding='utf-8', newline='') as f:
    # 写表头（无引号）
    f.write(','.join(new_header) + '\n')
    # 写数据行
    for row in rows:
        parts = []
        for i, val in enumerate(row):
            parts.append(format_cell(val, i == 0))
        f.write(','.join(parts) + '\n')

print(f"处理完成！共保留 {len(rows)} 行数据，输出文件：{output_file}")
