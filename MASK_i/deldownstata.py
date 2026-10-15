import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DIC_FILE = BASE_DIR / "dic.csv"
DOWNSTREAM_DIR = BASE_DIR / "downstreamstata"

# 存储每个 downstream 文件贡献的独特 WURCS
file_unique_wurcs = defaultdict(set)
all_downstream_wurcs = set()

print("🔍 扫描 downstreamstata 文件夹...")
for csv_file in DOWNSTREAM_DIR.glob("*.csv"):
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wurcs_col = next(
                (col for col in row if col.strip().lower() == "wurcs"), None
            )
            if wurcs_col and row[wurcs_col].strip():
                val = row[wurcs_col].strip()
                # 记录到全局集合
                all_downstream_wurcs.add(val)
                # 记录到该文件
                file_unique_wurcs[csv_file.name].add(val)

print(f"✅ 共发现 {len(all_downstream_wurcs)} 个独特的 WURCS\n")

# 打印每个文件的贡献
print("📁 各文件贡献的独特 WURCS 数量：")
for fname, wurcs_set in sorted(file_unique_wurcs.items()):
    print(f"   {fname}: {len(wurcs_set)} 个")

# 读取 dic.csv
with open(DIC_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    original_rows = list(reader)

total_before = len(original_rows)
removed_rows = []
kept_rows = []

print(f"\n🔄 处理 dic.csv（原始 {total_before} 行）...")

for row in original_rows:
    wurcs_col = next(
        (col for col in row if col.strip().lower() == "wurcs"), None
    )
    if wurcs_col and row[wurcs_col].strip() in all_downstream_wurcs:
        removed_rows.append(row)
    else:
        kept_rows.append(row)

total_removed = len(removed_rows)
total_after = len(kept_rows)

# 统计哪些文件“命中”了被删除的行
hit_files = defaultdict(int)
for row in removed_rows:
    wurcs_col = next(
        (col for col in row if col.strip().lower() == "wurcs"), None
    )
    if wurcs_col:
        wurcs_val = row[wurcs_col].strip()
        # 查找这个值来自哪些文件
        for fname, wurcs_set in file_unique_wurcs.items():
            if wurcs_val in wurcs_set:
                hit_files[fname] += 1

print(f"\n📊 删除统计：")
print(f"   原始行数: {total_before}")
print(f"   删除行数: {total_removed}")
print(f"   剩余行数: {total_after}")

if hit_files:
    print(f"\n🎯 各文件命中并导致删除的行数：")
    for fname, count in sorted(hit_files.items(), key=lambda x: -x[1]):
        print(f"   {fname}: {count} 行")
else:
    print("\n⚠️ 没有文件命中任何行")

# 写回 dic.csv
with open(DIC_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print("\n✅ dic.csv 更新完成")