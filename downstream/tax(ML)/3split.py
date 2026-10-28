import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

input_csv = os.path.join(SCRIPT_DIR, "allstata", "glycan_classification_final.csv")
output_dir = os.path.join(SCRIPT_DIR, "split_files")
log_file = os.path.join(SCRIPT_DIR, "split_log.txt")
os.makedirs(output_dir, exist_ok=True)

# 打开日志文件
log_f = open(log_file, "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_f.write(msg + "\n")

df = pd.read_csv(input_csv)
df = df[df["wurcs"] != "ERROR"].copy()

taxon_cols = [
    "species", "genus", "family", "order",
    "class", "phylum", "kingdom", "domain"
]

# 分类列标准化（原逻辑保留）
for col in taxon_cols:
    df[col] = df[col].astype(str).str.replace("_", "", regex=False)

# ✅ 新增：每个分类列独立处理（含低频标签过滤）
for col in taxon_cols:
    sub_df = df[[col, "wurcs", "glycan"]].copy()
    
    # 1. 过滤空字符串（原逻辑）
    sub_df = sub_df[sub_df[col].str.strip() != ""]
    if sub_df.empty:
        log(f"⚠️ {col}: 无有效数据，跳过")
        continue
    
    # 2. 统计当前分类列每个标签的出现次数
    label_counts = sub_df[col].value_counts()
    
    # 3. 找出出现次数<30的标签
    low_freq_labels = label_counts[label_counts < 30].index.tolist()
    
    # 4. 删除低频标签的所有行
    sub_df_filtered = sub_df[~sub_df[col].isin(low_freq_labels)]
    if sub_df_filtered.empty:
        log(f"⚠️ {col}: 所有标签出现次数均<30，无剩余数据")
        continue
    
    # 5. 基于wurcs合并（原逻辑保留）
    # 先检测是否有多个glycan对应同一个wurcs
    glycan_per_wurcs = sub_df_filtered.groupby("wurcs")["glycan"].apply(lambda x: set(x.dropna().astype(str)))
    multi_glycan_wurcs = glycan_per_wurcs[glycan_per_wurcs.apply(len) > 1]
    if len(multi_glycan_wurcs) > 0:
        log(f"⚠️ {col}: 发现 {len(multi_glycan_wurcs)} 个wurcs对应多个glycan：")
        for wurcs, glycans in multi_glycan_wurcs.items():
            log(f"   wurcs: {wurcs}")
            log(f"   glycan: {'_'.join(sorted(glycans))}")
            log("")
    
    merged = (
        sub_df_filtered
        .groupby("wurcs", as_index=False)
        .agg({
            col: lambda x: "_".join(sorted(set(x.dropna().astype(str)))),
            "glycan": lambda x: list(set(x.dropna().astype(str)))[0]  # 随机取一个
        })
    )
    
    # 6. 输出结果+被删标签
    out_path = os.path.join(output_dir, f"{col}.csv")
    merged.to_csv(out_path, index=False)
    
    if low_freq_labels:
        log(f"❌ {col} 删除低频标签（出现次数<30）：{low_freq_labels}")
    log(f"✅ {col}: 处理后 {len(merged)} 行")

log("\n🎉 完成")
log_f.close()