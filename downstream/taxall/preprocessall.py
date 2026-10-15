#!/usr/bin/env python3
"""
merge_glyco_final.py
✅ 修改 data 里的 CSV（添加 WURCS 列）
✅ 删 ~ → 删 G18907LN → 检查行数 → 基于 WURCS 合并
✅ 标签列名统一去掉前缀数字
✅ 输出文件名和标签列名一致，不带 _merged
✅ 每一步都有详细输出
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).parent.resolve()

for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue

    dic_files = list(folder.glob("dic.*"))
    data_dir = folder / "data"
    if not dic_files or not data_dir.exists():
        continue

    dic_path = dic_files[0]

    # 读 dic
    if dic_path.suffix == ".csv":
        dic = pd.read_csv(dic_path)
    elif dic_path.suffix == ".tsv":
        dic = pd.read_csv(dic_path, sep="\t")
    elif dic_path.suffix == ".xlsx":
        dic = pd.read_excel(dic_path)
    else:
        continue

    dic.columns = dic.columns.str.strip()

    glyco_col = "GlyTouCan ID"
    wurcs_col = "WURCS"

    if glyco_col not in dic.columns or wurcs_col not in dic.columns:
        print(f"⚠️ 跳过（dic 列不完整）: {dic_path}")
        continue

    dic_map = dict(zip(
        dic[glyco_col].astype(str).str.strip(),
        dic[wurcs_col].astype(str).str.strip()
    ))

    # ✅ 标签列名：统一去掉前缀数字
    label_name = re.sub(r'^\d+', '', folder.name)

    processed_dfs = []
    skipped_files = []

    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"❌ 读取失败: {csv_file} → {e}")
            continue

        df.columns = df.columns.str.strip()

        if glyco_col not in df.columns:
            print(f"⚠️ 跳过（缺少 {glyco_col}）: {csv_file.name}")
            skipped_files.append(csv_file.name)
            continue

        original_count = len(df)

        # ✅ 第一步：直接加 WURCS 列
        df[wurcs_col] = df[glyco_col].map(dic_map)
        df.to_csv(csv_file, index=False)
        print(f"✅ {csv_file.name}（添加 WURCS 列，匹配 {df[wurcs_col].notna().sum()}/{original_count}）")

        # ✅ 第二步：删除含 ~ 的行
        mask_tilde = np.zeros(len(df), dtype=bool)
        for col in df.columns:
            mask_tilde = mask_tilde | df[col].astype(str).str.contains('~', na=False)
        tilde_count = int(mask_tilde.sum())
        df = df[~mask_tilde]

        # ✅ 第三步：删除 G18907LN
        mask_g18907 = df[glyco_col].astype(str).str.strip() == "G18907LN"
        g18907_count = int(mask_g18907.sum())
        df = df[~mask_g18907]

        remaining = len(df)

        # ✅ 输出删除信息
        if remaining < 30:
            print(f"🗑️ {csv_file.name}（删除~行数：{tilde_count}；删除G18907LN行数：{g18907_count}；剩余行数：{remaining}；丢弃，不足30行）")
            skipped_files.append(csv_file.name)
            continue
        else:
            print(f"📌 {csv_file.name}（删除~行数：{tilde_count}；删除G18907LN行数：{g18907_count}；剩余行数：{remaining}；保留）")

        # 添加标签列（临时，用于后续合并）
        df['_temp_label'] = csv_file.stem
        processed_dfs.append(df)

    if not processed_dfs:
        print(f"⚠️ {folder.name} 没有可用数据")
        continue

    # ✅ 基于 WURCS 合并
    print(f"\n🔄 {folder.name}: 开始基于 WURCS 合并 {len(processed_dfs)} 个 CSV...")

    combined = pd.concat(processed_dfs, ignore_index=True)

    def merge_labels(group):
        labels = group['_temp_label'].unique()
        merged_label = '_'.join(sorted(labels))
        result = {}
        for col in group.columns:
            if col != '_temp_label':
                non_null_vals = group[col].dropna()
                result[col] = non_null_vals.iloc[0] if len(non_null_vals) > 0 else None
        result[label_name] = merged_label
        return pd.Series(result)

    final_df = combined.groupby(wurcs_col, dropna=False).apply(merge_labels).reset_index(drop=True)

    if '_temp_label' in final_df.columns:
        final_df = final_df.drop(columns=['_temp_label'])

    final_df = final_df.rename(columns={wurcs_col: "WURCS"})
    final_df = final_df.loc[:, ~final_df.columns.str.contains("^unnamed", case=False)]

    # ✅ 输出文件名和标签列名一致
    out_name = f"{label_name}.csv"
    out_path = folder / out_name
    final_df.to_csv(out_path, index=False)

    print(f"\n✅ {folder.name} → {out_name}")
    print(f"   处理的 CSV 数: {len(processed_dfs)}")
    print(f"   跳过的 CSV 数: {len(skipped_files)}")
    print(f"   合并后行数: {len(final_df)}")
    print(f"   列名: {list(final_df.columns)}")
    print(f"{'='*60}")