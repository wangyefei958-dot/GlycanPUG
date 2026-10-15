import pandas as pd

# 读取CSV文件
df = pd.read_csv('glyconnect_structures.csv')

print(f"原始行数: {len(df)}")

# 删除WURCS列中含有~符号的行
mask = df['WURCS'].str.contains('~', na=False)
removed_count = mask.sum()
df_filtered = df[~mask]
# 删除GlyTouCan ID列中值为G18907LN的行
mask_id = df_filtered['GlyTouCan ID'] == 'G18907LN'
removed_id_count = mask_id.sum()
df_filtered = df_filtered[~mask_id]

print(f"删除GlyTouCan ID为G18907LN的行数: {removed_id_count}")
print(f"删除的行数: {removed_count}")
print(f"剩余行数: {len(df_filtered)}")

# 保存回原文件
df_filtered.to_csv('creature_0wave.csv', index=False)

print("处理完成，已保存到 creature_0wave.csv")
