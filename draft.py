# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.1.csv')

# # 截取前xxx行
# df_headxxx = df.head(7500)

# # 保存为新的CSV文件
# df_headxxx.to_csv('main_df_4.1_head7500.csv', index=False)




# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_renamed.csv')

# # 获取holiday字段的所有唯一取值
# unique_holiday = df['holiday'].unique()
# print(unique_holiday)


# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_v3.0.csv')

# # 提取item_nbr列的所有唯一取值
# item_nbr_values = df['item_nbr'].unique()

# # 打印所有item_nbr的取值
# print(item_nbr_values)

# import pandas as pd

# # 你的商品序号列表
# item_nbr_list = [
#     402175, 459804, 759694, 1124165, 108952, 655749, 1132005, 1005465, 119193,
#     660310, 1686685, 1313223, 1472479, 1354390, 1354383, 1349808, 1441514, 1471460,
#     2010456, 2048246
# ]

# # 读取CSV文件
# df = pd.read_csv('main_df_v3.0.csv')

# # 只保留item_nbr在你的列表中的数据
# df_filtered = df[df['item_nbr'].isin(item_nbr_list)]

# # 按item_nbr分组，每组取第一行
# first_occurrence = df_filtered.groupby('item_nbr', as_index=False).first()

# # 打印或保存结果
# print(first_occurrence)





# # # 输出项目结构
# import os

# IGNORED = {"__pycache__", "node_modules", ".git", ".venv", "build", "dist", ".next", ".idea", ".vscode"}

# def list_dir_structure(startpath=".", output_file="project_structure.txt", max_depth=6):
#     with open(output_file, "w", encoding="utf-8") as f:
#         for root, dirs, files in os.walk(startpath):
#             depth = root.replace(startpath, "").count(os.sep)
#             if depth >= max_depth:
#                 dirs[:] = []  # 不再深入
#                 continue
#             dirs[:] = [d for d in dirs if d not in IGNORED]
#             indent = " " * 4 * depth
#             f.write(f"{indent}{os.path.basename(root)}/\n")
#             subindent = " " * 4 * (depth + 1)
#             for file in files:
#                 f.write(f"{subindent}{file}\n")

# list_dir_structure()






# import pandas as pd

# # 读取原始 CSV 文件
# df = pd.read_csv("main_df_4.1.csv")

# # 筛选 store_nbr 为 1 的行
# filtered_df = df[df["store_nbr"] == 1]

# # 将结果保存为新的 CSV 文件
# filtered_df.to_csv("main_df_4.2.csv", index=False)

# print("筛选完成，已保存为 main_df_4.2.csv")




# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.2.csv')

# # 保留标题行和第352行及其以后的数据（索引从0开始，所以351是第352行）
# df_new = df.iloc[10739:]

# # 保存到新CSV文件
# df_new.to_csv('main_df_4.2_1M.csv', index=False)



# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.2_1M.csv')

# # 替换item_category列的值
# df['category'] = df['category'].replace('GROCERY I', 'Pantry Staples')

# # 保存回CSV文件
# df.to_csv('main_df_4.3_1M.csv', index=False)



# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.3_1M.csv')

# # 删除指定的列
# df = df.drop(columns=['item_category'])

# # 保存为新的CSV文件（可以覆盖原文件）
# df.to_csv('main_df_4.4_1M.csv', index=False)


# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.2.csv')

# # 保留第7150行之后的所有数据（注意索引从0开始，因此是7149之后）
# new_df = df.iloc[7150:]

# # 保存为新文件
# new_df.to_csv('main_df_4.6.csv', index=False)



# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('main_df_4.6.csv', parse_dates=['date'])

# # 添加 weekday 列
# df['weekday'] = df['date'].dt.day_name()

# # 删除 item_category 列
# df.drop(columns=['item_category'], inplace=True)

# # 调整 item_name 到 item_nbr 后面
# cols = list(df.columns)
# # 找到 item_nbr 和 item_name 的位置
# item_nbr_index = cols.index('item_nbr')
# # 移除 item_name，然后插入到 item_nbr 后面
# cols.remove('item_name')
# cols.insert(item_nbr_index + 1, 'item_name')

# # 重新排列列顺序
# df = df[cols]

# # 保存修改后的CSV
# df.to_csv('main_df_4.6.2.csv', index=False)


# import pandas as pd

# # 读取数据
# df = pd.read_csv('main_df_4.6.2.csv')

# # 重新排序列
# new_order = [
#     'id', 'date', 'weekday', 'store_nbr', 'item_nbr', 'item_name',
#     'unit_sales', 'onpromotion', 'category', 'holiday',
#     'item_class', 'perishable', 'cost_price', 'price'
# ]

# df = df[new_order]

# # 保存结果
# df.to_csv('main_df_4.6.3.csv', index=False)



# import pandas as pd

# # 读取两个文件
# main_df = pd.read_csv('main_df_4.6.3.csv')
# weather_df = pd.read_csv('weather_20160715_20170815.csv')

# # 统一日期格式（防止格式不一致）
# main_df['date'] = pd.to_datetime(main_df['date'])
# weather_df['Date'] = pd.to_datetime(weather_df['Date'])

# # 重命名天气列
# weather_df = weather_df.rename(columns={'Date': 'date', 'Avg.(°F)': 'weather'})

# # 合并两个数据集，按日期
# merged_df = pd.merge(main_df, weather_df[['date', 'weather']], on='date', how='left')

# # 调整列顺序：将 'weather' 移动到 'weekday' 前
# cols = merged_df.columns.tolist()
# weekday_index = cols.index('weekday')
# # 先移除 'weather'，再插入到 weekday 前面
# cols.insert(weekday_index, cols.pop(cols.index('weather')))
# merged_df = merged_df[cols]

# # 保存结果
# merged_df.to_csv('main_df_with_weather.csv', index=False)
