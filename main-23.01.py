import pandas as pd
import numpy as np
from itertools import product
import pulp as pl
a = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
     'B13', 'B14', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E2', 'E3',
     'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'F1', 'F2', 'F3', 'F4', ]
c = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
     'E12', 'E13', 'E14', 'E15', 'E16', 'F1', 'F2', 'F3', 'F4']
b = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦',
     '大麦', '水稻', '豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜 ', '青椒', '菜花', '包菜', '油麦菜',
     '小青菜', '黄瓜', '生菜 ', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜', '榆黄菇', '香菇',
     '白灵菇', '羊肚菌']

# 创建一个82x41的全零数组
data = np.zeros((len(a), len(b)))
data1 = np.zeros((len(c),len(b)))
# 将数组转换为DataFrame，并设置行名和列名
df4 = pd.DataFrame(data, index=a, columns=b)
df5 = pd.DataFrame(data1, index=c, columns=b)
df = pd.read_excel('附件1.xlsx')
df0 = pd.read_excel('附件1.xlsx', sheet_name=1)  # 假定作物名称在第二个工作表
df6 = pd.read_excel('result1_1.xlsx')

# 地块名称
dkmc = df['地块名称'].tolist()

# 种植作物类型、年份
zw = df0['作物名称'].tolist()
years = ["2024", '2025', '2026', '2027', '2028', '2029', '2030']
seaborn = [1, 2]

# 读取农作物数据
df2 = pd.read_excel('附件2.xlsx', sheet_name=1, header=0)
acr = df2['亩产量/斤'].tolist()
danjia = df2['销售单价/(元/斤)'].tolist()
zzmj = df2["种植面积/亩"].tolist()

# 分割单价数据
split_danjia = []
for price_range in danjia:
    if isinstance(price_range, str):
        amount1, amount2 = price_range.split('-')
        amount1 = float(amount1)
        amount2 = float(amount2)
        split_danjia.append([amount1, amount2])
    else:
        print(f"Error: Expected a string, got {type(price_range)} instead.")

# 读取2023年的种植情况
df1 = pd.read_excel('附件2.xlsx', sheet_name=0, header=0)
current_year_planting = df1.set_index('种植地块')['作物名称'].to_dict()

# 创建LP问题
model_one = pl.LpProblem('pro_one', pl.LpMaximize)

# 决策变量
x = pl.LpVariable.dicts('Area', ((i, j, k, l) for i in range(len(dkmc)) for j in range(len(zw)) for k in range(len(years)) for l in range(len(seaborn))), lowBound=0, cat='Continuous')

# 目标函数: 最大化利润
max_profit1 = pl.lpSum([split_danjia[i][0] * acr[i] * x[(i, j, k, l)] for i, j, k, l in product(range(len(dkmc)), range(len(zw)), range(len(years)), range(len(seaborn)))])
model_one += max_profit1

# 约束条件
# 地块面积约束
for i in range(len(dkmc)):
    for k in range(len(years)):
        for l in range(len(seaborn)):
            model_one += pl.lpSum([x[(i, j, k, l)] for j in range(len(zw))]) <= zzmj[i]

# 每种作物在单个地块的种植面积不少于该地块面积的1/3
for i in range(len(dkmc)):
    for j in range(len(zw)):
        for k in range(len(years)):
            model_one += pl.lpSum([x[(i, j, k, l)] for l in range(len(seaborn))]) >= zzmj[i] / 3

# 避免连续重茬种植
for i in range(len(dkmc)):
    for j in range(len(zw)):
        for k in range(len(years) - 1):
            model_one += x[(i, j, k, 0)] + x[(i, j, k+1, 0)] <= 1

# 豆类作物种植要求
doulei = [1, 2, 3, 4, 5, 17, 18, 19]  # 假设这些是豆类作物的索引
for i in range(len(dkmc)):
    for year_group in range(0, len(years), 3):
        if year_group + 2 < len(years):
            model_one += pl.lpSum([x[(i, j, k, l)] for j in doulei for k in range(year_group, year_group + 3) for l in range(len(seaborn))]) >= zzmj[i] / 3

# 添加2023年种植情况的约束
for i, current_crop in current_year_planting.items():
    i_index = dkmc.index(i) if i in dkmc else -1  # 检查地块名称是否存在
    if i_index != -1:
        for j, crop in enumerate(zw):
            if current_crop == crop:
                for k in range(len(years)):
                    model_one += x[(i_index, j, k, 0)] == 0

# 解决模型
model_one.solve()

# 输出结果
print("第一种情况的结果:", pl.LpStatus[model_one.status])
print("目标函数值:", pl.value(model_one.objective))

# 打印具体种植情况
print("具体种植情况:")
for i, j, k, l in product(range(len(dkmc)), range(len(zw)), range(len(years)), range(len(seaborn))):
    var_value = x[(i, j, k, l)].varValue
    if var_value > 0:
        print(f"地块 {dkmc[i]} 在 {years[k]} 年种植作物 {zw[j]}, 季节 {seaborn[l]}, 面积 {var_value:.2f} 亩")

        if seaborn[l] == 1:
            # 指定行索引
            row_index = dkmc[i]

            #

            row_number = df4.index.get_loc(row_index)
            col_index = zw[j]

            # 获取列索引在DataFrame中的列数
            col_number = df4.columns.get_loc(col_index)
            element = df6.iloc[row_number + 1, col_number + 1]
        else:
            row_index = dkmc[i]

            #

            row_number = df5.index.get_loc(row_index)
            col_index = zw[j]

            # 获取列索引在DataFrame中的列数
            col_number = df5.columns.get_loc(col_index)
            element = df6.iloc[row_number + 55, col_number + 1]
df.to_excel('modified_result1_1.xlsx')