import pandas as pd
import numpy as np
from itertools import product
import pulp as pl

# 读取数据
df = pd.read_excel('附件1.xlsx')
df0 = pd.read_excel('附件1.xlsx', sheet_name=1)  # 假定作物名称在第二个工作表

# 地块名称
dkmc = df['地块名称'].tolist()
print(dkmc)
# 作物和土地类型的对应关系
zzgd = [
    ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦', '大麦'],
    ['水稻'],
    ['豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜 ', '青椒', '菜花', '包菜', '油麦菜', '小青菜', '黄瓜', '生菜 ', '辣椒', '空心菜', '黄心菜', '芹菜'],
    ['大白菜', '白萝卜', '红萝卜'],
    ['榆黄菇', '香菇', '白灵菇', '羊肚菌']
]
zz = [
    ['平旱地', '梯田', '山坡地'],
    ["水浇地"],
    ["水浇地第一季", "普通大棚 第一季", "智慧大棚第一季、第二季"],
    ['水浇地第二季'],
    ['普通大棚第二季']
]
dk_to_zz_index = {dk: idx for idx, dk_list in enumerate(dkmc) for dk in dk_list}
# 种植作物类型、年份
zw = df0['作物名称'].tolist()
years = ["2024", '2025', '2026', '2027', '2028', '2029', '2030']
seaborn = [1, 2]

# 读取农作物数据
df2 = pd.read_excel('附件2.xlsx', sheet_name=1, header=0)
acr = df2['亩产量/斤'].tolist()
danjia = df2['销售单价/(元/斤)'].tolist()
chengben = df2["种植成本/(元/亩)"].tolist()
zzmj = df2["种植面积/亩"].tolist()

# 每亩的最大销售收入
mbxl = [a * b for a, b in zip(acr, zzmj)]

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

# 创建LP问题
model_one = pl.LpProblem('pro_one', pl.LpMaximize)

# 决策变量
x = pl.LpVariable.dicts('Area', ((i, j, k, l) for i in range(len(dkmc)) for j in range(len(zw)) for k in range(len(years)) for l in range(len(seaborn))), lowBound=0, cat='Continuous')

# 目标函数: 最大化利润
max_profit1 = pl.lpSum([split_danjia[j][0] * acr[j] * x[(i, j, k, l)] for i, j, k, l in product(range(len(dkmc)), range(len(zw)), range(len(years)), range(len(seaborn)))])
model_one += max_profit1

# 约束条件
# 地块面积约束
for i in range(len(dkmc)):
    for k in range(len(years)):
        for l in range(len(seaborn)):
            model_one += pl.lpSum([x[(i, j, k, l)] for j in range(len(zw))]) <= zzmj[i]

# 避免连续重茬种植
for i in range(len(dkmc)):
    for j in range(len(zw)):
        for k in range(len(years) - 1):
            model_one += x[(i, j, k, 0)] + x[(i, j, k+1, 0)] <= 1

# 豆类作物种植要求
doulei = [1, 2, 3, 4, 5, 17, 18, 19]
for i in range(len(dkmc)):
    for year_group in range(0, len(years), 3):
        if year_group + 2 < len(years):
            model_one += pl.lpSum([x[(i, j, k, l)] for j in doulei for k in range(year_group, year_group + 3) for l in range(len(seaborn))]) >= zzmj[i] / 3

# 添加作物和土地类型的对应关系约束
for i in range(len(dkmc)):
    for j in range(len(zw)):
        for k in range(len(years)):
            for l in range(len(seaborn)):
                # 获取地块名称对应的土地类型列表索引
                dk_index = dk_to_zz_index.get(dkmc[i], -1)
                if dk_index == -1 or zw[j] not in zz[dk_index]:
                    model_one += x[(i, j, k, l)] == 0

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