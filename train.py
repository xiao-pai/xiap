import pandas as pd
from scipy.optimize import linprog

# 假设从附件中读取数据，这里需要根据实际数据格式进行调整
# data = pd.read_excel('data.xlsx')

# 假设已经根据附件 1 和附件 2 制定了以下常量
# 例子：每种作物的种植成本，亩产量，预期销售量，销售价格
crops = ['小麦', '玉米', '水稻', '蔬菜', '食用菌']
cost = [200, 250, 300, 150, 100]  # 种植成本（元/亩）
yield_per_acre = [600, 800, 500, 700, 300]  # 亩产量（斤）
expected_sales = [12000, 10000, 8000, 15000, 7000]  # 预期销售量（斤）
sale_price = [3, 2.5, 3.5, 4, 5]  # 销售价格（元/斤）

# 农田数据
total_land = 1201  # 总耕地面积（亩）


# 优化目标：最小化种植成本或最大化收益
# 这里以最大化收益为例
def solve_optimization(overproduce_sell=None):
    # 各作物的变量
    num_crops = len(crops)

    # 目标函数：收益（减去成本）
    c = [-1 * (sale_price[i] * yield_per_acre[i] for i in range(num_crops))]  # 减去种植成本
    A_eq = [[1 for _ in range(num_crops)]]  # 需限制总面积
    b_eq = [total_land]  # 总面积等于总耕地面积
    bounds = [(0, None) for _ in range(num_crops)]  # 每种作物种植面积大于等于0

    # 定义超卖的情况
    if overproduce_sell is not None:
        # 添加额外限制，考虑超过预期销售量的处理
        A_ub = []
        b_ub = []
        for i in range(num_crops):
            if overproduce_sell == '滞销':
                A_ub.append([-yield_per_acre[i]])
                b_ub.append(-expected_sales[i])
            elif overproduce_sell == '降价':
                # 降价条件，收益计算需要调整
                A_ub.append([-yield_per_acre[i] * 0.5])  # 降价的情况下处理
                b_ub.append(-expected_sales[i] * 0.5)

        # 使用线性规划求解
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        return result


# 对于两种情况分别求解
result1 = solve_optimization(overproduce_sell='滞销')
result2 = solve_optimization(overproduce_sell='降价')

# 见完整版
