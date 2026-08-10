# -*- coding:utf8 -*-
import numpy as np
import time
from DebugPrint import *

# 网络大小
X_MAX = 100
Y_MAX = 100

# 节点数量
N_NODES = 180

# 节点覆盖范围
R_ENB_MIN = 1
R_ENB_MAX = 3
R_GNB_MIN = 2
R_GNB_MAX = 2

#定义哈希表启用标志
HASH_TABLE_FLAG = False
debug_print(f"Hash Table Flag: {HASH_TABLE_FLAG}")

#MOSA启用标志
MOSA_ALGORITHM = True
debug_print(f"MOSA Algorithm Flag: {MOSA_ALGORITHM}")

# 遗传算法参数启用标志
FLAG_GENERATION_ALGORITHM = True
debug_print(f"Generation Algorithm Flag: {FLAG_GENERATION_ALGORITHM}")

# 定义半径和乘子作为邻区判定边界条件，由此，PCI重用距离即为邻区判定距离的2倍
R_SUM_MULTIPLIER = 1.5

# 定义EMA算法中当前估计值的权重
EMA_ALPHA = 0.8

# 定义EMA_LOSS算法中当前损失函数值的权重
EMA_LOSS = 0.8

# 定义神经网络
N_INPUT_FEATURE = N_NODES
N_HIDDEN_NEURON = 100
N_OUTPUT_FEATURE = N_NODES

# 定义训练参数
INIT_ALPHA = 0.004  # INIT_ALPHA=0.004与ALPHA_ADJUST_FACTOR=0.91配合使用可收敛至0.002
INIT_EPSILON = 0.1
INIT_GAMMA = 0.9

# 定义学习率动态调整因子
ALPHA_ADJUST_FACTOR = 0.91  # INIT_ALPHA=0.004与ALPHA_ADJUST_FACTOR=0.91配合使用可收敛至0.002

# 定义奖励动态调整因子
REWARD_ADJUST_FACTOR = 0.2

# 定义贪婪度动态调整因子
EPSILON_ADJUST_FACTOR = 1.1

# 定义优化目标列表
OPTIMIZATION_OBJECT = [
    "4gMod3",
    "4gMod6",
    "4gMod30",
    "4gConfusion",
    "5gMod3",
    "5gMod4",
    "5gMod30",
    "5gConfusion",
]

# 定义优化目标QNet字典
OPTIMIZATION_OBJECT_QNET = {
    "4gMod3": None,
    "4gMod6": None,
    "4gMod30": None,
    "4gConfusion": None,
    "5gMod3": None,
    "5gMod4": None,
    "5gMod30": None,
    "5gConfusion": None,
}

# 定义优化目标Optimizer字典
OPTIMIZATION_OBJECT_OPTIMIZER = {
    "4gMod3": None,
    "4gMod6": None,
    "4gMod30": None,
    "4gConfusion": None,
    "5gMod3": None,
    "5gMod4": None,
    "5gMod30": None,
    "5gConfusion": None,
}

# 定义经验池字典
EXPERIENCE_POOL_ELEMENT = {
    "curr_state": [],
    "curr_issue": {
        "4gMod3": None,
        "4gMod6": None,
        "4gMod30": None,
        "4gConfusion": None,
        "5gMod3": None,
        "5gMod4": None,
        "5gMod30": None,
        "5gConfusion": None,
    },
    "action": [],
    "reward": None,
    "next_state": [],
    "next_issue": {
        "4gMod3": None,
        "4gMod6": None,
        "4gMod30": None,
        "4gConfusion": None,
        "5gMod3": None,
        "5gMod4": None,
        "5gMod30": None,
        "5gConfusion": None,
    },
}

ENB_MOD30_PENALTY = -0.03
ENB_MOD6_PENALTY = -0.02
ENB_MOD3_PENALTY = -0.01
GNB_MOD30_PENALTY = -0.03
GNB_MOD4_PENALTY = -0.02
GNB_MOD3_PENALTY = -0.01
CONFUSION_PENALTY = -0.02

RANGE_ALPHA = np.arange(0.004, 0.005, 0.0001)  # min, max, step
print(f"RANGE_ALPHA: {RANGE_ALPHA}")
RANGE_EPSILON = np.arange(0.7, 0.8, 0.01)  # min, max, step
print(f"RANGE_EPSILON: {RANGE_EPSILON}")
RANGE_GAMMA = np.arange(0.9, 0.95, 0.005)  # min, max, step
print(f"RANGE_GAMMA: {RANGE_GAMMA}")

population_inner_list = []
population_list = []

for alpha in RANGE_ALPHA:
    for epsilon in RANGE_EPSILON:
        for gamma in RANGE_GAMMA:
            # print(f"Alpha: {iAlpha}, Epsilon: {iEpsilon}, Gamma: {iGamma}")
            population_inner_list = [alpha, epsilon, gamma]
            population_list.append(population_inner_list)

if len(population_list) != len(RANGE_ALPHA) * len(RANGE_EPSILON) * len(RANGE_GAMMA):
    print("Population List Error!")

POPULATION_SIZE = len(population_list)  # 种群大小
