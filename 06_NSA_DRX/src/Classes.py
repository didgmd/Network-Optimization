from enum import Enum


class DrxState(Enum):
    S1 = 1  # 连接态的活跃状态
    S2 = 2  # 连接态的短睡眠状态
    S3 = 3  # 连接态的波束搜索状态
    S4 = 4  # 连接态的长睡眠状态
    S5 = 5  # 连接释放状态
    S6 = 6  # 空闲态波束搜索状态
    S7 = 7  # 空闲态睡眠状态
    S8 = 8  # 连接建立状态
