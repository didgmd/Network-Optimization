# -*- coding: utf-8 -*-
import numpy as np

# 短睡眠状态(S2)的Q表尺度参数
t_N = 10  # 单位: s 短循环定时器
t_DS = 2  # 单位: s 短周期
N_DS = int(t_N / t_DS)  # N_DS必须为整型
N_STATES = N_DS

# 状态转移概率矩阵
t_I = 2
t_DL = 10  # 长睡眠时长
trc = 0.005  # RRC连接释放时长
tec = 0.01  # 空闲态连接建立时长
tbs = 1.44  # 波束搜索时长
tpc = 0.128
t = 0.1  # 激活时间
LAMBDA_ipc = 1 / 30  # 分组到达时间间隙
LAMBDA_is = 1 / 2000  # 会话到达时间间隙
LAMBDA_ip = 10  # 数据包到达时间间隙
LAMBDA = 0.01  # 寻呼时间间隙
Wpc = 5  # 会话数目
Ppc = 1 - 1 / Wpc
Ps = 1 / Wpc
Wp = 25  # 分组数目
P11 = Ppc * (1 - np.exp(-LAMBDA_ipc * t_I)) + Ps * (1 - np.exp(-LAMBDA_is * t_I))
P12 = Ppc * np.exp(-LAMBDA_ipc * t_I) + Ps * np.exp(-LAMBDA_is * t_I)
P23 = Ppc * (1 - np.exp(-LAMBDA_ipc * t_N)) + Ps * (1 - np.exp(-LAMBDA_is * t_N))
P24 = Ppc * np.exp(-LAMBDA_ipc * t_N) + Ps * np.exp(-LAMBDA_is * t_N)
P31 = Ppc * (1 - np.exp(-LAMBDA_ipc * tbs)) + Ps * (1 - np.exp(-LAMBDA_is * tbs))
P35 = Ppc * np.exp(-LAMBDA_ipc * tbs) + Ps * np.exp(-LAMBDA_is * tbs)
P43 = 1
P56 = 1
P67 = 1
P76 = 1 - np.exp(-LAMBDA * tpc)
P78 = np.exp(-LAMBDA * tpc)
P81 = 1

# 创建列表保存从S1到S8经历过的所有状态
state_history = []

# 创建列表保存停留于S2期间经历的所有状态和动作
state_action_in_s2 = []
# q_predict_in_s2 = []
# q_target_in_s2 = []
loss_in_s2 = []
list_epoch_loss = []

# 短期时延
short_term_delay = []

# 长期时延
long_term_delay = []

# 长期S2计时器
long_term_dsCounter = []

# 长期功耗节省因子
long_term_pwrConsumCoeff = []

# 长期列表保存停留于S2期间的所有状态和动作
long_term_state_action_in_s2 = []
long_term_loss_in_s2 = []
long_term_list_epoch_loss = []

# 用于计算奖励的几个变量
param_for_reward = {
    "N_e2": 0,
    "pi2": 0,
    "pi4": 0,
    "pi6": 0,
    "e4": 0,
    "e6": 0,
}


def print_parameters():
    print(f"################# Parameters begin #################")
    print(f"t_N: {t_N}")
    print(f"t_DS: {t_DS}")
    print(f"N_DS: {N_DS}")
    print(f"N_STATES: {N_STATES}")
    print(f"t_I: {t_I}")
    print(f"t_DL: {t_DL}")
    print(f"trc: {trc}")
    print(f"tec: {tec}")
    print(f"tbs: {tbs}")
    print(f"tpc: {tpc}")
    print(f"t: {t}")
    print(f"LAMBDA_ipc: {LAMBDA_ipc}")
    print(f"LAMBDA_is: {LAMBDA_is}")
    print(f"LAMBDA_ip: {LAMBDA_ip}")
    print(f"LAMBDA: {LAMBDA}")
    print(f"Wpc: {Wpc}")
    print(f"Ppc: {Ppc}")
    print(f"Ps: {Ps}")
    print(f"Wp: {Wp}")
    print(f"P11: {P11}")
    print(f"P12: {P12}")
    print(f"P23: {P23}")
    print(f"P24: {P24}")
    print(f"P31: {P31}")
    print(f"P35: {P35}")
    print(f"P43: {P43}")
    print(f"P56: {P56}")
    print(f"P67: {P67}")
    print(f"P76: {P76}")
    print(f"P78: {P78}")
    print(f"P81: {P81}")
    print(f"################# Parameters end #################")
