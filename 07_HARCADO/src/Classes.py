# -*- coding: utf-8 -*-
import torch
import numpy as np
from Parameters import (
    BS_TX_POWER,
    BS_FREQUENCY,
    SHADOW_SIGMA_DB,
    INIT_TTT,
    INIT_HOM,
    AREA_SCALE_X,
    AREA_SCALE_Y,
)
from Formular import path_loss_calculation, sinr_calculation


class DQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 增加一层隐藏层提升网络容量
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = torch.nn.functional.relu(x)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_head = torch.nn.Linear(hidden_dim, 1)
        self.advantage_head = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = torch.nn.Linear(hidden_dim, output_dim)
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class User:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.curr_x = None
        self.curr_y = None
        self.next_x = None
        self.next_y = None

        self.source_bs_id = None
        self.source_bs_distance = None
        self.source_bs_pl = None
        self.source_bs_rsrp = None
        self.source_bs_sinr = None

        self.target_bs_id = None
        self.target_bs_distance = None
        self.target_bs_pl = None
        self.target_bs_rsrp = None
        self.target_bs_sinr = None

        self.ttt_determined = INIT_TTT
        self.ttt_backup = INIT_TTT
        self.ttt_countdown = INIT_TTT
        self.hom_determined = INIT_HOM
        self.dl_source_bs_rsrp_list = []  # 初始化当前基站RSRP列表, for DL
        self.avg_rsrp_in_dbm = 0  # 初始化平均RSRP, for DL
        self.ttt_before = None
        self.hom_before = None
        # self.handover_result = None  # True=成功，False=失败，None=未定义

    def get_curr_location(self, curr_step):
        self.curr_x = self.trajectory[curr_step][2]
        self.curr_y = self.trajectory[curr_step][3]

    def get_next_location(self, curr_step):
        self.next_x = self.trajectory[curr_step + 1][2]
        self.next_y = self.trajectory[curr_step + 1][3]

    def set_source_bs(self, source_bs, bs_list):
        self.source_bs_id = source_bs.bs_id
        self.source_bs_distance = np.sqrt(
            (self.curr_x - source_bs.bs_x) ** 2 + (self.curr_y - source_bs.bs_y) ** 2
        )
        self.source_bs_pl = path_loss_calculation(self.source_bs_distance, BS_FREQUENCY)
        self.source_bs_rsrp = (
            BS_TX_POWER
            - self.source_bs_pl - np.random.normal(loc=0, scale=SHADOW_SIGMA_DB)
        )

        # 计算 SINR
        self.source_bs_sinr = sinr_calculation(
            self.source_bs_rsrp,
            self.source_bs_id,
            self,
            bs_list,
            BS_FREQUENCY,
            BS_TX_POWER,
            SHADOW_SIGMA_DB,
        )

    def set_target_bs(self, target_bs, bs_list):
        self.target_bs_id = target_bs.bs_id
        self.target_bs_distance = np.sqrt(
            (self.curr_x - target_bs.bs_x) ** 2 + (self.curr_y - target_bs.bs_y) ** 2
        )
        self.target_bs_pl = path_loss_calculation(self.target_bs_distance, BS_FREQUENCY)
        self.target_bs_rsrp = (
            BS_TX_POWER
            - self.target_bs_pl
            - np.random.normal(loc=0, scale=SHADOW_SIGMA_DB)
        )

        # 计算 SINR
        self.target_bs_sinr = sinr_calculation(
            self.target_bs_rsrp,
            self.target_bs_id,
            self,
            bs_list,
            BS_FREQUENCY,
            BS_TX_POWER,
            SHADOW_SIGMA_DB,
        )

    def set_next_source_bs(self, bs_id, bs_list):
        for bs in bs_list:
            if bs.bs_id != self.source_bs_id:
                continue

            self.source_bs_distance = np.sqrt(
                (self.next_x - bs.bs_x) ** 2 + (self.next_y - bs.bs_y) ** 2
            )

            self.source_bs_pl = path_loss_calculation(
                self.source_bs_distance, BS_FREQUENCY
            )
            self.source_bs_rsrp = (
                BS_TX_POWER
                - self.source_bs_pl
                - np.random.normal(loc=0, scale=SHADOW_SIGMA_DB)
            )

            # 计算 SINR
            self.source_bs_sinr = sinr_calculation(
                self.source_bs_rsrp,
                self.source_bs_id,
                self,
                bs_list,
                BS_FREQUENCY,
                BS_TX_POWER,
                SHADOW_SIGMA_DB,
            )
            break

    def set_next_target_bs(self, bs_list):
        # 找出用户当前位置距离最近的非服务基站,作为目标基站
        min_distance = AREA_SCALE_X * AREA_SCALE_Y
        min_distance_bs_id = None
        for bs in bs_list:
            if bs.bs_id == self.source_bs_id:
                continue
            distance = np.sqrt(
                (self.next_x - bs.bs_x) ** 2 + (self.next_y - bs.bs_y) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                min_distance_bs_id = bs.bs_id

        self.target_bs_id = min_distance_bs_id
        self.target_bs_distance = min_distance

        self.target_bs_pl = path_loss_calculation(self.target_bs_distance, BS_FREQUENCY)
        self.target_bs_rsrp = (
            BS_TX_POWER
            - self.target_bs_pl
            - np.random.normal(loc=0, scale=SHADOW_SIGMA_DB)
        )

        # 计算 SINR
        self.target_bs_sinr = sinr_calculation(
            self.target_bs_rsrp,
            self.target_bs_id,
            self,
            bs_list,
            BS_FREQUENCY,
            BS_TX_POWER,
            SHADOW_SIGMA_DB,
        )

    def next_step_calculation_no_handover(self, bs_list):
        # SourceBsId不变
        self.set_next_source_bs(self.source_bs_id, bs_list)
        self.set_next_target_bs(bs_list)

    def next_step_calculation_with_handover(self, bs_list):
        self.source_bs_id = self.target_bs_id
        self.set_next_source_bs(self.source_bs_id, bs_list)
        self.set_next_target_bs(bs_list)

        # # 添加切换成功判定逻辑
        # # 联合规则：SINR 和 RSRP 同时满足阈值视为切换成功
        # sinr_success_threshold = -5  # db
        # rsrp_success_threshold = -100  # dbm
        #
        # if (
        #         self.source_bs_sinr >= sinr_success_threshold
        #         and self.source_bs_rsrp >= rsrp_success_threshold
        # ):
        #     self.handover_result = True
        # else:
        #     self.handover_result = False


class BS:
    def __init__(self, bs_id, bs_x, bs_y):
        self.bs_id = bs_id
        self.bs_x = bs_x
        self.bs_y = bs_y
