import numpy as np
import torch
from Normalization import *


# 选择是否执行NN2和NN3决定的动作
def nn1_choose_action(nn1_curr_state, q_net, nn1_epsilon, nn1_actions, device):
    normalized_nn1_curr_state = nn1_state_normalization(nn1_curr_state)
    # 选择动作
    if np.random.uniform() > nn1_epsilon:
        action = np.random.choice(nn1_actions)
    else:
        actions_tensor = q_net(
            torch.tensor(normalized_nn1_curr_state).float().to(device)
        )
        action = nn1_actions[actions_tensor.argmax()]

    return action


# 由选择S5改为选择A2阈值和A4阈值
def nn2_choose_action(
    nn2_curr_state, q_net, nn2_epsilon, nn2_actions_a2, nn2_actions_a4, device
):
    # 归一化
    normalized_nn2_curr_state = nn2_state_normalization(nn2_curr_state)

    # 选择动作
    if np.random.uniform() > nn2_epsilon:
        action_a2 = np.random.choice(nn2_actions_a2)
        action_a4 = np.random.choice(nn2_actions_a4)
    else:
        actions_a2_tensor, actions_a4_tensor = q_net(
            torch.tensor(normalized_nn2_curr_state).float().to(device)
        )
        action_a2 = nn2_actions_a2[actions_a2_tensor.argmax()]
        action_a4 = nn2_actions_a4[actions_a4_tensor.argmax()]

    return action_a2, action_a4


# 选择S6
def nn3_choose_action(nn3_curr_state, q_net, nn3_epsilon, nn3_actions, device):
    normalized_nn3_curr_state = nn3_state_normalization(nn3_curr_state)
    # 选择动作
    if np.random.uniform() > nn3_epsilon:
        action = np.random.choice(nn3_actions)
    else:
        actions_tensor = q_net(
            torch.tensor(normalized_nn3_curr_state).float().to(device)
        )
        action = nn3_actions[actions_tensor.argmax()]

    return action
