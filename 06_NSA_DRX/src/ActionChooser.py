import numpy as np


def choose_action(curr_state, q_table, epsilon, action_space):
    state_action_values = q_table[curr_state, :]
    if np.random.uniform() > epsilon or state_action_values.all() == 0:
        action = np.random.choice(action_space)  # 1-epsilon的概率随机选择动作
    else:
        action = action_space[state_action_values.argmax()]

    return action
