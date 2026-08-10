import numpy as np
import torch

# 自定义模块
from DebugPrint import *


def choose_action(curr_state, q_net, epsilon, list_size, device, n_nodes_to_change_pci):
    # 初始化动作索引列表
    actionIndexList = []
    if np.random.uniform() > epsilon:
        while len(actionIndexList) < n_nodes_to_change_pci:
            actionIndex = np.random.choice(range(list_size))
            if actionIndex not in actionIndexList:
                actionIndexList.append(actionIndex)

    else:
        currStateTensor = torch.tensor(curr_state).float().to(device)
        # actionIndex = q_net(currStateTensor).argmax()
        topKValues, topKIndices = torch.topk(
            q_net(currStateTensor), n_nodes_to_change_pci
        )
        debug(f"topKValues: {topKValues}, topKIndices: {topKIndices}")
        actionIndexList = topKIndices.tolist()

    return actionIndexList
