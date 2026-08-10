import numpy as np
import torch
import torch.nn as nn
from DebugPrint import *
from random import sample
from Parameters import *


# 定义Q网络
class QNetOptObject(nn.Module):
    def __init__(self, opt_object, n_states, n_hidden, n_actions):
        super(QNetOptObject, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.opt_object = opt_object
        self.currState = []
        self.nextState = []

    def __repr__(self):
        return f"QNet: opt_object = {self.opt_object}"

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_tensor = self.fc2(x)
        return actions_tensor


def reallocate_pci(obj_pci, node_list, action_index, pci_pool):
    # 在重新分配PCI之前，将该节点从原PCI的节点列表中删除，如果该原PCI的节点列表为空，则将其置为未被使用
    for obj_pci_temp in pci_pool:
        if node_list[action_index].posX in obj_pci_temp.nodeXList:
            obj_pci_temp.nodeXList.remove(node_list[action_index].posX)
        if node_list[action_index].posY in obj_pci_temp.nodeYList:
            obj_pci_temp.nodeYList.remove(node_list[action_index].posY)
        if node_list[action_index].radius in obj_pci_temp.radiusList:
            obj_pci_temp.radiusList.remove(node_list[action_index].radius)
        # if obj_pci_temp.pci == node_list[action_index].pci:
        #     obj_pci_temp.nodeXList.remove(node_list[action_index].posX)
        #     obj_pci_temp.nodeYList.remove(node_list[action_index].posY)
        #     obj_pci_temp.radiusList.remove(node_list[action_index].radius)
            if (
                len(obj_pci_temp.nodeXList) == 0
                and len(obj_pci_temp.nodeYList) == 0
                and len(obj_pci_temp.radiusList) == 0
            ):
                obj_pci_temp.isUsed = False
                debug(f"PCI {obj_pci_temp.pci} is unused now")
            break

    node_list[action_index].pci = obj_pci.pci
    obj_pci.isUsed = True
    obj_pci.nodeXList.append(node_list[action_index].posX)
    obj_pci.nodeYList.append(node_list[action_index].posY)
    obj_pci.radiusList.append(node_list[action_index].radius)
    debug(
        f"PCI {obj_pci.pci} is assigned to "
        f"Node ({node_list[action_index].posX}, {node_list[action_index].posY})"
    )


class QNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.device = None
        self.nStates = n_states
        self.nHidden = n_hidden
        self.nActions = n_actions
        self.currState = []
        self.nextState = []
        self.reward = 0.0

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_tensor = self.fc2(x)
        return actions_tensor

    def choose_action(self, epsilon, num_nodes_to_change_pci):
        # 初始化动作索引列表
        action_index_list = []
        if np.random.uniform() > epsilon:
            while len(action_index_list) < num_nodes_to_change_pci:
                action_index = np.random.choice(range(self.nActions))
                if action_index not in action_index_list:
                    action_index_list.append(action_index)
        else:
            curr_state_tensor = torch.tensor(self.currState).float().to(self.device)
            top_k_values, top_k_indices = torch.topk(
                self(curr_state_tensor), num_nodes_to_change_pci
            )
            debug(f"topKValues: {top_k_values}, topKIndices: {top_k_indices}")
            action_index_list = top_k_indices.tolist()

        return action_index_list

    def ac_reward_calculator(
        self,
        experience_pool,
        curr_num_enb_mod_30,
        curr_num_enb_mod_6,
        curr_num_enb_mod_3,
        curr_num_enb_confusion,
        curr_num_gnb_mod_30,
        curr_num_gnb_mod_4,
        curr_num_gnb_mod_3,
        curr_num_gnb_confusion,
        action_index_list,
    ):
        # 初始化AC奖励
        ac_reward = 0.0

        # 计算AC奖励
        for experience in experience_pool:
            if (
                experience["curr_state"] == self.currState
                and experience["curr_issue"]["4gMod30"] == curr_num_enb_mod_30
                and experience["curr_issue"]["4gMod6"] == curr_num_enb_mod_6
                and experience["curr_issue"]["4gMod3"] == curr_num_enb_mod_3
                and experience["curr_issue"]["4gConfusion"] == curr_num_enb_confusion
                and experience["curr_issue"]["5gMod30"] == curr_num_gnb_mod_30
                and experience["curr_issue"]["5gMod4"] == curr_num_gnb_mod_4
                and experience["curr_issue"]["5gMod3"] == curr_num_gnb_mod_3
                and experience["curr_issue"]["5gConfusion"] == curr_num_gnb_confusion
            ):
                # ToDo @LJN: AC奖励值或许需要调整
                if experience["reward"] > 0:
                    if experience["action"] == action_index_list:
                        ac_reward += 0.01
                    else:
                        ac_reward -= 0.01
                elif experience["reward"] < 0:
                    if experience["action"] == action_index_list:
                        ac_reward -= 0.01
                    else:
                        ac_reward += 0.01
                else:
                    ac_reward = 0.0

        return ac_reward

    def change_state(
        self, node_list, action_index_list, pci_pool_enb, pci_pool_gnb, r_sum_multiplier
    ):
        # debug(f"Inside change_state: action_index_list = {action_index_list}")
        self.nextState = self.currState.copy()
        for action_index in action_index_list:
            # debug(f"Inside change_state: action_index = {action_index}")
            while True:
                if node_list[action_index].nodeType == "enb":
                    obj_pci = sample(pci_pool_enb, 1)[0]
                    debug(f"PCI {obj_pci.pci} is chosen for eNB")

                    # 如果PCI未被使用，则直接分配
                    if not obj_pci.isUsed:
                        reallocate_pci(obj_pci, node_list, action_index, pci_pool_enb)
                        break

                    # 如果随机抽取的PCI与当前待重新分配PCI的节点相同，则重新抽取
                    elif obj_pci.pci == node_list[action_index].pci:
                        debug(
                            f"PCI {obj_pci.pci} is the same as Node ({node_list[action_index].posX}, "
                            f"{node_list[action_index].posY})'s PCI"
                        )
                        continue

                    # 如果PCI已被使用，则遍历使用该PCI的所有节点坐标，如果所有节点坐标与待重新分配PCI的节点坐标的距离均大于PCI重用距离阈值，则分配该PCI
                    else:
                        # 首先设定标志位为真，如果有一个或以上节点坐标与待重新分配PCI的节点坐标的距离小于等于PCI重用距离阈值，则将标志位设为假
                        pci_reuse_available = True
                        for nodeX, nodeY, radius in zip(
                            obj_pci.nodeXList, obj_pci.nodeYList, obj_pci.radiusList
                        ):
                            if (
                                np.sqrt(
                                    (nodeX - node_list[action_index].posX) ** 2
                                    + (nodeY - node_list[action_index].posY) ** 2
                                )
                                > (radius + node_list[action_index].radius)
                                * r_sum_multiplier
                            ):
                                debug(
                                    f"Node ({nodeX}, {nodeY}) is far enough from Node "
                                    f"({node_list[action_index].posX}, {node_list[action_index].posY})"
                                )
                                continue
                            else:
                                debug(
                                    f"Node ({nodeX}, {nodeY}) is not far enough from Node "
                                    f"({node_list[action_index].posX}, {node_list[action_index].posY})"
                                )
                                pci_reuse_available = False
                                break

                        # 如果标志位为真，则分配该PCI
                        if pci_reuse_available:
                            reallocate_pci(
                                obj_pci, node_list, action_index, pci_pool_enb
                            )
                            break

                elif node_list[action_index].nodeType == "gnb":
                    obj_pci = sample(pci_pool_gnb, 1)[0]
                    debug(f"PCI {obj_pci.pci} is chosen for gNB")

                    # 如果PCI未被使用，则直接分配
                    if not obj_pci.isUsed:
                        reallocate_pci(obj_pci, node_list, action_index, pci_pool_gnb)
                        break

                    # 如果随机抽取的PCI与当前待重新分配PCI的节点相同，则重新抽取
                    elif obj_pci.pci == node_list[action_index].pci:
                        debug(
                            f"PCI {obj_pci.pci} is the same as Node ({node_list[action_index].posX}, "
                            f"{node_list[action_index].posY})'s PCI"
                        )
                        continue

                    # 如果PCI已被使用，则遍历使用该PCI的所有节点坐标，如果所有节点坐标与待重新分配PCI的节点坐标的距离均大于PCI重用距离阈值，则分配该PCI
                    else:
                        # 首先设定标志位为真，如果有一个或以上节点坐标与待重新分配PCI的节点坐标的距离小于等于PCI重用距离阈值，则将标志位设为假
                        pci_reuse_available = True
                        for nodeX, nodeY, radius in zip(
                            obj_pci.nodeXList, obj_pci.nodeYList, obj_pci.radiusList
                        ):
                            if (
                                np.sqrt(
                                    (nodeX - node_list[action_index].posX) ** 2
                                    + (nodeY - node_list[action_index].posY) ** 2
                                )
                                > (radius + node_list[action_index].radius)
                                * r_sum_multiplier
                            ):
                                debug(
                                    f"Node ({nodeX}, {nodeY}) is far enough from Node "
                                    f"({node_list[action_index].posX}, {node_list[action_index].posY})"
                                )
                                continue
                            else:
                                debug(
                                    f"Node ({nodeX}, {nodeY}) is not far enough from Node "
                                    f"({node_list[action_index].posX}, {node_list[action_index].posY})"
                                )
                                pci_reuse_available = False
                                break

                        # 如果标志位为真，则分配该PCI
                        if pci_reuse_available:
                            reallocate_pci(
                                obj_pci, node_list, action_index, pci_pool_gnb
                            )
                            break

            self.nextState[action_index] = obj_pci.pci

        return self.nextState

    def reward_calculation(
        self,
        curr_num_enb_mod_30,
        curr_num_enb_mod_6,
        curr_num_enb_mod_3,
        curr_num_enb_confusion,
        curr_num_gnb_mod_30,
        curr_num_gnb_mod_4,
        curr_num_gnb_mod_3,
        curr_num_gnb_confusion,
        next_num_enb_mod_30,
        next_num_enb_mod_6,
        next_num_enb_mod_3,
        next_num_enb_confusion,
        next_num_gnb_mod_30,
        next_num_gnb_mod_4,
        next_num_gnb_mod_3,
        next_num_gnb_confusion,
    ):
        # 初始化奖励
        reward = 0.0

        reward += (next_num_enb_mod_30 - curr_num_enb_mod_30) * ENB_MOD30_PENALTY
        reward += (next_num_enb_mod_6 - curr_num_enb_mod_6) * ENB_MOD6_PENALTY
        reward += (next_num_enb_mod_3 - curr_num_enb_mod_3) * ENB_MOD3_PENALTY
        reward += (next_num_enb_confusion - curr_num_enb_confusion) * CONFUSION_PENALTY
        reward += (next_num_gnb_mod_30 - curr_num_gnb_mod_30) * GNB_MOD30_PENALTY
        reward += (next_num_gnb_mod_4 - curr_num_gnb_mod_4) * GNB_MOD4_PENALTY
        reward += (next_num_gnb_mod_3 - curr_num_gnb_mod_3) * GNB_MOD3_PENALTY
        reward += (next_num_gnb_confusion - curr_num_gnb_confusion) * CONFUSION_PENALTY

        self.reward = reward

        return reward
