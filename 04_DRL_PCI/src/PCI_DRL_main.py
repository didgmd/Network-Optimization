# -*- coding:utf8 -*-
import torch
import DataSave
import torch.nn as nn
import random

from Parameters import *
from NetworkTopology import *
from NeighborList import *
from PciAllocation import *
from QNet import QNet
from CheckForCollision import *
from CheckForConfusion import *
from Normalization import *
from FigurePlotter import *

# 定义全局冲突和混淆点数目列表
globalCollisionNumList = []
globalConfusionNumList = []
globalenbMod30CollisionList = []
globalenbMod6CollisionList = []
globalenbMod3CollisionList = []
globalgnbMod30CollisionList = []
globalgnbMod4CollisionList = []
globalgnbMod3CollisionList = []

#定义子循环和主循环时间列表
globalSubLoopTimeList = []
globalMainLoopTimeList = []

def check_for_pci_collision_and_confusion(enb_list, gnb_list, nodesToChangePCIIndexList):
    # 检查eNB冲突数量
    num_enb_mod_30, num_enb_mod_6, num_enb_mod_3, nodesToChangePCIIndexList = check_for_enb_pci_collision(enb_list, nodesToChangePCIIndexList)
    debug(f"Number of eNB PCI mod 30 is {num_enb_mod_30}, mod 6 is {num_enb_mod_6}, mod 3 is {num_enb_mod_3}")

    # 检查eNB混淆数量
    num_enb_confusion, nodesToChangePCIIndexList = check_for_enb_pci_confusion(enb_list, nodesToChangePCIIndexList)
    debug(f"Number of eNB PCI confusion is {num_enb_confusion}")

    # 检查gNB冲突数量
    num_gnb_mod_30, num_gnb_mod_4, num_gnb_mod_3, nodesToChangePCIIndexList = check_for_gnb_pci_collision(gnb_list, nodesToChangePCIIndexList)
    debug(f"Number of gNB PCI mod 30 is {num_gnb_mod_30}, mod 4 is {num_gnb_mod_4}, mod 3 is {num_gnb_mod_3}")

    # 检查gNB混淆数量
    num_gnb_confusion, nodesToChangePCIIndexList = check_for_gnb_pci_confusion(gnb_list, nodesToChangePCIIndexList)
    debug(f"Number of gNB PCI confusion is {num_gnb_confusion}")

    return (
        num_enb_mod_30,
        num_enb_mod_6,
        num_enb_mod_3,
        num_enb_confusion,
        num_gnb_mod_30,
        num_gnb_mod_4,
        num_gnb_mod_3,
        num_gnb_confusion,
        nodesToChangePCIIndexList
    )


def count_enb_to_change_pci(enb_list):
    collision_list = []
    for enb in enb_list:
        for collisionNode in enb.collisionList:
            if collisionNode not in collision_list:
                collision_list.append(collisionNode)
        for confusionNode in enb.confusionList:
            if confusionNode not in collision_list:
                collision_list.append(confusionNode)
    return len(collision_list)


def count_gnb_to_change_pci(gnb_list):
    collision_list = []
    for gnb in gnb_list:
        for collisionNode in gnb.collisionList:
            if collisionNode not in collision_list:
                collision_list.append(collisionNode)
        for confusionNode in gnb.confusionList:     
            if confusionNode not in collision_list:
                collision_list.append(confusionNode)
    return len(collision_list)


def num_decrease_calculation(curr_value, next_value):
    debug(f"Current value is {curr_value}, next value is {next_value}")
    if curr_value == 0 and next_value == 0:
        return 1
    elif curr_value != 0 and next_value < curr_value:
        return 1
    else:
        return 0

def rl():
    q_net = QNet(N_INPUT_FEATURE, N_HIDDEN_NEURON, N_OUTPUT_FEATURE).to(device)
    debug(f"QNet: {q_net} is created.")
    q_net.device = device
    optimizer = torch.optim.Adam(q_net.parameters(), lr=INIT_ALPHA)
    debug(f"Optimizer: {optimizer} is created.")

    # debug(f"optimizer.param_groups: {optimizer.param_groups}")

    # 初始化经验池（8元组：当前状态、各种冲突数、各种混淆数、动作、奖励、下个状态、各种冲突数、各种混淆数）
    experience_pool = []

    #初始化平均损失函数
    prevLoss = 0
    avgLoss = 0.0

    if FLAG_GENERATION_ALGORITHM:
        # 初始化遗传算法适应度
        prev_fitness = 0.0

    epsilon = INIT_EPSILON  # 初始化贪婪度
    gamma = INIT_GAMMA  # 初始化折扣因子

    # 创建列表存储每Epoch最终的损失函数值
    lossList = []

    #创建列表存储每Epoch结束后的平均损失函数值
    avgLossList = []

    # 创建列表保存需更改PCI的节点
    nodesToChangePCIIndexList = []

    # 初始化主循环计数器
    epoch = 1

    while True:
        debug_print(f"################ Epoch {epoch} Start ################")
        #主循环开始时间
        mainLoopStartTime = time.time()

        #初始化PCI池
        pci_pool_enb, pci_pool_gnb = initialize_pci_pool(
           enb_pci_min, enb_pci_max, gnb_pci_min, gnb_pci_max
        )
        #pci_pool_enb, pci_pool_gnb = initialize_pci_pool(12, 19, 31, 40)

        #初始化节点PCI为-1
        for objNode in nodeList:
            objNode.pci = -1

        # 基于哈希表初始PCI分配
        if HASH_TABLE_FLAG:
            hashmap_based_pci_initial_allocation(
                pci_pool_enb,
                pci_pool_gnb,
                enbList,
                gnbList,
                enb_pci_min,
                enb_pci_max,
                gnb_pci_min,
                gnb_pci_max,
                iteration=epoch,
            )
        else:
            random_pci_initial_allocation(enbList, gnbList)

        if epoch == 1:
            # 绘制eNB分布图
            plot_enb(enbList, X_MAX, Y_MAX)

            # 绘制gNB分布图
            plot_gnb(gnbList, X_MAX, Y_MAX)

        # 初始化当前状态，即输入特征列表
        curr_state = []
        for objNode in nodeList:
            curr_state.append(objNode.pci)
        debug(f"Current state: {curr_state}")
        q_net.currState = curr_state

        # 初始化子循环计数器
        episode = 1

        # 子循环
        while True:
            debug_print(f"################ Episode {episode} Start ################")
            #debug(f"Current state: {curr_state}")

            #子循环开始时间
            subLoopStartTime = time.time()

            nodesToChangePCIIndexList.clear()  # 初始化需更改PCI的节点列表
            globalCollisionNumList.clear()  # 初始化全局冲突数目列表
            globalConfusionNumList.clear()  # 初始化全局混淆数目列表
            globalenbMod30CollisionList.clear()  # 初始化全局eNB PCI mod 30冲突数目列表
            globalenbMod6CollisionList.clear()  # 初始化全局eNB PCI mod 6冲突数目列表
            globalenbMod3CollisionList.clear()  # 初始化全局eNB PCI mod 3冲突数目列表
            globalgnbMod30CollisionList.clear()  # 初始化全局gNB PCI mod 30冲突数目列表
            globalgnbMod4CollisionList.clear()  # 初始化全局gNB PCI mod 4冲突数目列表
            globalgnbMod3CollisionList.clear()  # 初始化全局gNB PCI mod 3冲突数目列表

            # 计算当前的冲突和混淆数
            (
                curr_num_enb_mod_30,
                curr_num_enb_mod_6,
                curr_num_enb_mod_3,
                curr_num_enb_confusion,
                curr_num_gnb_mod_30,
                curr_num_gnb_mod_4,
                curr_num_gnb_mod_3,
                curr_num_gnb_confusion,
                nodesToChangePCIIndexList
            ) = check_for_pci_collision_and_confusion(enbList, gnbList, nodesToChangePCIIndexList)

            #将冲突混淆数目添加进列表
            globalCollisionNumList.append(curr_num_enb_mod_30 + curr_num_enb_mod_6 + curr_num_enb_mod_3 + curr_num_gnb_mod_30 + curr_num_gnb_mod_4 + curr_num_gnb_mod_3)
            globalConfusionNumList.append(curr_num_enb_confusion + curr_num_gnb_confusion)
            globalenbMod30CollisionList.append(curr_num_enb_mod_30)
            globalenbMod6CollisionList.append(curr_num_enb_mod_6)
            globalenbMod3CollisionList.append(curr_num_enb_mod_3)
            globalgnbMod30CollisionList.append(curr_num_gnb_mod_30)
            globalgnbMod4CollisionList.append(curr_num_gnb_mod_4)
            globalgnbMod3CollisionList.append(curr_num_gnb_mod_3)

            debug(f"Current number of eNB PCI mod 30 is {curr_num_enb_mod_30}, mod 6 is {curr_num_enb_mod_6}, mod 3 is {curr_num_enb_mod_3}, confusion is {curr_num_enb_confusion}. "
                  f"Current number of gNB PCI mod 30 is {curr_num_gnb_mod_30}, mod 4 is {curr_num_gnb_mod_4}, mod 3 is {curr_num_gnb_mod_3}, confusion is {curr_num_gnb_confusion}")

            num_nodes_to_change_pci = len(nodesToChangePCIIndexList)
            debug_print(f"Number of nodes to change PCI is {num_nodes_to_change_pci}")

            # 如果需要改变PCI的节点数为0，则子循环结束
            if num_nodes_to_change_pci == 0:
                debug(f"All collisions and confusions are resolved!!!")
                break

            # 执行者 Actor 选择动作
            action_index_list = q_net.choose_action(epsilon, num_nodes_to_change_pci)
            debug(f"Action index list: {action_index_list}")

            # 当经验池不为空时，评价者 Critic 根据经验池计算AC奖励，随着经验池的增加，AC奖励逐渐收敛 Todo @LJN: AC奖励值或许需要调整
            if len(experience_pool) > 0:
                ac_reward = q_net.ac_reward_calculator(
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
                ) / len(experience_pool)
                debug(f"AC reward: {ac_reward}")

            # 执行动作并计算下个状态
            next_state = q_net.change_state(
                nodeList,
                action_index_list,
                pci_pool_enb,
                pci_pool_gnb,
                R_SUM_MULTIPLIER,
            )
            debug(f"Next state: {next_state}")

             #归一化
            currStateNorm = min_max_normalization(nodeList, curr_state)
            debug(f"currStateNorm is {currStateNorm}")

            nextStateNorm = min_max_normalization(nodeList, next_state)
            debug(f"nextStateNorm is {nextStateNorm}")


            nodesToChangePCIIndexList.clear()  # 清空需更改PCI的节点列表

            # 计算执行动作后的冲突和混淆数
            (
                next_num_enb_mod_30,
                next_num_enb_mod_6,
                next_num_enb_mod_3,
                next_num_enb_confusion,
                next_num_gnb_mod_30,
                next_num_gnb_mod_4,
                next_num_gnb_mod_3,
                next_num_gnb_confusion,
                nodesToChangePCIIndexList
            ) = check_for_pci_collision_and_confusion(enbList, gnbList, nodesToChangePCIIndexList)

            num_nodes_to_change_pci = len(nodesToChangePCIIndexList)

            # MOSA模拟退火算法开始
            if MOSA_ALGORITHM:
                # 初始温度即当前的冲突和混淆数量已经确定，动作也已执行完毕，现需要判断下个状态的冲突和混淆数量，进而决定是否采用该动作

                enb_mod_30_decrease = num_decrease_calculation(
                    curr_num_enb_mod_30, next_num_enb_mod_30
                )
                debug(f"enb_mod_30_decrease is {enb_mod_30_decrease}")
                enb_mod_6_decrease = num_decrease_calculation(
                    curr_num_enb_mod_6, next_num_enb_mod_6
                )
                debug(f"enb_mod_6_decrease is {enb_mod_6_decrease}")
                enb_mod_3_decrease = num_decrease_calculation(
                    curr_num_enb_mod_3, next_num_enb_mod_3
                )
                debug(f"enb_mod_3_decrease is {enb_mod_3_decrease}")
                enb_confusion_decrease = num_decrease_calculation(
                    curr_num_enb_confusion, next_num_enb_confusion
                )
                debug(f"enb_confusion_decrease is {enb_confusion_decrease}")
                gnb_mod_30_decrease = num_decrease_calculation(
                    curr_num_gnb_mod_30, next_num_gnb_mod_30
                )
                debug(f"gnb_mod_30_decrease is {gnb_mod_30_decrease}")
                gnb_mod_4_decrease = num_decrease_calculation(
                    curr_num_gnb_mod_4, next_num_gnb_mod_4
                )
                debug(f"gnb_mod_4_decrease is {gnb_mod_4_decrease}")
                gnb_mod_3_decrease = num_decrease_calculation(
                    curr_num_gnb_mod_3, next_num_gnb_mod_3
                )
                debug(f"gnb_mod_3_decrease is {gnb_mod_3_decrease}")
                gnb_confusion_decrease = num_decrease_calculation(
                    curr_num_gnb_confusion, next_num_gnb_confusion
                )
                debug(f"gnb_confusion_decrease is {gnb_confusion_decrease}")

                # 发生减少的条目个数，最多为8个
                num_decrease = (
                    enb_mod_30_decrease
                    + enb_mod_6_decrease
                    + enb_mod_3_decrease
                    + enb_confusion_decrease
                    + gnb_mod_30_decrease
                    + gnb_mod_4_decrease
                    + gnb_mod_3_decrease
                    + gnb_confusion_decrease
                )

                debug_print(f"Total number of decreased items is {num_decrease}")

                # 如果该动作执行后，所有冲突和混淆条目的数量均减少，则直接接受该动作，继续下一步
                if num_decrease == 8:
                    debug("All targets decreased, proceed to next step")

                # 如果该动作执行后，只有部分冲突和混淆条目的数量减少，则以一定概率接受该动作，当接受该动作时，继续下一步
                elif 0 < num_decrease < 8:
                    mosa_weights = [
                        0.2,
                        0.1,
                        0.05,
                        0.15,
                        0.2,
                        0.1,
                        0.05,
                        0.15,
                    ]  # ToDo @LJN 后续可考虑微调，保持权重和为1即可

                    probability_for_proceeding = (
                        enb_mod_30_decrease * mosa_weights[0]
                        + enb_mod_6_decrease * mosa_weights[1]
                        + enb_mod_3_decrease * mosa_weights[2]
                        + enb_confusion_decrease * mosa_weights[3]
                        + gnb_mod_30_decrease * mosa_weights[4]
                        + gnb_mod_4_decrease * mosa_weights[5]
                        + gnb_mod_3_decrease * mosa_weights[6]
                        + gnb_confusion_decrease * mosa_weights[7]
                    )
                    debug(f"Probability for proceeding is {probability_for_proceeding}")

                    if np.random.uniform() >= probability_for_proceeding:
                        debug("Partial targets decreased, proceed to next step")
                    else:
                        debug("Partial targets decreased, fallback to previous step")
                        continue

                # 如果该动作执行后，所有冲突和混淆条目的数量均未减少，则放弃该动作，回退到上一步
                else:
                    debug("No target decreased, fallback to previous step")
                    continue

            # MOSA模拟退火算法结束

            # 计算奖励
            reward = q_net.reward_calculation(
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
            )

            # 创建新的经验池元素，用来存储当前状态、各种冲突数、各种混淆数、动作、奖励、下个状态、各种冲突数、各种混淆数
            experience_pool_element = EXPERIENCE_POOL_ELEMENT.copy()
            flag_element_traversed = False
            for element in experience_pool:
                if element["curr_state"] == curr_state:
                    flag_element_traversed = True
                    if reward >= element["reward"]:
                        debug(f"Reward is {reward}, greater than {element['reward']}, update the experience pool.")
                        element["curr_state"] = curr_state.copy()
                        element["curr_issue"]["enb_mod_30"] = curr_num_enb_mod_30
                        element["curr_issue"]["enb_mod_6"] = curr_num_enb_mod_6
                        element["curr_issue"]["enb_mod_3"] = curr_num_enb_mod_3
                        element["curr_issue"]["enb_confusion"] = curr_num_enb_confusion
                        element["curr_issue"]["gnb_mod_30"] = curr_num_gnb_mod_30
                        element["curr_issue"]["gnb_mod_4"] = curr_num_gnb_mod_4
                        element["curr_issue"]["gnb_mod_3"] = curr_num_gnb_mod_3
                        element["curr_issue"]["gnb_confusion"] = curr_num_gnb_confusion
                        element["action"] = action_index_list.copy()
                        element["reward"] = reward
                        element["next_state"] = next_state.copy()
                        element["next_issue"]["enb_mod_30"] = next_num_enb_mod_30
                        element["next_issue"]["enb_mod_6"] = next_num_enb_mod_6
                        element["next_issue"]["enb_mod_3"] = next_num_enb_mod_3
                        element["next_issue"]["enb_confusion"] = next_num_enb_confusion
                        element["next_issue"]["gnb_mod_30"] = next_num_gnb_mod_30
                        element["next_issue"]["gnb_mod_4"] = next_num_gnb_mod_4
                        element["next_issue"]["gnb_mod_3"] = next_num_gnb_mod_3
                        element["next_issue"]["gnb_confusion"] = next_num_gnb_confusion
                        break
                    else:
                        debug(f"Reward is {reward}, less than {element['reward']}, keep the experience pool.")
                        break

            if not flag_element_traversed:
                debug(f"Reward is {reward}, no element traversed, add a new element to the experience pool.")
                experience_pool_element["curr_state"] = curr_state.copy()
                experience_pool_element["curr_issue"]["enb_mod_30"] = curr_num_enb_mod_30
                experience_pool_element["curr_issue"]["enb_mod_6"] = curr_num_enb_mod_6
                experience_pool_element["curr_issue"]["enb_mod_3"] = curr_num_enb_mod_3
                experience_pool_element["curr_issue"][
                    "enb_confusion"
                ] = curr_num_enb_confusion
                experience_pool_element["curr_issue"]["gnb_mod_30"] = curr_num_gnb_mod_30
                experience_pool_element["curr_issue"]["gnb_mod_4"] = curr_num_gnb_mod_4
                experience_pool_element["curr_issue"]["gnb_mod_3"] = curr_num_gnb_mod_3
                experience_pool_element["curr_issue"][
                    "gnb_confusion"
                ] = curr_num_gnb_confusion
                experience_pool_element["action"] = action_index_list.copy()
                experience_pool_element["reward"] = reward
                experience_pool_element["next_state"] = next_state.copy()
                experience_pool_element["next_issue"]["enb_mod_30"] = next_num_enb_mod_30
                experience_pool_element["next_issue"]["enb_mod_6"] = next_num_enb_mod_6
                experience_pool_element["next_issue"]["enb_mod_3"] = next_num_enb_mod_3
                experience_pool_element["next_issue"][
                    "enb_confusion"
                ] = next_num_enb_confusion
                experience_pool_element["next_issue"]["gnb_mod_30"] = next_num_gnb_mod_30
                experience_pool_element["next_issue"]["gnb_mod_4"] = next_num_gnb_mod_4
                experience_pool_element["next_issue"]["gnb_mod_3"] = next_num_gnb_mod_3
                experience_pool_element["next_issue"][
                    "gnb_confusion"
                ] = next_num_gnb_confusion
                experience_pool.append(experience_pool_element)
            # 以上完成经验池的更新

            # #归一化
            # currStateNorm = min_max_normalization(nodeList, curr_state)
            # debug(f"currStateNorm is {currStateNorm}")

            # nextStateNorm = min_max_normalization(nodeList, next_state)
            # debug(f"nextStateNorm is {nextStateNorm}")

            # 计算估计值 ToDo 1. 状态需进行归一化；2. EMA待添加
            q_predict = q_net(torch.tensor(currStateNorm).float().to(device))[
                action_index_list[0]
            ]
            debug(f"Estimated Q-value is {q_predict}")

            # 计算经引导的目标值
            output_q_list = q_net(torch.tensor(nextStateNorm).float().to(device))
            output_max_q = output_q_list.min()  # 初始化为最小值
            for guideIndex in range(len(action_index_list)):
                if output_q_list[action_index_list[guideIndex]] > output_max_q:
                    output_max_q = output_q_list[action_index_list[guideIndex]]

            q_target = reward + gamma * output_max_q
            debug(f"Guided target Q-value is {q_target}")

            # 计算损失函数
            flagHuberLoss = True
            if flagHuberLoss:
                objSmoothL1Loss = nn.SmoothL1Loss(reduction="mean")
                loss = objSmoothL1Loss(q_predict, q_target)
            else:
                loss = nn.functional.mse_loss(q_predict, q_target)
            debug(f"Loss value is: {loss}")

            # 计算平均损失函数
            avgLoss=(EMA_LOSS * loss.cpu().detach().numpy().item()
                     + (1 - EMA_LOSS) * prevLoss)
            prevLoss = avgLoss
            debug(f"avgLoss value is: {avgLoss}")

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()


            if FLAG_GENERATION_ALGORITHM:
                # 遗传算法
                curr_fitness = avgLoss - loss  # 适应度越大越好

                if curr_fitness > prev_fitness:
                    prev_fitness = curr_fitness
                    debug(f"Epoch {epoch} Episode {episode} GA fitness is {curr_fitness} -> Update prev_fitness.")
                else:
                    debug(f"Epoch {epoch} Episode {episode} GA fitness is {curr_fitness} -> Execute GA.")
                   
                    alpha = optimizer.param_groups[0]['lr']  # 获取当前学习率

                    if curr_fitness < 0:
                        # 只要loss小于avgLoss，则交叉：改变所有参数
                        while True:
                            parameter_set_selected = random.choice(population_list)
                            if parameter_set_selected[0] != alpha and parameter_set_selected[1] != epsilon and parameter_set_selected[2] != gamma:
                                debug(f"parameter_set_selected is {parameter_set_selected}")
                                break

                        optimizer.param_groups[0]['lr'] = parameter_set_selected[0]  # 更新学习率
                        epsilon = parameter_set_selected[1]  # 更新贪婪度
                        gamma = parameter_set_selected[2]  # 更新折扣因子

                    else:
                        # 其他所有不大于prev_fitness的情况，均变异：改变单个参数
                        while True:
                            parameter_set_selected = random.choice(population_list)
                            if parameter_set_selected[0] != alpha and parameter_set_selected[1] == epsilon and parameter_set_selected[2] == gamma:
                                debug(f"parameter_set_selected is {parameter_set_selected}")
                                optimizer.param_groups[0]['lr'] = parameter_set_selected[0]  # 更新学习率
                                break
                            elif parameter_set_selected[0] == alpha and parameter_set_selected[1] != epsilon and parameter_set_selected[2] == gamma:
                                debug(f"parameter_set_selected is {parameter_set_selected}")
                                epsilon = parameter_set_selected[1]  # 更新贪婪度
                                break
                            elif parameter_set_selected[0] == alpha and parameter_set_selected[1] == epsilon and parameter_set_selected[2] != gamma:
                                debug(f"parameter_set_selected is {parameter_set_selected}")
                                gamma = parameter_set_selected[2]  # 更新折扣因子
                                break
            
            # 确认子循环收敛条件
            if num_nodes_to_change_pci == 0:
                lossList.append(loss)
                avgLossList.append(avgLoss)
                debug_print(
                        f"Epoch {epoch} converged at episode {episode}, final reward is {reward}, "
                        f"average loss is {avgLoss}."
                    )
                
                #子循环结束时间
                subLoopEndTime = time.time()

                #计算子循环时间
                subLoopTime = subLoopEndTime - subLoopStartTime

                #将子循环时间添加进列表
                globalSubLoopTimeList.append(subLoopTime)
                
                #将列表保存至表格
                list_to_save = [str(subLoopTime)]
                list_name = "mainLoopTime"
                DataSave.save_data_to_excel(list_to_save, list_name)

                list_to_save=[str(item) for item in globalCollisionNumList]
                list_name="collisionNum"
                DataSave.save_data_to_excel(list_to_save,list_name)
                globalCollisionNumList.clear()

                list_to_save=[str(item) for item in globalConfusionNumList]
                list_name="confusionNum"
                DataSave.save_data_to_excel(list_to_save,list_name)
                globalConfusionNumList.clear()

                list_to_save = [str(item) for item in globalenbMod30CollisionList]
                list_name = "enbmod30Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalenbMod30CollisionList.clear()

                list_to_save = [str(item) for item in globalenbMod6CollisionList]
                list_name = "enbmod6Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalenbMod6CollisionList.clear()

                list_to_save = [str(item) for item in globalenbMod3CollisionList]
                list_name = "enbmod3Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalenbMod3CollisionList.clear()

                list_to_save = [str(item) for item in globalgnbMod30CollisionList]
                list_name = "gnbmod30Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalgnbMod30CollisionList.clear()

                list_to_save = [str(item) for item in globalgnbMod4CollisionList]
                list_name = "gnbmod4Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalgnbMod4CollisionList.clear()

                list_to_save = [str(item) for item in globalgnbMod3CollisionList]
                list_name = "gnbmod3Collision"
                DataSave.save_data_to_excel(list_to_save, list_name)
                globalgnbMod3CollisionList.clear()

                # #将损失函数列表保存至表格
                # list_to_save=[str(item) for item in lossList]
                # list_name="lossList"
                # DataSave.save_data_to_excel(list_to_save,list_name)

                # #将平均损失函数保存至表格
                # list_to_save=[str(item) for item in avgLossList]
                # list_name="avgLoss"
                # DataSave.save_data_to_excel(list_to_save,list_name)
                
                # 每轮优化结束后，清除冲突点和混淆点全局列表
                globalCollisionNumList.clear()
                globalConfusionNumList.clear()

                break

            # 状态切换
            curr_state = next_state.copy()

            # 更新子循环计数器
            episode += 1

        debug_print(f"lossList is {lossList}")

        # 确认主循环收敛条件
        if avgLoss <= 10e-5:
            debug_print(
                f"Epoch {epoch} converged, average loss is {avgLoss}."
            )
            #主循环结束时间
            mainLoopEndTime = time.time()

            #计算主循环时间
            mainLoopTime = mainLoopEndTime - mainLoopStartTime

            #将主循环时间添加进列表
            globalMainLoopTimeList.append(mainLoopTime)

            break

        # 更新主循环计数器
        epoch += 1

    #将损失函数保存至表格
    list_to_save=[str(item) for item in lossList]
    list_name="losslist"
    DataSave.save_data_to_excel(list_to_save,list_name)

    #将平均损失函数保存至表格
    list_to_save=[str(item) for item in avgLossList]
    list_name="avgLoss"
    DataSave.save_data_to_excel(list_to_save,list_name)

    #将主循环时间保存至表格
    list_to_save = [str(mainLoopTime)]
    list_name = "mainLoopTime"
    DataSave.save_data_to_excel(list_to_save, list_name)
    
    # 每轮优化结束后，清除冲突点和混淆点全局列表
    globalCollisionNumList.clear()
    globalConfusionNumList.clear()
    globalMainLoopTimeList.clear()

if __name__ == "__main__":
    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"device: {device}")

    # PCI 取值范围
    enb_pci_min = int(input("Please input the minimum value of PCI for eNB: "))
    enb_pci_max = int(input("Please input the maximum value of PCI for eNB: "))
    debug_print(f"eNB PCI range is from {enb_pci_min} to {enb_pci_max}")

    gnb_pci_min = int(input("Please input the minimum value of PCI for gNB: "))
    gnb_pci_max = int(input("Please input the maximum value of PCI for gNB: "))
    debug_print(f"gNB PCI range is from {gnb_pci_min} to {gnb_pci_max}")

    # 构建网络拓扑
    network_topology_construct(
        X_MAX, Y_MAX, N_NODES, R_ENB_MIN, R_ENB_MAX, R_GNB_MIN, R_GNB_MAX
    )
    debug(f"Number of eNB is {len(enbList)}")
    debug(f"Number of gNB is {len(gnbList)}")
    debug(f"Total number of eNB and gNB is {len(nodeList)}")

    # 构建邻区列表
    neighbor_list_construct(enbList, gnbList, R_SUM_MULTIPLIER)

    # 输出训练参数
    debug_print(
        f"N_HIDDEN_NEURON is {N_HIDDEN_NEURON}, "
        f"INIT_ALPHA is {INIT_ALPHA} and ALPHA_ADJUST_FACTOR is {ALPHA_ADJUST_FACTOR}, "
        f"INIT_EPSILON is {INIT_EPSILON} and EPSILON_ADJUST_FACTOR is {EPSILON_ADJUST_FACTOR}, "
        f"REWARD_ADJUST_FACTOR is {REWARD_ADJUST_FACTOR}, INIT_GAMMA is {INIT_GAMMA}"
    )

    #输出当前节点总数，基站类型比例及数目，覆盖半径，PCI取值范围，loss列表，回合数，每一轮需要更改的节点数
    debug_print(f"Number of nodes is {N_NODES}")
    debug_print(f"Number of eNB is {len(enbList)}")
    debug_print(f"Number of gNB is {len(gnbList)}")


    # 强化学习
    rl()
