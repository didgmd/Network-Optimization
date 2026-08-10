import time
import DataSave

# 自定义模块
from QNet import *
from NetworkTopology import *
from PciAllocation import *
from NeighborList import *
from ActionChooser import *
from StateChanger import *
from RewardCalculator import *
from FigurePlotter import *
from Normalization import *
from Validation import *


# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_print(f"device is {device}")


def rl():
    qNet = QNet(N_INPUT_FEATURE, N_HIDDEN_NEURON, N_OUTPUT_FEATURE).to(device)
    debug(qNet)
    # 增加动态调整学习率机制 2023.11.16
    flagAlphaDynamicAdjustment = True
    if flagAlphaDynamicAdjustment:
        optimizer = torch.optim.Adam(qNet.parameters(), lr=INIT_ALPHA)
        # 动态调整学习率：初始化调度器
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=ALPHA_ADJUST_FACTOR
        )
    else:
        optimizer = torch.optim.Adam(qNet.parameters(), lr=INIT_ALPHA)

    # 在程序开始时初始化两个全局列表，分别用于启用/禁用引导目标值策略时每个Epoch的Episode数目
    episodesListGuidedTargetEnabled = []
    episodesListGuidedTargetDisabled = []

    #for flagGuidedTarget in [True, False]:
    for flagGuidedTarget in [True]:
    #for flagGuidedTarget in [False]:
        # 初始化估计值
        qPredict = 0.0

        # 初始化主循环计数器
        epoch = 1

        # 创建列表存储每Epoch最终的损失函数值
        lossList = []

        # 动态调整学习率：初始化损失函数最小值
        lossMin = 0.0
        alphaList = []

        # 初始化EMA_LOSS机制所需变量
        prevLoss = 0.0

        # 每轮主循环首先对每个节点进行PCI分配，然后定义初始当前状态即输入特征列表，接着定义下个状态
        while True:
            # 主循环开始前的时间
            mainLoopStartTime = time.time()
            
            debug_print(f"################ Epoch {epoch} Start ################")

            # 初始化PCI池
            for objPCI in pciPooleNB:
                objPCI.isUsed = False
                objPCI.nodeXList.clear()
                objPCI.nodeYList.clear()
            for objPCI in pciPoolgNB:
                objPCI.isUsed = False
                objPCI.nodeXList.clear()
                objPCI.nodeYList.clear()

            # 初始化节点PCI为-1
            for objNode in nodeList:
                objNode.pci = -1

            # 分配初始PCI
            pci_initial_allocation(enbList, gnbList)

            if DEBUG_MODE:
                for enb in enbList:
                    debug(enb)
                    for neighbor_enb in enb.neighborList:
                        debug(f"Neighbor {neighbor_enb}")
                for gnb in gnbList:
                    debug(gnb)
                    for neighbor_gnb in gnb.neighborList:
                        debug(f"Neighbor {neighbor_gnb}")

            if epoch == 1:
                # 绘制eNB分布图
                plot_enb(enbList, X_MAX, Y_MAX)

                # 绘制gNB分布图
                plot_gnb(gnbList, X_MAX, Y_MAX)

            # 定义初始当前状态即输入特征列表
            currState = []
            for objNode in nodeList:
                currState.append(objNode.pci)

            # 初始化子循环计数器
            episode = 1

            # 初始化奖励动态调整机制所需变量
            nNodesToChangePCIPrev = 0

            # 初始化贪婪度动态调整机制所需变量
            epsilon = INIT_EPSILON
            nNodesToChangePCIMin = 0

            # 初始化Epoch损失函数列表
            epochLossList = []

            # EMA：每轮子循环前，初始化估计值
            qPredict = 0.0

            # 每轮子循环首先选择一个动作索引，以此修改相应节点的PCI并计算下个状态即输入特征列表。
            # 然后根据当前状态计算估计值，再根据奖励和下个状态计算目标值，最后计算损失函数并反向传播
            while True:

                # 子循环开始前的时间
                subLoopStartTime = time.time()
                # debug(
                #     f"################ Epoch {epoch} Episode {episode} Start: "
                #     f"{get_num_available_pci_enb()} PCIs available for eNBs, "
                #     f"{get_num_available_pci_gnb()} PCIs available for gNBs ################"
                # )
                debug(
                    f"################ Epoch {epoch} Episode {episode} Start ################"
                )
                debug(f"currState is {currState}")

                # 计算奖励并获取需要改变PCI的节点数量及节点索引列表，索引列表暂未使用，后续可考虑根据该索引列表确定目标值
                (
                    reward,
                    nNodesToChangePCI,
                    nodesToChangePCIIndexList,
                ) = reward_calculation(enbList, gnbList, nodeList)
                debug(
                    f"reward is {reward} and nNodesToChangePCI is {nNodesToChangePCI}"
                )
            
                # 增加奖励动态调整机制 2023.11.15
                flagRewardDynamicAdjustment = True
                if flagRewardDynamicAdjustment:
                    if episode == 1:
                        nNodesToChangePCIPrev = nNodesToChangePCI
                    elif episode > 1:
                        if nNodesToChangePCI > nNodesToChangePCIPrev:
                            reward -= abs(reward) * REWARD_ADJUST_FACTOR
                            debug(f"reward after dynamic adjustment is {reward}")
                        elif nNodesToChangePCI < nNodesToChangePCIPrev:
                            reward += abs(reward) * REWARD_ADJUST_FACTOR
                            debug(f"reward after dynamic adjustment is {reward}")

                        nNodesToChangePCIPrev = nNodesToChangePCI

                # 绘制冲突点和混淆点的图像
                # episodesForPlot = [100, 200, 400]
                # if episode in episodesForPlot:
                # if episode % 1000 == 0 or episode == 1:
                #     plot_collision_and_confusion(nodeList, epoch, episode, X_MAX, Y_MAX)

                # 由于本项目中下个状态基于当前状态改变，因此可以先将下个状态初始化为当前状态
                nextState = currState.copy()

                # 对当前状态进行归一化处理
                # 此处，仅将归一化后的currStateNorm作为神经网络的输入，而currState仍作为nextState的前置状态
                currStateNorm = min_max_normalization(nodeList, currState)
                debug(f"currStateNorm is {currStateNorm}")

                # 增加贪婪度动态调整机制 2023.11.15
                flagEpsilonDynamicAdjustment = True
                if flagEpsilonDynamicAdjustment:
                    if episode == 1:
                        nNodesToChangePCIMin = nNodesToChangePCI
                    elif episode > 1:
                        if epsilon != 1 and nNodesToChangePCI < nNodesToChangePCIMin:
                            if (epsilon * EPSILON_ADJUST_FACTOR) > 1:
                                epsilon = 1
                            else:
                                epsilon *= EPSILON_ADJUST_FACTOR
                            debug(f"Epsilon after dynamic adjustment is {epsilon}")

                # 根据需要改变PCI的节点数量动态选择动作索引列表
                actionIndexList = choose_action(
                    currStateNorm,
                    qNet,
                    epsilon,
                    len(nodeList),
                    device,
                    nNodesToChangePCI,
                )
                debug(f"actionIndexList is {actionIndexList}")

                # 确定下个状态即下一次的输入特征值列表
                for actionIndex in actionIndexList:
                    if nodeList[actionIndex].nodeType == "enb":
                        nextState = change_state(
                            nextState,
                            nodeList,
                            actionIndex,
                            pciPooleNB,
                            R_SUM_MULTIPLIER,
                        )
                    elif nodeList[actionIndex].nodeType == "gnb":
                        nextState = change_state(
                            nextState,
                            nodeList,
                            actionIndex,
                            pciPoolgNB,
                            R_SUM_MULTIPLIER,
                        )
                debug(f"nextState is {nextState}")

                # 根据EMA（指数移动平均）计算估计值，当通过神经网络选择动作时，动作索引列表中的首项应为Q值最大项
                debug(f"qPredict in last episode is {qPredict}")

                if (
                    len(actionIndexList) == 0
                ):  # 如果动作索引列表为空，则说明本轮循环中没有需要改变PCI的节点，此时应使用上轮循环中计算得到的估计值
                    if qPredict == 0.0:
                        qPredict = qNet(
                            torch.tensor(currStateNorm).float().to(device)
                        ).max()
                    else:
                        qPredict = qPredict.data  # 此处等式右侧的qPredict应为上轮循环中计算得到的估计值
                else:
                    if qPredict == 0.0:
                        qPredict = qNet(torch.tensor(currStateNorm).float().to(device))[
                            actionIndexList[0]
                        ]
                    else:
                        qPredict = (
                            EMA_ALPHA
                            * qNet(torch.tensor(currStateNorm).float().to(device))[
                                actionIndexList[0]
                            ]
                            + (1 - EMA_ALPHA) * qPredict.data
                        )  # 此处等式右侧的qPredict应为上轮循环中计算得到的估计值

                debug(f"qPredict in this episode is {qPredict}")
                # debug(f"type of qPredict is {type(qPredict)}")

                # 对下个状态进行归一化处理
                # 此处，仅将归一化后的nextStateNorm作为神经网络的输入，而nextState仍作为下轮currState的前置状态
                nextStateNorm = min_max_normalization(nodeList, nextState)
                debug(f"nextStateNorm is {nextStateNorm}")

                # 统计冲突点和混淆点的数目
                #if episode in [1,3,5,10]:
                    #print(f"Episode {episode} of epoch {epoch}:globalCollisionNumList is {globalCollisionNumList}")

                # 重新启用引导目标值的方法 2023.11.15
                if flagGuidedTarget:
                    # 根据nodesToChangePCIIndexList选择目标值
                    outputQList = qNet(torch.tensor(nextStateNorm).float().to(device))
                    outputMaxQ = outputQList.min()
                    for leadIndex in nodesToChangePCIIndexList:
                        if outputQList[leadIndex] > outputMaxQ:
                            outputMaxQ = outputQList[leadIndex]
                    # debug_print(nodesToChangePCIIndexList)

                    qTarget = reward + GAMMA * outputMaxQ
                else:
                    qTarget = (
                        reward
                        + GAMMA
                        * qNet(torch.tensor(nextStateNorm).float().to(device)).max()
                    )

                debug(f"qTarget is {qTarget}")
                # debug(f"type of qTarget is {type(qTarget)}")

                # 计算损失函数
                flagHuberLoss = True
                if flagHuberLoss:
                    objSmoothL1Loss = nn.SmoothL1Loss(reduction="mean")
                    loss = objSmoothL1Loss(qPredict, qTarget)
                else:
                    loss = nn.functional.mse_loss(qPredict, qTarget)

                # epochLossList.append(loss.cpu().detach().numpy().item())
                # avgLoss = sum(epochLossList) / len(epochLossList)

                # 增加EMA_LOSS机制 2023.11.17
                avgLoss = (
                    EMA_LOSS * loss.cpu().detach().numpy().item()
                    + (1 - EMA_LOSS) * prevLoss
                )
                prevLoss = avgLoss

                if episode % 1000 == 0:
                    debug_print(
                        f"Average loss is {avgLoss} while nNodesToChangePCI is {nNodesToChangePCI} at "
                        f"episode {episode} of epoch {epoch}"
                    )
                optimizer.zero_grad()
                loss.backward()  # 反向传播
                optimizer.step()

                # 确认子循环收敛条件
                # if reward >= 0:
                if nNodesToChangePCI == 0:
                    if flagGuidedTarget:
                        episodesListGuidedTargetEnabled.append(episode)
                    else:
                        episodesListGuidedTargetDisabled.append(episode)
                    # lossList.append(loss.cpu().detach().numpy().item())
                    lossList.append(avgLoss)
                    debug_print(
                        f"Epoch {epoch} converged at episode {episode}, final reward is {reward}, "
                        f"average loss is {avgLoss}."
                    )

                    # 动态调整学习率：每个epoch结束后，调整学习率
                    if flagAlphaDynamicAdjustment:
                        if epoch == 1:
                            lossMin = avgLoss
                        elif epoch > 1 and avgLoss < lossMin:
                            lossMin = avgLoss
                            scheduler.step()
                            alphaList.append(optimizer.param_groups[0]["lr"])
                            debug_print(f"alphaList is {alphaList}")

                    # 绘制冲突点和混淆点的图像
                    # plot_collision_and_confusion_final(nodeList, X_MAX, Y_MAX, epoch)


                    epochsForPlot = [1, 20, 50, 70, 100, 130, 160, 200, 250, 300, 400, 500, 600]
                    if epoch in epochsForPlot:
                        #绘制冲突和混淆点数目的曲线图
                        plot_collision_and_confusion_curve(
                            globalCollisionNumList, globalConfusionNumList, epoch
                        )

                    # 输出冲突和混淆点数目的最终值
                    # debug_print(f"Final Collision Number: {globalCollisionNumList[-1]}")
                    # debug_print(f"Final Confusion Number: {globalConfusionNumList[-1]}")

                    # 将列表保存到Excel文件中
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
                    
                    # 每轮优化结束后，清除冲突点和混淆点全局列表
                    globalCollisionNumList.clear()
                    globalConfusionNumList.clear()

                    # 子循环结束后的时间
                    subLoopEndTime = time.time()

                    # 计算子循环运行时间
                    subLoopDuration = subLoopEndTime - subLoopStartTime
                    debug_print(
                        f"*******************子循环收敛时间: {subLoopDuration} seconds*****************"
                    )

                    break

                # 状态切换
                currState = nextState.copy()

                # 更新子循环计数器
                episode += 1

            # if epoch>20:
            #     break

            debug_print(f"lossList is {lossList}")

            # 确认主循环收敛条件
            # if loss <= 0.001:
            if epoch >= 10 and avgLoss <= 0.1:
                debug_print(
                    f"Process ended at epoch {epoch} episode {episode} due to average loss reaches {avgLoss}."
                )

                # 绘制loss损失函数值的曲线图
                plot_loss_curve(lossList)

                # 绘制每个epoch中的episode数目的曲线图
                plot_episodes_in_each_epoch(
                    episodesListGuidedTargetEnabled, episodesListGuidedTargetDisabled
                )

                # 主循环结束后的时间
                mainLoopEndTime = time.time()

                # 计算主循环运行时间
                mainLoopDuration = mainLoopEndTime - mainLoopStartTime
                debug_print(
                    f"********************主循环收敛时间: {mainLoopDuration} seconds***********************"
                )

                break
            #清空全局列表，重新计数
            #globalenbMod30CollisionList.clear()

            # 更新主循环计数器
            epoch += 1

        # 将losslist保存到Excel文件中
        list_to_save = [str(item) for item in lossList]
        list_name = "lossList"
        DataSave.save_data_to_excel(list_to_save, list_name)

        # 测试部分
        # 由于PCI优化过程中可以允许得到反馈，因此考虑两种测试方式：有反馈和无反馈
        while True:
            set_debug_mode(False)
            testWithFeedback = input(
                "The training process is over. "
                "Before we start the testing process, "
                "please choose whether to test with feedback or not. y/n: "
            )

            if testWithFeedback == "y":
                debug_print(
                    "################ Test with Feedback Start ################"
                )
                test_with_feedback(
                    qNet,
                    REWARD_ADJUST_FACTOR,
                    device,
                    R_SUM_MULTIPLIER,
                    EMA_ALPHA,
                    flagGuidedTarget,
                    GAMMA,
                    EMA_LOSS,
                    optimizer,
                    flagAlphaDynamicAdjustment,
                    scheduler,
                )
                break
            elif testWithFeedback == "n":
                debug_print(
                    "################ Test without Feedback Start ################"
                )
                test_without_feedback(qNet, device, R_SUM_MULTIPLIER)
                break
            else:
                debug_print("Invalid input, please try again.")
                continue


if __name__ == "__main__":
    set_debug_mode(False)

    # 网络大小
    X_MAX = 100
    Y_MAX = 100

    # 节点数量
    N_NODES = 150

    # 节点覆盖范围
    R_ENB_MIN = 2
    R_ENB_MAX = 3
    R_GNB_MIN = 1
    R_GNB_MAX = 1

    # 构建网络拓扑
    network_topology_construct(
        X_MAX, Y_MAX, N_NODES, R_ENB_MIN, R_ENB_MAX, R_GNB_MIN, R_GNB_MAX
    )
    debug(f"Number of eNB is {len(enbList)}")
    debug(f"Number of gNB is {len(gnbList)}")
    debug(f"Total number of eNB and gNB is {len(nodeList)}")

    # 定义半径和乘子作为邻区判定边界条件，由此，PCI重用距离即为邻区判定距离的2倍
    R_SUM_MULTIPLIER = 1.5
    # 构建邻区列表
    neighbor_list_construct(enbList, gnbList, R_SUM_MULTIPLIER)

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
    GAMMA = 0.9

    # 定义学习率动态调整因子
    ALPHA_ADJUST_FACTOR = 0.91  # INIT_ALPHA=0.004与ALPHA_ADJUST_FACTOR=0.91配合使用可收敛至0.002

    # 定义奖励动态调整因子
    REWARD_ADJUST_FACTOR = 0.2

    # 定义贪婪度动态调整因子
    EPSILON_ADJUST_FACTOR = 1.1

    # 输出训练参数
    debug_print(
        f"N_HIDDEN_NEURON is {N_HIDDEN_NEURON}, "
        f"INIT_ALPHA is {INIT_ALPHA} and ALPHA_ADJUST_FACTOR is {ALPHA_ADJUST_FACTOR}, "
        f"INIT_EPSILON is {INIT_EPSILON} and EPSILON_ADJUST_FACTOR is {EPSILON_ADJUST_FACTOR}, "
        f"REWARD_ADJUST_FACTOR is {REWARD_ADJUST_FACTOR}, GAMMA is {GAMMA}"
    )

    # 强化学习
    rl()
