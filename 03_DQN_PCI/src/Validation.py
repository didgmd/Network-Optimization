from torch import nn
import time

# 自定义模块
from PciAllocation import *
from NetworkTopology import *
from FigurePlotter import *
from RewardCalculator import *
from Normalization import *
from ActionChooser import *
from StateChanger import *


def test_with_feedback(
    q_net,
    reward_adjust_factor,
    device,
    r_sum_multiplier,
    ema_alpha,
    flag_guided_target,
    gamma,
    ema_loss,
    optimizer,
    flag_alpha_dynamic_adjustment,
    scheduler,
):
    # 初始化主循环计数器
    epoch = 1
    MAX_EPOCH = 1000

    # 动态调整学习率：初始化损失函数最小值
    lossMin = 0.0
    alphaList = []

    # 初始化EMA_LOSS机制所需变量
    prevLoss = 0.0

    while epoch <= MAX_EPOCH:
        debug_print(f"################ Test Epoch {epoch} Start ################")

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

        # EMA：每轮子循环前，初始化估计值
        qPredict = 0.0

        while True:
            debug(
                f"################ Test Epoch {epoch} Episode {episode} Start ################"
            )
            debug(f"Current state: {currState}")

            # 计算奖励并获取需要改变PCI的节点数量及节点索引列表，索引列表暂未使用，后续可考虑根据该索引列表确定目标值
            (
                reward,
                nNodesToChangePCI,
                nodesToChangePCIIndexList,
            ) = reward_calculation(enbList, gnbList, nodeList)
            debug(f"reward is {reward} and nNodesToChangePCI is {nNodesToChangePCI}")

            # 增加奖励动态调整机制 2023.11.15
            flagRewardDynamicAdjustment = True
            if flagRewardDynamicAdjustment:
                if episode == 1:
                    nNodesToChangePCIPrev = nNodesToChangePCI
                elif episode > 1:
                    if nNodesToChangePCI > nNodesToChangePCIPrev:
                        reward -= abs(reward) * reward_adjust_factor
                        debug(f"reward after dynamic adjustment is {reward}")
                    elif nNodesToChangePCI < nNodesToChangePCIPrev:
                        reward += abs(reward) * reward_adjust_factor
                        debug(f"reward after dynamic adjustment is {reward}")

                    nNodesToChangePCIPrev = nNodesToChangePCI

            # 绘制冲突点和混淆点的图像
            if epoch <10:
                episodesForPlot = [1,20,40,60,80,100,130,160, 200]
                if episode in episodesForPlot:
                #if episode % 1000 == 0 or episode == 1:
                    plot_collision_and_confusion(nodeList, epoch, episode, X_MAX, Y_MAX)
                    time.sleep(0.1 / 1000)  # 休眠0.1ms，防止图像闪烁

            # 由于本项目中下个状态基于当前状态改变，因此可以先将下个状态初始化为当前状态
            nextState = currState.copy()

            # 对当前状态进行归一化处理
            # 此处，仅将归一化后的currStateNorm作为神经网络的输入，而currState仍作为nextState的前置状态
            currStateNorm = min_max_normalization(nodeList, currState)
            debug(f"currStateNorm is {currStateNorm}")

            # 根据需要改变PCI的节点数量动态选择动作索引列表
            # 测试过程中，贪婪度设置为1
            actionIndexList = choose_action(
                currStateNorm,
                q_net,
                1,
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
                        r_sum_multiplier,
                    )
                elif nodeList[actionIndex].nodeType == "gnb":
                    nextState = change_state(
                        nextState,
                        nodeList,
                        actionIndex,
                        pciPoolgNB,
                        r_sum_multiplier,
                    )
            debug(f"nextState is {nextState}")

            # 根据EMA（指数移动平均）计算估计值，当通过神经网络选择动作时，动作索引列表中的首项应为Q值最大项
            debug(f"qPredict in last episode is {qPredict}")

            if (
                len(actionIndexList) == 0
            ):  # 如果动作索引列表为空，则说明本轮循环中没有需要改变PCI的节点，此时应使用上轮循环中计算得到的估计值
                if qPredict == 0.0:
                    qPredict = q_net(
                        torch.tensor(currStateNorm).float().to(device)
                    ).max()
                else:
                    qPredict = qPredict.data  # 此处等式右侧的qPredict应为上轮循环中计算得到的估计值
            else:
                if qPredict == 0.0:
                    qPredict = q_net(torch.tensor(currStateNorm).float().to(device))[
                        actionIndexList[0]
                    ]
                else:
                    qPredict = (
                        ema_alpha
                        * q_net(torch.tensor(currStateNorm).float().to(device))[
                            actionIndexList[0]
                        ]
                        + (1 - ema_alpha) * qPredict.data
                    )  # 此处等式右侧的qPredict应为上轮循环中计算得到的估计值

            debug(f"qPredict in this episode is {qPredict}")
            # debug(f"type of qPredict is {type(qPredict)}")

            # 对下个状态进行归一化处理
            # 此处，仅将归一化后的nextStateNorm作为神经网络的输入，而nextState仍作为下轮currState的前置状态
            nextStateNorm = min_max_normalization(nodeList, nextState)
            debug(f"nextStateNorm is {nextStateNorm}")

            # 重新启用引导目标值的方法 2023.11.15
            if flag_guided_target:
                # 根据nodesToChangePCIIndexList选择目标值
                outputQList = q_net(torch.tensor(nextStateNorm).float().to(device))
                outputMaxQ = outputQList.min()
                for leadIndex in nodesToChangePCIIndexList:
                    if outputQList[leadIndex] > outputMaxQ:
                        outputMaxQ = outputQList[leadIndex]
                # debug_print(nodesToChangePCIIndexList)

                qTarget = reward + gamma * outputMaxQ
            else:
                qTarget = (
                    reward
                    + gamma
                    * q_net(torch.tensor(nextStateNorm).float().to(device)).max()
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

            # 增加EMA_LOSS机制 2023.11.17
            avgLoss = (
                ema_loss * loss.cpu().detach().numpy().item()
                + (1 - ema_loss) * prevLoss
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

            if nNodesToChangePCI == 0:
                debug_print(
                    f"Epoch {epoch} converged at episode {episode}, final reward is {reward}, "
                    f"average loss is {avgLoss}."
                )

                # 动态调整学习率：每个epoch结束后，调整学习率
                if flag_alpha_dynamic_adjustment:
                    if epoch == 1:
                        lossMin = avgLoss
                    elif epoch > 1 and avgLoss < lossMin:
                        lossMin = avgLoss
                        scheduler.step()
                        alphaList.append(optimizer.param_groups[0]["lr"])
                        debug_print(f"alphaList is {alphaList}")

                if epoch < 10:
                #绘制冲突点和混淆点的图
                    plot_collision_and_confusion_final(nodeList, X_MAX, Y_MAX, episode,epoch)
                    time.sleep(0.1 / 1000)  # 休眠0.1ms，防止图像闪烁
                    image_path = os.path.join(global_folder, f'Collision_and_Confusion_epoch_{epoch}_at_episode_{episode}.pdf')
                    plt.savefig(image_path, dpi=300, format="pdf")


                # 绘制冲突和混淆点数目的曲线图
                # plot_collision_and_confusion_curve(
                #     globalCollisionNumList, globalConfusionNumList, epoch
                # )

                # 输出冲突和混淆点数目的最终值
                # debug_print(f"Final Collision Number: {globalCollisionNumList[-1]}")
                # debug_print(f"Final Confusion Number: {globalConfusionNumList[-1]}")

                # 每轮优化结束后，清除冲突点和混淆点全局列表
                globalCollisionNumList.clear()
                globalConfusionNumList.clear()

                break

            # 状态切换
            currState = nextState.copy()
            # 更新子循环计数器
            episode += 1

        # 更新主循环计数器
        epoch += 1


def test_without_feedback(q_net, device, r_sum_multiplier):
    # 初始化主循环计数器
    epoch = 1
    MAX_EPOCH = 1000

    while epoch <= MAX_EPOCH:
        debug_print(f"################ Test Epoch {epoch} Start ################")

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

        while True:
            debug(
                f"################ Test Epoch {epoch} Episode {episode} Start ################"
            )
            debug(f"Current state: {currState}")

            # 计算奖励并获取需要改变PCI的节点数量及节点索引列表，索引列表暂未使用，后续可考虑根据该索引列表确定目标值
            # 测试环节中，仅需计算需要改变PCI的节点数量，但为了保持函数调用的一致性，仍然调用reward_calculation函数
            (
                reward,
                nNodesToChangePCI,
                nodesToChangePCIIndexList,
            ) = reward_calculation(enbList, gnbList, nodeList)
            debug(f"reward is {reward} and nNodesToChangePCI is {nNodesToChangePCI}")

            # 绘制冲突点和混淆点的图像
            # episodesForPlot = [100, 200, 400]
            # if episode in episodesForPlot:
            if epoch < 10:
                episodesForPlot = [1, 20, 40, 60, 80, 100, 130, 160, 200]
                if episode in episodesForPlot:
                    # if episode % 1000 == 0 or episode == 1:
                    plot_collision_and_confusion(nodeList, epoch, episode, X_MAX, Y_MAX)
                    time.sleep(0.1 / 1000)  # 休眠0.1ms，防止图像闪烁

            # 由于本项目中下个状态基于当前状态改变，因此可以先将下个状态初始化为当前状态
            nextState = currState.copy()

            # 对当前状态进行归一化处理
            # 此处，仅将归一化后的currStateNorm作为神经网络的输入，而currState仍作为nextState的前置状态
            currStateNorm = min_max_normalization(nodeList, currState)
            debug(f"currStateNorm is {currStateNorm}")

            # 根据需要改变PCI的节点数量动态选择动作索引列表
            # 测试过程中，贪婪度设置为1
            actionIndexList = choose_action(
                currStateNorm,
                q_net,
                1,
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
                        r_sum_multiplier,
                    )
                elif nodeList[actionIndex].nodeType == "gnb":
                    nextState = change_state(
                        nextState,
                        nodeList,
                        actionIndex,
                        pciPoolgNB,
                        r_sum_multiplier,
                    )
            debug(f"nextState is {nextState}")

            if nNodesToChangePCI == 0:
                debug_print(f"Epoch {epoch} converged at episode {episode}.")

                # 绘制冲突点和混淆点的图像
                if epoch < 10:
                    # 绘制冲突点和混淆点的图
                    plot_collision_and_confusion_final(nodeList, X_MAX, Y_MAX, episode, epoch)
                    time.sleep(0.1 / 1000)  # 休眠0.1ms，防止图像闪烁
                    image_path = os.path.join(global_folder,
                                              f'Collision_and_Confusion_epoch_{epoch}_at_episode_{episode}.pdf')
                    plt.savefig(image_path, dpi=300, format="pdf")

                # 绘制冲突和混淆点数目的曲线图
                # plot_collision_and_confusion_curve(
                #     globalCollisionNumList, globalConfusionNumList, epoch
                # )

                # 输出冲突和混淆点数目的最终值
                # debug_print(f"Final Collision Number: {globalCollisionNumList[-1]}")
                # debug_print(f"Final Confusion Number: {globalConfusionNumList[-1]}")

                # 每轮优化结束后，清除冲突点和混淆点全局列表
                globalCollisionNumList.clear()
                globalConfusionNumList.clear()

                break

            # 状态切换
            currState = nextState.copy()
            # 更新子循环计数器
            episode += 1

        # 更新主循环计数器
        epoch += 1
