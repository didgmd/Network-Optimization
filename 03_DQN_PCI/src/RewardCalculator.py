# 自定义模块
from DebugPrint import *

# 定义全局冲突和混淆点数目列表
globalCollisionNumList = []
globalConfusionNumList = []
globalenbMod30CollisionList = []
globalenbMod6CollisionList = []
globalenbMod3CollisionList = []
globalgnbMod30CollisionList = []
globalgnbMod4CollisionList = []
globalgnbMod3CollisionList = []

def reward_calculation(enb_list, gnb_list, node_list):
    # 初始化冲突与混淆坐标列表
    for node in node_list:
        node.collisionList.clear()
        node.confusionList.clear()
        node.enbMod30CollisionList.clear()
        node.enbMod6CollisionList.clear()
        node.enbMod3CollisionList.clear()
        node.gnbMod30CollisionList.clear()
        node.gnbMod4CollisionList.clear()
        node.gnbMod3CollisionList.clear()

    # 初始化奖励
    reward = 0

    # 原来的惩罚值过大，现将其减小到十分之一 2023.11.17
    ENB_MOD3_PENALTY = -0.01
    ENB_MOD6_PENALTY = -0.02
    ENB_MOD30_PENALTY = -0.03
    GNB_MOD3_PENALTY = -0.01
    GNB_MOD4_PENALTY = -0.02
    GNB_MOD30_PENALTY = -0.03
    CONFUSION_PENALTY = -0.02

    # 初始化需要改变PCI的节点索引列表
    nodesToChangePCIIndexList = []

    #初始化各冲突节点数量
    nenbMod30Collision=0
    nenbMod6Collision=0
    nenbMod3Collision=0
    ngnbMod30Collision=0
    ngnbMod4Collision=0
    ngnbMod3Collision=0

    # 初始化需要改变PCI的节点数量
    nNodesToChangePCI = 0

    # 检查eNB间的PCI冲突
    for enb in enb_list:
        for neighbor_cell in enb.neighborList:
            # 如果该邻区已经在冲突列表中，则跳过
            if (neighbor_cell.posX, neighbor_cell.posY) in enb.collisionList:
                continue
            # 如果该邻区未在冲突列表中，则判断是否存在冲突，判断冲突优先级为30 > 6 > 3
            flagCollision = False  # 用于标记是否存在冲突
            flagenbmod3Collision = False
            flagenbmod6Collision = False
            flagenbmod30Collision = False

            if (enb.pci % 30) == (neighbor_cell.pci % 30):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 30 Collision"
                )
                reward += ENB_MOD30_PENALTY
                flagCollision = True
                flagenbmod30Collision = True
            elif (enb.pci % 6) == (neighbor_cell.pci % 6):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 6 Collision"
                )
                reward += ENB_MOD6_PENALTY
                flagCollision = True
                flagenbmod6Collision = True
            elif (enb.pci % 3) == (neighbor_cell.pci % 3):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 3 Collision"
                )
                reward += ENB_MOD3_PENALTY
                flagCollision = True
                flagenbmod3Collision = True

            # 如果存在冲突，则互相加入对方的冲突列表
            if flagCollision:
                enb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((enb.posX, enb.posY))
                if (node_list.index(enb) not in nodesToChangePCIIndexList) and (
                    node_list.index(neighbor_cell) not in nodesToChangePCIIndexList
                ):
                    nodesToChangePCIIndexList.append(node_list.index(enb))
                    nNodesToChangePCI += 1

            if flagenbmod3Collision:
                enb.enbMod3CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.enbMod3CollisionList.append((enb.posX, enb.posY))
                nenbMod3Collision += 1

            if flagenbmod6Collision:
                enb.enbMod6CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.enbMod6CollisionList.append((enb.posX, enb.posY))
                nenbMod6Collision += 1
 
            if flagenbmod30Collision:
                enb.enbMod30CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.enbMod30CollisionList.append((enb.posX, enb.posY))
                nenbMod30Collision += 1

    # 检查gNB间的PCI冲突
    for gnb in gnb_list:
        for neighbor_cell in gnb.neighborList:
            # 如果该邻区已经在冲突列表中，则跳过
            if (neighbor_cell.posX, neighbor_cell.posY) in gnb.collisionList:
                continue
            # 如果该邻区未在冲突列表中，则判断是否存在冲突，判断冲突优先级为30 > 4 > 3
            flagCollision = False  # 用于标记是否存在冲突
            flaggnbmod3Collision = False
            flaggnbmod4Collision = False
            flaggnbmod30Collision = False
            if (gnb.pci % 30) == (neighbor_cell.pci % 30):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 30 Collision"
                )
                reward += GNB_MOD30_PENALTY
                flagCollision = True
                flaggnbmod30Collision = True
            elif (gnb.pci % 4) == (neighbor_cell.pci % 4):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 4 Collision"
                )
                reward += GNB_MOD4_PENALTY
                flagCollision = True
                flaggnbmod4Collision = True
            elif (gnb.pci % 3) == (neighbor_cell.pci % 3):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 3 Collision"
                )
                reward += GNB_MOD3_PENALTY
                flagCollision = True
                flaggnbmod3Collision = True

            # 如果存在冲突，则互相加入对方的冲突列表
            if flagCollision:
                gnb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((gnb.posX, gnb.posY))
                if (node_list.index(gnb) not in nodesToChangePCIIndexList) and (
                    node_list.index(neighbor_cell) not in nodesToChangePCIIndexList
                ):
                    nodesToChangePCIIndexList.append(node_list.index(gnb))
                    nNodesToChangePCI += 1

            if flaggnbmod3Collision:
                gnb.gnbMod3CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.gnbMod3CollisionList.append((gnb.posX, gnb.posY))
                ngnbMod3Collision += 1
           
            if flaggnbmod4Collision:
                gnb.gnbMod4CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.gnbMod4CollisionList.append((gnb.posX, gnb.posY))
                ngnbMod4Collision += 1
            
            if flaggnbmod30Collision:
                gnb.gnbMod30CollisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.gnbMod30CollisionList.append((gnb.posX, gnb.posY))
                ngnbMod30Collision += 1


    # 检查PCI混淆
    for node in node_list:
        for first_neighbor_cell in node.neighborList:
            for second_neighbor_cell in node.neighborList:
                # 如果第一个邻区节点和第二个邻区节点是同一个节点，则跳过
                if first_neighbor_cell == second_neighbor_cell:
                    continue

                # 如果第一个邻区节点和第二个邻区节点已经在对方混淆列表中，则跳过
                if (
                    second_neighbor_cell.posX,
                    second_neighbor_cell.posY,
                ) in first_neighbor_cell.confusionList:
                    continue

                # 如果第一个邻区节点和第二个邻区节点未在对方混淆列表中，则判断是否存在混淆
                if first_neighbor_cell.pci == second_neighbor_cell.pci:
                    debug(
                        f"Neighbor Cell ({first_neighbor_cell.posX}, {first_neighbor_cell.posY}) "
                        f"with PCI {first_neighbor_cell.pci} and "
                        f"Neighbor Cell ({second_neighbor_cell.posX}, {second_neighbor_cell.posY}) "
                        f"with PCI {second_neighbor_cell.pci} of "
                        f"Node ({node.posX}, {node.posY}) with PCI {node.pci} have PCI Confusion"
                    )
                    reward += CONFUSION_PENALTY
                    first_neighbor_cell.confusionList.append(
                        (second_neighbor_cell.posX, second_neighbor_cell.posY)
                    )
                    second_neighbor_cell.confusionList.append(
                        (first_neighbor_cell.posX, first_neighbor_cell.posY)
                    )

                    if (
                        node_list.index(first_neighbor_cell)
                        not in nodesToChangePCIIndexList
                    ) and (
                        node_list.index(second_neighbor_cell)
                        not in nodesToChangePCIIndexList
                    ):
                        nodesToChangePCIIndexList.append(
                            node_list.index(first_neighbor_cell)
                        )
                        nNodesToChangePCI += 1

    # 在reward_calculation函数中计算冲突点数目
    collisionPointList = []
    numOfCollision = 0
    for node in node_list:
        if node.collisionList:
            collisionPointList += node.collisionList
        numOfCollision = len(set(collisionPointList))

    # 在reward_calculation函数中计算混淆点数目
    confusion_points = []
    numOfConfusion = 0
    for node in node_list:
        if node.confusionList:
            confusion_points += node.confusionList
        numOfConfusion = len(set(confusion_points))

    # 在计算冲突和混淆点数目后，将其添加到列表中
    globalCollisionNumList.append(numOfCollision)
    globalConfusionNumList.append(numOfConfusion)

    # 在reward_calculation函数中计算各冲突点数目
    enbMod30CollisionList = []
    numOfenbMod30Collision = 0
    for enb in enb_list:
        if enb.enbMod30CollisionList:
            enbMod30CollisionList += enb.enbMod30CollisionList
        numOfenbMod30Collision = len(set(enbMod30CollisionList))

    enbMod6CollisionList = []
    numOfenbMod6Collision = 0
    for enb in enb_list:
        if enb.enbMod6CollisionList:
            enbMod6CollisionList += enb.enbMod6CollisionList
        numOfenbMod6Collision = len(set(enbMod6CollisionList))

    enbMod3CollisionList = []
    numOfenbMod3Collision = 0
    for enb in enb_list:
        if enb.enbMod3CollisionList:
            enbMod3CollisionList += enb.enbMod3CollisionList
        numOfenbMod3Collision = len(set(enbMod3CollisionList))

    globalenbMod30CollisionList.append(numOfenbMod30Collision)
    globalenbMod6CollisionList.append(numOfenbMod6Collision)
    globalenbMod3CollisionList.append(numOfenbMod3Collision)
    
    gnbMod30CollisionList = []
    numOfgnbMod30Collision = 0
    for gnb in gnb_list:
        if gnb.gnbMod30CollisionList:
            gnbMod30CollisionList += gnb.gnbMod30CollisionList
        numOfgnbMod30Collision = len(set(gnbMod30CollisionList))
    
    gnbMod4CollisionList = []
    numOfgnbMod4Collision = 0
    for gnb in gnb_list:
        if gnb.gnbMod4CollisionList:
            gnbMod4CollisionList += gnb.gnbMod4CollisionList
        numOfgnbMod4Collision = len(set(gnbMod4CollisionList))
    
    gnbMod3CollisionList = []
    numOfgnbMod3Collision = 0
    for gnb in gnb_list:
        if gnb.gnbMod3CollisionList:
            gnbMod3CollisionList += gnb.gnbMod3CollisionList
        numOfgnbMod3Collision = len(set(gnbMod3CollisionList))

    globalgnbMod30CollisionList.append(numOfgnbMod30Collision)
    globalgnbMod4CollisionList.append(numOfgnbMod4Collision)
    globalgnbMod3CollisionList.append(numOfgnbMod3Collision)  
    
    if reward == 0:
        reward = 1

    return reward, nNodesToChangePCI, nodesToChangePCIIndexList
