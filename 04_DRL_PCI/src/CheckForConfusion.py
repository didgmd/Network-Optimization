# -*- coding:utf8 -*-
from DebugPrint import *


def check_for_enb_pci_confusion(enb_list, nodesToChangePCIIndexList):
    # 定义计数器
    counter_enb_confusion = 0

    # 检查eNB间的PCI混淆
    for enb in enb_list:
        # debug(f"Size of neighborList: {len(enb.neighborList)}")

        for first_neighbor_cell in enb.neighborList:
            for second_neighbor_cell in enb.neighborList:
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
                        f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} have PCI Confusion"
                    )
                    first_neighbor_cell.confusionList.append(
                        (second_neighbor_cell.posX, second_neighbor_cell.posY)
                    )
                    second_neighbor_cell.confusionList.append(
                        (first_neighbor_cell.posX, first_neighbor_cell.posY)
                    )
                    counter_enb_confusion += 1

                    if (enb_list.index(first_neighbor_cell) not in nodesToChangePCIIndexList) and (enb_list.index(second_neighbor_cell) not in nodesToChangePCIIndexList):
                        nodesToChangePCIIndexList.append(enb_list.index(first_neighbor_cell))
                        debug(f"add {enb_list.index(first_neighbor_cell)} to nodesToChangePCIIndexList")


    return counter_enb_confusion, nodesToChangePCIIndexList


def check_for_gnb_pci_confusion(gnb_list, nodesToChangePCIIndexList):
    # 定义计数器
    counter_gnb_confusion = 0

    # 检查gNB间的PCI混淆
    for gnb in gnb_list:
        # debug(f"Size of neighborList: {len(gnb.neighborList)}")

        for first_neighbor_cell in gnb.neighborList:
            for second_neighbor_cell in gnb.neighborList:
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
                        f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} have PCI Confusion"
                    )
                    first_neighbor_cell.confusionList.append(
                        (second_neighbor_cell.posX, second_neighbor_cell.posY)
                    )
                    second_neighbor_cell.confusionList.append(
                        (first_neighbor_cell.posX, first_neighbor_cell.posY)
                    )
                    counter_gnb_confusion += 1

                    if (gnb_list.index(first_neighbor_cell) not in nodesToChangePCIIndexList) and (gnb_list.index(second_neighbor_cell) not in nodesToChangePCIIndexList):
                        nodesToChangePCIIndexList.append(gnb_list.index(first_neighbor_cell))
                        debug(f"add {gnb_list.index(first_neighbor_cell)} to nodesToChangePCIIndexList")

    return counter_gnb_confusion, nodesToChangePCIIndexList
