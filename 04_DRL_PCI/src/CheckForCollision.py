# -*- coding:utf8 -*-
from DebugPrint import *


def check_for_enb_pci_collision(enb_list, nodesToChangePCIIndexList):  
      
    # 定义计数器
    counter_enb_mod_30, counter_enb_mod_6, counter_enb_mod_3 = 0, 0, 0

    # 优先级定义：30 > 6 > 3  # ToDo @LJN: 优先级定义能否更加合理？
    # 检查eNB间的PCI模30冲突，然后检查PCI模6冲突，最后检查PCI模3冲突
    for enb in enb_list:
        # debug(f"Size of neighborList: {len(enb.neighborList)}")
        for neighbor_cell in enb.neighborList:
            # 如果该邻区已经在冲突列表中，则跳过
            if (neighbor_cell.posX, neighbor_cell.posY) in enb.collisionList:
                continue
            # 如果该邻区未在冲突列表中，则判断是否存在冲突，判断冲突优先级为30 > 6 > 3
            if (enb.pci % 30) == (neighbor_cell.pci % 30):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 30 Collision"
                )
                counter_enb_mod_30 += 1
                enb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((enb.posX, enb.posY))

                if enb_list.index(enb) not in nodesToChangePCIIndexList and enb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(enb_list.index(enb))
                    debug(f"add {enb_list.index(enb)} to nodesToChangePCIIndexList")

            elif (enb.pci % 6) == (neighbor_cell.pci % 6):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 6 Collision"
                )
                counter_enb_mod_6 += 1
                enb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((enb.posX, enb.posY))

                if enb_list.index(enb) not in nodesToChangePCIIndexList and enb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(enb_list.index(enb))
                    debug(f"add {enb_list.index(enb)} to nodesToChangePCIIndexList")

            elif (enb.pci % 3) == (neighbor_cell.pci % 3):
                debug(
                    f"eNB ({enb.posX}, {enb.posY}) with PCI {enb.pci} and "
                    f"eNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 3 Collision"
                )
                counter_enb_mod_3 += 1
                enb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((enb.posX, enb.posY))

                if enb_list.index(enb) not in nodesToChangePCIIndexList and enb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(enb_list.index(enb))
                    debug(f"add {enb_list.index(enb)} to nodesToChangePCIIndexList")
   
    debug(f"counter_enb_mod_30: {counter_enb_mod_30}, counter_enb_mod_6: {counter_enb_mod_6}, counter_enb_mod_3: {counter_enb_mod_3}")

    return counter_enb_mod_30, counter_enb_mod_6, counter_enb_mod_3, nodesToChangePCIIndexList


def check_for_gnb_pci_collision(gnb_list, nodesToChangePCIIndexList):

    # 定义计数器
    counter_gnb_mod_30, counter_gnb_mod_4, counter_gnb_mod_3 = 0, 0, 0

    # 优先级定义：30 > 4 > 3  # ToDo @LJN: 优先级定义能否更加合理？
    # 检查gNB间的PCI模30冲突，然后检查PCI模4冲突，最后检查PCI模3冲突
    for gnb in gnb_list:
        # debug(f"Size of neighborList: {len(gnb.neighborList)}")
        for neighbor_cell in gnb.neighborList:
            # 如果该邻区已经在冲突列表中，则跳过
            if (neighbor_cell.posX, neighbor_cell.posY) in gnb.collisionList:
                continue
            # 如果该邻区未在冲突列表中，则判断是否存在冲突，判断冲突优先级为30 > 6 > 3
            if (gnb.pci % 30) == (neighbor_cell.pci % 30):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 30 Collision"
                )
                counter_gnb_mod_30 += 1
                gnb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((gnb.posX, gnb.posY))

                if gnb_list.index(gnb) not in nodesToChangePCIIndexList and gnb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(gnb_list.index(gnb))
                    debug(f"add {gnb_list.index(gnb)} to nodesToChangePCIIndexList")

            elif (gnb.pci % 4) == (neighbor_cell.pci % 4):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 4 Collision"
                )
                counter_gnb_mod_4 += 1
                gnb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((gnb.posX, gnb.posY))

                if gnb_list.index(gnb) not in nodesToChangePCIIndexList and gnb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(gnb_list.index(gnb))
                    debug(f"add {gnb_list.index(gnb)} to nodesToChangePCIIndexList")

            elif (gnb.pci % 3) == (neighbor_cell.pci % 3):
                debug(
                    f"gNB ({gnb.posX}, {gnb.posY}) with PCI {gnb.pci} and "
                    f"gNB ({neighbor_cell.posX}, {neighbor_cell.posY}) with PCI {neighbor_cell.pci} "
                    f"have PCI Mod 3 Collision"
                )
                counter_gnb_mod_3 += 1
                gnb.collisionList.append((neighbor_cell.posX, neighbor_cell.posY))
                neighbor_cell.collisionList.append((gnb.posX, gnb.posY))

                if gnb_list.index(gnb) not in nodesToChangePCIIndexList and gnb_list.index(neighbor_cell) not in nodesToChangePCIIndexList:
                    nodesToChangePCIIndexList.append(gnb_list.index(gnb))
                    debug(f"add {gnb_list.index(gnb)} to nodesToChangePCIIndexList")

    return counter_gnb_mod_30, counter_gnb_mod_4, counter_gnb_mod_3, nodesToChangePCIIndexList
