import numpy as np
from random import sample

# 自定义模块
from DebugPrint import *


def change_state(next_state, node_list, action_index, pci_pool, r_sum_multiplier):
    # 设置PCI重用距离阈值乘子，即PCI重用距离阈值为r_sum_multiplier * 2 * (r1 + r2)
    pciReuseThresholdMultiplier = r_sum_multiplier * 2
    # 选择PCI，并考虑是否超过PCI重用距离
    while True:
        # objPci = np.random.choice(pci_pool, replace=False)
        objPci = sample(pci_pool, 1)[0]
        debug(f"PCI {objPci.pci} is chosen")

        # 如果PCI未被使用，则直接分配
        if not objPci.isUsed:
            # BugFix20231118：在重新分配PCI之前，将该节点从原PCI的节点列表中删除，如果该原PCI的节点列表为空，则将其置为未被使用
            for objPciTemp in pci_pool:
                if objPciTemp.pci == node_list[action_index].pci:
                    objPciTemp.nodeXList.remove(node_list[action_index].posX)
                    objPciTemp.nodeYList.remove(node_list[action_index].posY)
                    if (
                        len(objPciTemp.nodeXList) == 0
                        and len(objPciTemp.nodeYList) == 0
                    ):
                        objPciTemp.isUsed = False
                        debug(f"PCI {objPciTemp.pci} is unused now")
                    break

            node_list[action_index].pci = objPci.pci
            objPci.isUsed = True
            objPci.nodeXList.append(node_list[action_index].posX)
            objPci.nodeYList.append(node_list[action_index].posY)
            break
        elif objPci.pci == node_list[action_index].pci:
            # BugFix20231117：如果随机抽取的PCI与当前待重新分配PCI的节点的PCI相同，则重新抽取
            debug(
                f"PCI {objPci.pci} is the same as "
                f"Node ({node_list[action_index].posX}, {node_list[action_index].posY})'s PCI"
            )
            continue
        else:
            # 如果PCI已被使用，则遍历使用该PCI的所有节点坐标，如果所有节点坐标与待重新分配PCI的节点坐标的距离均大于PCI重用距离阈值，则分配该PCI
            # 首先设定标志位为真，如果有一个或以上节点坐标与待重新分配PCI的节点坐标的距离小于等于PCI重用距离阈值，则将标志位设为假
            pciReuseAvailable = True
            for nodeX, nodeY, radius in zip(
                objPci.nodeXList, objPci.nodeYList, objPci.radiusList
            ):
                if (
                    np.sqrt(
                        (nodeX - node_list[action_index].posX) ** 2
                        + (nodeY - node_list[action_index].posY) ** 2
                    )
                    > (radius + node_list[action_index].radius)
                    * pciReuseThresholdMultiplier
                ):
                    debug(
                        f"Node ({nodeX}, {nodeY}) is far enough from "
                        f"Node ({node_list[action_index].posX}, {node_list[action_index].posY})"
                    )
                    continue
                else:
                    debug(
                        f"Node ({nodeX}, {nodeY}) is too close to "
                        f"Node ({node_list[action_index].posX}, {node_list[action_index].posY})"
                    )
                    pciReuseAvailable = False
                    break

            # 如果标志位为真，则分配该PCI
            if pciReuseAvailable:
                # BugFix20231117：在重新分配PCI之前，将该节点从原PCI的节点列表中删除，如果该原PCI的节点列表为空，则将其置为未被使用
                for objPciTemp in pci_pool:
                    if objPciTemp.pci == node_list[action_index].pci:
                        objPciTemp.nodeXList.remove(node_list[action_index].posX)
                        objPciTemp.nodeYList.remove(node_list[action_index].posY)
                        if (
                            len(objPciTemp.nodeXList) == 0
                            and len(objPciTemp.nodeYList) == 0
                        ):
                            objPciTemp.isUsed = False
                            debug(f"PCI {objPciTemp.pci} is unused now")
                        break

                node_list[action_index].pci = objPci.pci
                objPci.nodeXList.append(node_list[action_index].posX)
                objPci.nodeYList.append(node_list[action_index].posY)
                break

    debug(
        f"PCI {objPci.pci} is assigned to "
        f"Node ({node_list[action_index].posX}, {node_list[action_index].posY})"
    )
    next_state[action_index] = objPci.pci

    return next_state
