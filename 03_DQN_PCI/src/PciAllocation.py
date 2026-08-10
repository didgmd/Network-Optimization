import numpy as np
import matplotlib.pyplot as plt

# 自定义模块
from DebugPrint import *


class PCI:
    def __init__(self, pci):
        self.pci = pci
        self.isUsed = False
        self.nodeXList = []
        self.nodeYList = []
        self.radiusList = []

    def __repr__(self):
        return (
            "PCI: pci = %d, isUsed = %s, nodeXList = %s, nodeYList = %s, radiusList = %s"
            % (
                self.pci,
                self.isUsed,
                self.nodeXList,
                self.nodeYList,
                self.radiusList,
            )
        )


# 网络拓扑坐标
X_MAX = 100
Y_MAX = 100

# 生成PCI池
pciPooleNB = []
for i in range(0, 9):
    pciPooleNB.append(PCI(i))

pciPoolgNB = []
for i in range(0, 17):
    pciPoolgNB.append(PCI(i))


def pci_initial_allocation(enb_list, gnb_list):
    if len(enb_list) == 0 and len(gnb_list) == 0:
        return

    # 为eNB和gNB分配PCI
    for enb in enb_list:
        while enb.pci == -1:
            objPci = np.random.choice(pciPooleNB)
            #if not objPci.isUsed:
            enb.pci = objPci.pci
            objPci.isUsed = True
            objPci.nodeXList.append(enb.posX)
            objPci.nodeYList.append(enb.posY)
            objPci.radiusList.append(enb.radius)
            break

    for gnb in gnb_list:
        while gnb.pci == -1:
            objPci = np.random.choice(pciPoolgNB)
            #if not objPci.isUsed:
            gnb.pci = objPci.pci
            objPci.isUsed = True
            objPci.nodeXList.append(gnb.posX)
            objPci.nodeYList.append(gnb.posY)
            objPci.radiusList.append(gnb.radius)
            break


def get_num_available_pci_enb():
    num = 0
    for objPci in pciPooleNB:
        if not objPci.isUsed:
            num += 1
    return num


def get_num_available_pci_gnb():
    num = 0
    for objPci in pciPoolgNB:
        if not objPci.isUsed:
            num += 1
    return num


if __name__ == "__main__":
    set_debug_mode(True)

    for pci in pciPooleNB:
        debug(pci)
    for pci in pciPoolgNB:
        debug(pci)
