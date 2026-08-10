import numpy as np
import random
import matplotlib.pyplot as plt
from Parameters import *

# 自定义模块
from DebugPrint import *


class PCI:
    def __init__(self, pci):
        self.pci = pci
        self.isUsed = False
        self.nodeXList = []
        self.nodeYList = []
        self.radiusList = []
        self.hashmap = []

    def __repr__(self):
        return (
            "PCI: pci = %d, isUsed = %s, nodeXList = %s, nodeYList = %s, radiusList = %s, hashmap = %s"
            % (
                self.pci,
                self.isUsed,
                self.nodeXList,
                self.nodeYList,
                self.radiusList,
                self.hashmap,
            )
        )

pci_pool_enb, pci_pool_gnb = [], []

def initialize_pci_pool(enb_pci_min, enb_pci_max, gnb_pci_min, gnb_pci_max):
    debug("PCI pool initialization")
    #将PCI池中的所有PCI属性重置
    for objPci in pci_pool_enb:
        objPci.isUsed = False
        objPci.nodeXList.clear()
        objPci.nodeYList.clear()
        objPci.radiusList.clear()
    for objPci in pci_pool_gnb:
        objPci.isUsed = False
        objPci.nodeXList.clear()
        objPci.nodeYList.clear()
        objPci.radiusList.clear()

    for i in range(enb_pci_min, enb_pci_max + 1):
        pci_pool_enb.append(PCI(i))

    for j in range(gnb_pci_min, gnb_pci_max + 1):
        pci_pool_gnb.append(PCI(j))

    return pci_pool_enb, pci_pool_gnb

random.seed(0)  # 使用固定的种子来初始化随机数生成器

def hashmap_function(cgi, pci_pool, pci_min, pci_max, loop_iteration):
    i = (cgi + random.randint(0, loop_iteration)) % (pci_max - pci_min + 1) + pci_min
    for objPci in pci_pool:
        if objPci.pci == i:
            objPci.hashmap.append(cgi)
            return objPci


def hashmap_based_pci_initial_allocation(
    pci_pool_enb,
    pci_pool_gnb,
    enb_list,
    gnb_list,
    enb_pci_min,
    enb_pci_max,
    gnb_pci_min,
    gnb_pci_max,
    iteration,
):
    if len(enb_list) == 0 and len(gnb_list) == 0:
        return

    # 为eNB和gNB分配PCI
    for enb in enb_list:
        while enb.pci == -1:
            obj_pci = hashmap_function(enb.cgi, pci_pool_enb, enb_pci_min, enb_pci_max,iteration)
            enb.pci = obj_pci.pci
            obj_pci.isUsed = True
            obj_pci.nodeXList.append(enb.posX)
            obj_pci.nodeYList.append(enb.posY)
            obj_pci.radiusList.append(enb.radius)
            break

    for gnb in gnb_list:
        while gnb.pci == -1:
            obj_pci = hashmap_function(gnb.cgi, pci_pool_gnb, gnb_pci_min, gnb_pci_max,iteration)
            gnb.pci = obj_pci.pci
            obj_pci.isUsed = True
            obj_pci.nodeXList.append(gnb.posX)
            obj_pci.nodeYList.append(gnb.posY)
            obj_pci.radiusList.append(gnb.radius)
            break


def random_pci_initial_allocation(enb_list, gnb_list):
    if len(enb_list) == 0 and len(gnb_list) == 0:
        return

    # 为eNB和gNB分配PCI
    for enb in enb_list:
        while enb.pci == -1:
            objPci = np.random.choice(pci_pool_enb)
            #debug(f"objPci: {objPci}")
            # if not objPci.isUsed:
            enb.pci = objPci.pci
            objPci.isUsed = True
            objPci.nodeXList.append(enb.posX)
            objPci.nodeYList.append(enb.posY)
            objPci.radiusList.append(enb.radius)
            break

    for gnb in gnb_list:
        while gnb.pci == -1:
            objPci = np.random.choice(pci_pool_gnb)
            # if not objPci.isUsed:
            gnb.pci = objPci.pci
            objPci.isUsed = True
            objPci.nodeXList.append(gnb.posX)
            objPci.nodeYList.append(gnb.posY)
            objPci.radiusList.append(gnb.radius)
            break


def get_num_available_pci_enb():
    num = 0
    for objPci in pci_pool_enb:
        if not objPci.isUsed:
            num += 1
    return num


def get_num_available_pci_gnb():
    num = 0
    for objPci in pci_pool_gnb:
        if not objPci.isUsed:
            num += 1
    return num


if __name__ == "__main__":
    set_debug_mode(True)

    for pci in pci_pool_enb:
        debug(pci)
    for pci in pci_pool_gnb:
        debug(pci)
