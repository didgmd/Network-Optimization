import numpy as np
import sys
from inspect import getframeinfo, stack

np.set_printoptions(threshold=np.inf)

# Debug mode: True, False
DEBUG_MODE = False

# PCI optimization mode
OPTIMIZATION_MODE = True

# Episodes for PCI configuration and PCI optimization
if OPTIMIZATION_MODE:
    MAX_EPISODES = 2
    MAX_TRIES = 100
else:
    MAX_EPISODES = 100
    MAX_TRIES = 1

MAX_CHOICES = 10

# Number of eNBs
numOfEnb = 19


# noinspection DuplicatedCode,PyShadowingNames
def debug(msg):
    if not DEBUG_MODE:
        return

    caller = getframeinfo(stack()[1][0])
    print("Line %d #" % caller.lineno, end=" ")

    if type(msg).__name__ == "PCI":
        print(
            "pci: %3s, isUsed: %5s, isUsedBy: %s" % (msg.pci, msg.isUsed, msg.isUsedBy)
        )

    elif type(msg).__name__ == "ENB":
        print(
            "<eNB> |  *: (%2d, %2d), pci: %3s, numOfGnb: %d, "
            "confusionList: %s, mod30CollisionList: %s, mod6CollisionList: %s"
            % (
                msg.posX,
                msg.posY,
                msg.pci,
                msg.numOfGnb,
                msg.confusionList,
                msg.mod30CollisionList,
                msg.mod6CollisionList,
            )
        )

        if msg.numOfGnb > 0:
            print("         gNBList", end="")
        for i in range(msg.numOfGnb):
            print(
                " | %2d: (%2d, %2d), pci: %3s"
                % (i, msg.gNBList[i].posX, msg.gNBList[i].posY, msg.gNBList[i].pci),
                end="",
            )
            if i == msg.numOfGnb - 1:
                print("")

        if len(msg.eNBNRList) > 0:
            print("       eNBNRList", end="")
        for i in range(len(msg.eNBNRList)):
            print(
                " | %2d: (%2d, %2d), pci: %3s"
                % (
                    i,
                    msg.eNBNRList[i].posX,
                    msg.eNBNRList[i].posY,
                    msg.eNBNRList[i].pci,
                ),
                end="",
            )
            if i == len(msg.eNBNRList) - 1:
                print("")

    elif type(msg).__name__ == "GNB":
        print(
            "<gNB> |  *: (%2d, %2d), pci: %3s, confusionList: %s, mod30CollisionList: %s, mod4CollisionList: %s"
            % (
                msg.posX,
                msg.posY,
                msg.pci,
                msg.confusionList,
                msg.mod30CollisionList,
                msg.mod4CollisionList,
            )
        )

        if len(msg.gNBNRList) > 0:
            print("       gNBNRList", end="")
        for i in range(len(msg.gNBNRList)):
            print(
                " | %2d: (%2d, %2d), pci: %3s"
                % (
                    i,
                    msg.gNBNRList[i].posX,
                    msg.gNBNRList[i].posY,
                    msg.gNBNRList[i].pci,
                ),
                end="",
            )
            if i == len(msg.gNBNRList) - 1:
                print("")

    else:
        print(msg)


class PCI:
    def __init__(self, pci):
        self.pci = pci
        self.isUsed = False
        self.isUsedBy = None


# noinspection PyShadowingNames
def oam_pci_list_init(num_of_pci: int):
    oam_pci_list = []
    for i in range(num_of_pci):
        oam_pci_list.append(PCI(i))
        debug(oam_pci_list[i])

    return oam_pci_list


class ENB:
    def __init__(self, pos_x, pos_y):
        self.nodeType = "eNB"
        self.posX = pos_x
        self.posY = pos_y
        self.pci = None
        self.gNBList = []
        self.numOfGnb = 0
        self.eNBNRList = []
        self.pciList = []
        # self.scGroup = None
        # self.confusion = False
        self.confusionList = []
        # self.mod30Collision = False
        self.mod30CollisionList = []
        # self.mod6Collision = False
        self.mod6CollisionList = []
        self.mod3CollisionList = []

    def to_list(self, gnb_pos_x, gnb_pos_y, global_gnb_list):
        self.gNBList.append(GNB(gnb_pos_x, gnb_pos_y))
        # debug(self.gNBList[self.numOfGnb])
        global_gnb_list[gnb_pos_x][gnb_pos_y] = self.gNBList[self.numOfGnb]
        self.gNBList[self.numOfGnb].headEnb = self
        self.numOfGnb += 1
        return global_gnb_list

    def add_gnb(self, num_of_gnb_per_cluster, global_gnb_list):
        while self.numOfGnb < num_of_gnb_per_cluster:
            gnb_pos_x = self.posX - 1
            gnb_pos_y = self.posY - 1
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

            gnb_pos_x = self.posX - 1
            gnb_pos_y = self.posY + 1
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

            gnb_pos_x = self.posX
            gnb_pos_y = self.posY - 2
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

            gnb_pos_x = self.posX
            gnb_pos_y = self.posY + 2
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

            gnb_pos_x = self.posX + 1
            gnb_pos_y = self.posY - 1
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

            gnb_pos_x = self.posX + 1
            gnb_pos_y = self.posY + 1
            global_gnb_list = self.to_list(gnb_pos_x, gnb_pos_y, global_gnb_list)
            if self.numOfGnb >= num_of_gnb_per_cluster:
                break

        return global_gnb_list


# noinspection PyShadowingNames,PyUnusedLocal
def enb_list_init(
    num_of_enb: int, side_length: int, dimension_x_max: int, dimension_y_max: int, i=0
):
    # x_max = 4*(n-1)+3, y_max = 6*(n-1)+5
    enb_list = np.array(
        [[None for x in range(dimension_y_max + 1)] for y in range(dimension_x_max + 1)]
    )
    # print(f"Shape of enb_list: {enb_list.shape}")

    while i < num_of_enb:
        # j=0: org_x=2, org_y=3
        # j=1: org_x=3, org_y=8
        # j=2: org_x=4, org_y=13
        for j in range(side_length):
            org_x = j + 2
            org_y = j * 5 + 3
            for k in range(side_length):
                pos_x = org_x + k * 3
                pos_y = org_y + k
                enb_list[pos_x][pos_y] = ENB(pos_x, pos_y)
                debug(enb_list[pos_x][pos_y])
                i += 1
                if i >= num_of_enb:
                    break
            if i >= num_of_enb:
                break

    return enb_list


class GNB:
    def __init__(self, pos_x, pos_y):
        self.nodeType = "gNB"
        self.posX = pos_x
        self.posY = pos_y
        self.pci = None
        self.headEnb = None
        self.gNBNRList = []
        # self.scGroup = None
        # self.confusion = False
        self.confusionList = []
        # self.mod30Collision = False
        self.mod30CollisionList = []
        # self.mod6Collision = False
        self.mod4CollisionList = []
        self.mod3CollisionList = []


# noinspection PyShadowingNames,PyUnusedLocal
def gnb_list_per_cluster_init(
    num_of_gnb_per_cluster: int, dimension_x_max: int, dimension_y_max: int
):
    # x_max = 4*(n-1)+3, y_max = 6*(n-1)+5
    global_gnb_list = np.array(
        [[None for x in range(dimension_y_max + 1)] for y in range(dimension_x_max + 1)]
    )
    # print(f"Shape of global_gnb_list: {global_gnb_list.shape}")

    for row in eNBList:
        for eNB in row:
            if eNB is None:
                continue
            eNB.add_gnb(num_of_gnb_per_cluster, global_gnb_list)
            debug(eNB)

    return global_gnb_list


# noinspection PyShadowingNames
def enb_nr_list_generation():
    for row_1 in eNBList:
        for enb in row_1:
            if enb is None:
                continue
            for row_2 in eNBList:
                for nr_enb in row_2:
                    if nr_enb is None or enb == nr_enb:
                        continue
                    if (
                        abs(enb.posX - nr_enb.posX) <= 3
                        and abs(enb.posY - nr_enb.posY) <= 5
                    ):
                        enb.eNBNRList.append(nr_enb)
            debug(enb)


# noinspection PyShadowingNames
def gnb_nr_list_generation():
    for row_1 in globalGNBList:
        for gnb in row_1:
            if gnb is None:
                continue
            for row_2 in globalGNBList:
                for nr_gnb in row_2:
                    if nr_gnb is None or gnb == nr_gnb:
                        continue
                    if (
                        abs(gnb.posX - nr_gnb.posX) <= 1
                        and abs(gnb.posY - nr_gnb.posY) <= 2
                    ):
                        gnb.gNBNRList.append(nr_gnb)
            debug(gnb)


# noinspection PyShadowingNames
def pci_self_configuration():
    for row in eNBList:
        for enb in row:
            if enb is None:
                continue
            if numOfPci >= numOfGnbPerCluster + 1:
                pci_candidates = np.random.choice(
                    oamPciList, numOfGnbPerCluster + 1, replace=False
                )
            else:
                pci_candidates = np.random.choice(
                    oamPciList, numOfGnbPerCluster + 1, replace=True
                )

            for i in range(len(pci_candidates)):
                enb.pciList.append(pci_candidates[i].pci)
                pci_candidates[i].isUsed = True
                debug(pci_candidates[i])
            debug(enb.pciList)

            enb.pci = np.random.choice(
                enb.pciList,
                1,
            )[0]
            debug(enb)
            enb.pciList.remove(enb.pci)

            for gnb in enb.gNBList:
                gnb.pci = np.random.choice(
                    enb.pciList,
                    1,
                )[0]
                debug(gnb)
                enb.pciList.remove(gnb.pci)


# noinspection PyShadowingNames
def pci_confusion_counting(pci_confusion_cnt):
    for row in eNBList:
        for enb in row:
            if enb is None:
                continue
            for nr_enb_m in enb.eNBNRList:
                for nr_enb_n in enb.eNBNRList:
                    if nr_enb_m == nr_enb_n or nr_enb_n in nr_enb_m.confusionList:
                        continue
                    if nr_enb_m.pci == nr_enb_n.pci:
                        nr_enb_m.confusionList.append(nr_enb_n)
                        nr_enb_n.confusionList.append(nr_enb_m)
                        debug(nr_enb_m)
                        debug(nr_enb_n)
                        pci_confusion_cnt["eNB"] += 1

    for row in globalGNBList:
        for gnb in row:
            if gnb is None:
                continue
            for nr_gnb_m in gnb.gNBNRList:
                for nr_gnb_n in gnb.gNBNRList:
                    if nr_gnb_m == nr_gnb_n or nr_gnb_n in nr_gnb_m.confusionList:
                        continue
                    if nr_gnb_m.pci == nr_gnb_n.pci:
                        nr_gnb_m.confusionList.append(nr_gnb_n)
                        nr_gnb_n.confusionList.append(nr_gnb_m)
                        debug(nr_gnb_m)
                        debug(nr_gnb_n)
                        pci_confusion_cnt["gNB"] += 1


# noinspection PyShadowingNames
def pci_collision_counting(pci_collision_cnt):
    for row in eNBList:
        for enb in row:
            if enb is None:
                continue
            for nr_enb in enb.eNBNRList:
                if (
                    nr_enb not in enb.mod30CollisionList
                    and enb.pci % 30 == nr_enb.pci % 30
                ):
                    enb.mod30CollisionList.append(nr_enb)
                    nr_enb.mod30CollisionList.append(enb)
                    debug(enb)
                    debug(nr_enb)
                    pci_collision_cnt["eNBMod30"] += 1
                if (
                    nr_enb not in enb.mod6CollisionList
                    and enb.pci % 6 == nr_enb.pci % 6
                ):
                    enb.mod6CollisionList.append(nr_enb)
                    nr_enb.mod6CollisionList.append(enb)
                    debug(enb)
                    debug(nr_enb)
                    pci_collision_cnt["eNBMod6"] += 1
                if (
                    nr_enb not in enb.mod3CollisionList
                    and enb.pci % 3 == nr_enb.pci % 3
                ):
                    enb.mod3CollisionList.append(nr_enb)
                    nr_enb.mod3CollisionList.append(enb)
                    debug(enb)
                    debug(nr_enb)
                    pci_collision_cnt["eNBMod3"] += 1

    for row in globalGNBList:
        for gnb in row:
            if gnb is None:
                continue
            for nr_gnb in gnb.gNBNRList:
                if (
                    nr_gnb not in gnb.mod30CollisionList
                    and gnb.pci % 30 == nr_gnb.pci % 30
                ):
                    gnb.mod30CollisionList.append(nr_gnb)
                    nr_gnb.mod30CollisionList.append(gnb)
                    debug(gnb)
                    debug(nr_gnb)
                    pci_collision_cnt["gNBMod30"] += 1
                if (
                    nr_gnb not in gnb.mod4CollisionList
                    and gnb.pci % 4 == nr_gnb.pci % 4
                ):
                    gnb.mod4CollisionList.append(nr_gnb)
                    nr_gnb.mod4CollisionList.append(gnb)
                    debug(gnb)
                    debug(nr_gnb)
                    pci_collision_cnt["gNBMod4"] += 1
                if (
                    nr_gnb not in gnb.mod3CollisionList
                    and gnb.pci % 3 == nr_gnb.pci % 3
                ):
                    gnb.mod3CollisionList.append(nr_gnb)
                    nr_gnb.mod3CollisionList.append(gnb)
                    debug(gnb)
                    debug(nr_gnb)
                    pci_collision_cnt["gNBMod3"] += 1


# noinspection PyShadowingNames
def pci_collision_resolution(node, mod: int):
    if node.nodeType == "eNB":
        for nr_enb in node.eNBNRList:
            if nr_enb.pci % mod == node.pci % mod:
                debug(node)
                debug(nr_enb)
                pciSelected = False
                debug(node.pciList)
                for pci in node.pciList:
                    if pci != node.pci and pci % mod != nr_enb.pci % mod:
                        node.pci = pci
                        pciSelected = True
                        debug(node)
                        break
                if pciSelected is False:
                    for i in range(MAX_CHOICES):
                        debug(f"Randomly choose PCI from oamPciList {i + 1}")
                        objPci = np.random.choice(oamPciList, 1)[0]
                        if (
                            objPci.isUsed is False
                            or objPci.pci % mod != nr_enb.pci % mod
                        ):
                            node.pciList.append(node.pci)
                            node.pci = objPci.pci
                            objPci.isUsed = True
                            debug(node)
                            break

    elif node.nodeType == "gNB":
        for nr_gnb in node.gNBNRList:
            if nr_gnb.pci % mod == node.pci % mod:
                debug(node)
                debug(nr_gnb)
                pciSelected = False
                debug(node.headEnb.pciList)
                for pci in node.headEnb.pciList:
                    if pci != node.pci and pci % mod != nr_gnb.pci % mod:
                        node.pci = pci
                        pciSelected = True
                        debug(node)
                        break
                if pciSelected is False:
                    for i in range(MAX_CHOICES):
                        debug(f"Randomly choose PCI from oamPciList {i + 1}")
                        objPci = np.random.choice(oamPciList, 1)[0]
                        if (
                            objPci.isUsed is False
                            or objPci.pci % mod != nr_gnb.pci % mod
                        ):
                            node.headEnb.pciList.append(node.pci)
                            node.pci = objPci.pci
                            objPci.isUsed = True
                            debug(node)
                            break
    else:
        debug("!!! Invalid nodeType !!!")


# noinspection PyShadowingNames
def pci_confusion_resolution(node_type: str, x: int, y: int):
    if node_type == "eNB":
        debug(eNBList[x][y])
        pciSelected = False
        debug(eNBList[x][y].pciList)
        for pci in eNBList[x][y].pciList:
            if pci != eNBList[x][y].pci:
                eNBList[x][y].pci = pci
                eNBList[x][y].pciList.remove(pci)
                pciSelected = True
                debug(eNBList[x][y])
                break
        if pciSelected is False:
            for i in range(MAX_CHOICES):
                debug(f"Randomly choose PCI from oamPciList {i + 1}")
                objPci = np.random.choice(oamPciList, 1)[0]
                if objPci.isUsed is False or objPci.pci != eNBList[x][y].pci:
                    eNBList[x][y].pciList.append(objPci.pci)
                    eNBList[x][y].pci = objPci.pci
                    objPci.isUsed = True
                    debug(eNBList[x][y])
                    break

    elif node_type == "gNB":
        debug(globalGNBList[x][y])
        pciSelected = False
        debug(globalGNBList[x][y].headEnb.pciList)
        for pci in globalGNBList[x][y].headEnb.pciList:
            if pci != globalGNBList[x][y].pci:
                globalGNBList[x][y].pci = pci
                globalGNBList[x][y].headEnb.pciList.remove(pci)
                pciSelected = True
                debug(globalGNBList[x][y])
                break
        if pciSelected is False:
            for i in range(MAX_CHOICES):
                debug(f"Randomly choose PCI from oamPciList {i + 1}")
                objPci = np.random.choice(oamPciList, 1)[0]
                if objPci.isUsed is False or objPci.pci != globalGNBList[x][y].pci:
                    globalGNBList[x][y].headEnb.pciList.append(objPci.pci)
                    globalGNBList[x][y].pci = objPci.pci
                    objPci.isUsed = True
                    debug(globalGNBList[x][y])
                    break

    else:
        debug("!!! Invalid nodeType !!!")


# noinspection PyShadowingNames
def sc_algorithm(node):
    # print(f"SC Algorithm for {node.nodeType} at {node.posX}, {node.posY}")
    debug(node)
    if node.nodeType == "eNB":
        x = node.posX
        y = node.posY
        # For diagonal neighbor eNBs
        # 1
        if (
            x + 1 <= dimension_x_max
            and y + 5 <= dimension_y_max
            and x - 1 >= dimension_x_min
            and y - 5 >= dimension_y_min
            and eNBList[x + 1][y + 5] is not None
            and eNBList[x - 1][y - 5] is not None
        ):
            if eNBList[x + 1][y + 5].pci == eNBList[x - 1][y - 5].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+5}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 5)
        # 2
        if (
            x + 3 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and x - 3 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and eNBList[x + 3][y + 1] is not None
            and eNBList[x - 3][y - 1] is not None
        ):
            if eNBList[x + 3][y + 1].pci == eNBList[x - 3][y - 1].pci:
                debug(f"PCI Confusion Detected at {x+3}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 3, y + 1)
        # 3
        if (
            x + 2 <= dimension_x_max
            and y - 4 >= dimension_y_min
            and x - 2 >= dimension_x_min
            and y + 4 <= dimension_y_max
            and eNBList[x + 2][y - 4] is not None
            and eNBList[x - 2][y + 4] is not None
        ):
            if eNBList[x + 2][y - 4].pci == eNBList[x - 2][y + 4].pci:
                debug(f"PCI Confusion Detected at {x+2}, {y-4}")
                pci_confusion_resolution(node.nodeType, x + 2, y - 4)

    elif node.nodeType == "gNB":
        x = node.posX
        y = node.posY
        # For diagonal neighbor gNBs
        # 1
        if (
            y + 2 <= dimension_y_max
            and y - 2 >= dimension_y_min
            and globalGNBList[x][y + 2] is not None
            and globalGNBList[x][y - 2] is not None
        ):
            if globalGNBList[x][y + 2].pci == globalGNBList[x][y - 2].pci:
                debug(f"PCI Confusion Detected at {x}, {y+2}")
                pci_confusion_resolution(node.nodeType, x, y + 2)
        # 2
        if (
            x + 1 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and x - 1 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and globalGNBList[x + 1][y + 1] is not None
            and globalGNBList[x - 1][y - 1] is not None
        ):
            if globalGNBList[x + 1][y + 1].pci == globalGNBList[x - 1][y - 1].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 1)
        # 3
        if (
            x + 1 <= dimension_x_max
            and y - 1 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y + 1 <= dimension_y_max
            and globalGNBList[x + 1][y - 1] is not None
            and globalGNBList[x - 1][y + 1] is not None
        ):
            if globalGNBList[x + 1][y - 1].pci == globalGNBList[x - 1][y + 1].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y-1}")
                pci_confusion_resolution(node.nodeType, x + 1, y - 1)

    else:
        raise Exception(f"Unknown node type {node.nodeType} in sc_algorithm()")


# noinspection PyShadowingNames
def stc_algorithm(node):
    # print(f"STC Algorithm for {node.nodeType} at {node.posX}, {node.posY}")
    debug(node)
    if node.nodeType == "eNB":
        x = node.posX
        y = node.posY
        # For triangle shaped neighbor eNBs
        # 1
        if (
            x - 2 >= dimension_x_min
            and y + 4 <= dimension_y_max
            and x + 3 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and eNBList[x - 2][y + 4] is not None
            and eNBList[x + 3][y + 1] is not None
        ):
            if eNBList[x - 2][y + 4].pci == eNBList[x + 3][y + 1].pci:
                debug(f"PCI Confusion Detected at {x-2}, {y+4}")
                pci_confusion_resolution(node.nodeType, x - 2, y + 4)
        # 2
        if (
            x + 3 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and x - 1 >= dimension_x_min
            and y - 5 >= dimension_y_min
            and eNBList[x + 3][y + 1] is not None
            and eNBList[x - 1][y - 5] is not None
        ):
            if eNBList[x + 3][y + 1].pci == eNBList[x - 1][y - 5].pci:
                debug(f"PCI Confusion Detected at {x+3}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 3, y + 1)
        # 3
        if (
            x - 1 >= dimension_x_min
            and y - 5 >= dimension_y_min
            and x - 2 >= dimension_x_min
            and y + 4 <= dimension_y_max
            and eNBList[x - 1][y - 5] is not None
            and eNBList[x - 2][y + 4] is not None
        ):
            if eNBList[x - 1][y - 5].pci == eNBList[x - 2][y + 4].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y-5}")
                pci_confusion_resolution(node.nodeType, x - 1, y - 5)
        # 4
        if (
            x - 3 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and x + 1 <= dimension_x_max
            and y + 5 <= dimension_y_max
            and eNBList[x - 3][y - 1] is not None
            and eNBList[x + 1][y + 5] is not None
        ):
            if eNBList[x - 3][y - 1].pci == eNBList[x + 1][y + 5].pci:
                debug(f"PCI Confusion Detected at {x-3}, {y-1}")
                pci_confusion_resolution(node.nodeType, x - 3, y - 1)
        # 5
        if (
            x + 1 <= dimension_x_max
            and y + 5 <= dimension_y_max
            and x + 2 <= dimension_x_max
            and y - 4 >= dimension_y_min
            and eNBList[x + 1][y + 5] is not None
            and eNBList[x + 2][y - 4] is not None
        ):
            if eNBList[x + 1][y + 5].pci == eNBList[x + 2][y - 4].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+5}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 5)
        # 6
        if (
            x + 2 <= dimension_x_max
            and y - 4 >= dimension_y_min
            and x - 3 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and eNBList[x + 2][y - 4] is not None
            and eNBList[x - 3][y - 1] is not None
        ):
            if eNBList[x + 2][y - 4].pci == eNBList[x - 3][y - 1].pci:
                debug(f"PCI Confusion Detected at {x+2}, {y-4}")
                pci_confusion_resolution(node.nodeType, x + 2, y - 4)

        # For direct neighboring eNBs
        # 1
        if (
            x - 2 >= dimension_x_min
            and y + 4 <= dimension_y_max
            and x + 1 <= dimension_x_max
            and y + 5 <= dimension_y_max
            and eNBList[x - 2][y + 4] is not None
            and eNBList[x + 1][y + 5] is not None
        ):
            if eNBList[x - 2][y + 4].pci == eNBList[x + 1][y + 5].pci:
                debug(f"PCI Confusion Detected at {x-2}, {y+4}")
                pci_confusion_resolution(node.nodeType, x - 2, y + 4)
        # 2
        if (
            x + 1 <= dimension_x_max
            and y + 5 <= dimension_y_max
            and x + 3 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and eNBList[x + 1][y + 5] is not None
            and eNBList[x + 3][y + 1] is not None
        ):
            if eNBList[x + 1][y + 5].pci == eNBList[x + 3][y + 1].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+5}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 5)
        # 3
        if (
            x + 3 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and x + 2 <= dimension_x_max
            and y - 4 >= dimension_y_min
            and eNBList[x + 3][y + 1] is not None
            and eNBList[x + 2][y - 4] is not None
        ):
            if eNBList[x + 3][y + 1].pci == eNBList[x + 2][y - 4].pci:
                debug(f"PCI Confusion Detected at {x+3}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 3, y + 1)
        # 4
        if (
            x + 2 <= dimension_x_max
            and y - 4 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y - 5 >= dimension_y_min
            and eNBList[x + 2][y - 4] is not None
            and eNBList[x - 1][y - 5] is not None
        ):
            if eNBList[x + 2][y - 4].pci == eNBList[x - 1][y - 5].pci:
                debug(f"PCI Confusion Detected at {x+2}, {y-4}")
                pci_confusion_resolution(node.nodeType, x + 2, y - 4)
        # 5
        if (
            x - 1 >= dimension_x_min
            and y - 5 >= dimension_y_min
            and x - 3 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and eNBList[x - 1][y - 5] is not None
            and eNBList[x - 3][y - 1] is not None
        ):
            if eNBList[x - 1][y - 5].pci == eNBList[x - 3][y - 1].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y-5}")
                pci_confusion_resolution(node.nodeType, x - 1, y - 5)
        # 6
        if (
            x - 3 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and x - 2 >= dimension_x_min
            and y + 4 <= dimension_y_max
            and eNBList[x - 3][y - 1] is not None
            and eNBList[x - 2][y + 4] is not None
        ):
            if eNBList[x - 3][y - 1].pci == eNBList[x - 2][y + 4].pci:
                debug(f"PCI Confusion Detected at {x-3}, {y-1}")
                pci_confusion_resolution(node.nodeType, x - 3, y - 1)

    elif node.nodeType == "gNB":
        x = node.posX
        y = node.posY
        # For triangle shaped neighboring gNBs
        # 1
        if (
            x - 1 >= dimension_x_min
            and y + 1 <= dimension_y_max
            and x + 1 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and globalGNBList[x - 1][y + 1] is not None
            and globalGNBList[x + 1][y + 1] is not None
        ):
            if globalGNBList[x - 1][y + 1].pci == globalGNBList[x + 1][y + 1].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y+1}")
                pci_confusion_resolution(node.nodeType, x - 1, y + 1)
        # 2
        if (
            x + 1 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and y - 2 >= dimension_y_min
            and globalGNBList[x + 1][y + 1] is not None
            and globalGNBList[x][y - 2] is not None
        ):
            if globalGNBList[x + 1][y + 1].pci == globalGNBList[x][y - 2].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 1)
        # 3
        if (
            y - 2 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y + 1 <= dimension_x_max
            and globalGNBList[x][y - 2] is not None
            and globalGNBList[x - 1][y + 1] is not None
        ):
            if globalGNBList[x][y - 2].pci == globalGNBList[x - 1][y + 1].pci:
                debug(f"PCI Confusion Detected at {x}, {y-2}")
                pci_confusion_resolution(node.nodeType, x, y - 2)
        # 4
        if (
            x - 1 >= dimension_x_min
            and y - 1 <= dimension_y_max
            and y + 2 <= dimension_y_max
            and globalGNBList[x - 1][y - 1] is not None
            and globalGNBList[x][y + 2] is not None
        ):
            if globalGNBList[x - 1][y - 1].pci == globalGNBList[x][y + 2].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y-1}")
                pci_confusion_resolution(node.nodeType, x - 1, y - 1)
        # 5
        if (
            y + 2 <= dimension_y_max
            and x + 1 <= dimension_x_max
            and y - 1 >= dimension_y_min
            and globalGNBList[x][y + 2] is not None
            and globalGNBList[x + 1][y - 1] is not None
        ):
            if globalGNBList[x][y + 2].pci == globalGNBList[x + 1][y - 1].pci:
                debug(f"PCI Confusion Detected at {x}, {y+2}")
                pci_confusion_resolution(node.nodeType, x, y + 2)
        # 6
        if (
            x + 1 <= dimension_x_max
            and y - 1 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and globalGNBList[x + 1][y - 1] is not None
            and globalGNBList[x - 1][y - 1] is not None
        ):
            if globalGNBList[x + 1][y - 1].pci == globalGNBList[x - 1][y - 1].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y-1}")
                pci_confusion_resolution(node.nodeType, x + 1, y - 1)

        # For direct neighboring gNBs
        # 1
        if (
            x - 1 >= dimension_x_min
            and y + 1 <= dimension_y_max
            and y + 2 <= dimension_y_max
            and globalGNBList[x - 1][y + 1] is not None
            and globalGNBList[x][y + 2] is not None
        ):
            if globalGNBList[x - 1][y + 1].pci == globalGNBList[x][y + 2].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y+1}")
                pci_confusion_resolution(node.nodeType, x - 1, y + 1)
        # 2
        if (
            y + 2 <= dimension_y_max
            and x + 1 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and globalGNBList[x][y + 2] is not None
            and globalGNBList[x + 1][y + 1] is not None
        ):
            if globalGNBList[x][y + 2].pci == globalGNBList[x + 1][y + 1].pci:
                debug(f"PCI Confusion Detected at {x}, {y+2}")
                pci_confusion_resolution(node.nodeType, x, y + 2)
        # 3
        if (
            x + 1 <= dimension_x_max
            and y + 1 <= dimension_y_max
            and x + 1 <= dimension_x_max
            and y - 1 >= dimension_y_min
            and globalGNBList[x + 1][y + 1] is not None
            and globalGNBList[x + 1][y - 1] is not None
        ):
            if globalGNBList[x + 1][y + 1].pci == globalGNBList[x + 1][y - 1].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y+1}")
                pci_confusion_resolution(node.nodeType, x + 1, y + 1)
        # 4
        if (
            x + 1 <= dimension_x_max
            and y - 1 >= dimension_y_min
            and y - 2 >= dimension_y_min
            and globalGNBList[x + 1][y - 1] is not None
            and globalGNBList[x][y - 2] is not None
        ):
            if globalGNBList[x + 1][y - 1].pci == globalGNBList[x][y - 2].pci:
                debug(f"PCI Confusion Detected at {x+1}, {y-1}")
                pci_confusion_resolution(node.nodeType, x + 1, y - 1)
        # 5
        if (
            y - 2 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and globalGNBList[x][y - 2] is not None
            and globalGNBList[x - 1][y - 1] is not None
        ):
            if globalGNBList[x][y - 2].pci == globalGNBList[x - 1][y - 1].pci:
                debug(f"PCI Confusion Detected at {x}, {y-2}")
                pci_confusion_resolution(node.nodeType, x, y - 2)
        # 6
        if (
            x - 1 >= dimension_x_min
            and y - 1 >= dimension_y_min
            and x - 1 >= dimension_x_min
            and y + 1 <= dimension_y_max
            and globalGNBList[x - 1][y - 1] is not None
            and globalGNBList[x - 1][y + 1] is not None
        ):
            if globalGNBList[x - 1][y - 1].pci == globalGNBList[x - 1][y + 1].pci:
                debug(f"PCI Confusion Detected at {x-1}, {y-1}")
                pci_confusion_resolution(node.nodeType, x - 1, y - 1)

    else:
        raise Exception(f"Unknown node type {node.nodeType} in stc_algorithm()")


# noinspection PyShadowingNames
def pci_optimization_for_enb():
    optimized = False

    # PCI Collision resolution
    if (optimizationTarget == "0" or optimizationTarget == "e30") and pciCollisionCnt[
        "eNBMod30"
    ] > 0:
        for row in eNBList:
            for enb in row:
                if enb is None:
                    continue
                pci_collision_resolution(enb, 30)
        optimized = True

    if (optimizationTarget == "0" or optimizationTarget == "e6") and pciCollisionCnt[
        "eNBMod6"
    ] > 0:
        for row in eNBList:
            for enb in row:
                if enb is None:
                    continue
                pci_collision_resolution(enb, 6)
        optimized = True

    if (optimizationTarget == "0" or optimizationTarget == "e3") and pciCollisionCnt[
        "eNBMod3"
    ] > 0:
        for row in eNBList:
            for enb in row:
                if enb is None:
                    continue
                pci_collision_resolution(enb, 3)
        optimized = True

    # PCI Confusion resolution
    if (optimizationTarget == "0" or optimizationTarget == "e0") and pciConfusionCnt[
        "eNB"
    ] > 0:
        for row in eNBList:
            for enb in row:
                if enb is None:
                    continue
                # Symmetrical Comparison algorithm
                sc_algorithm(enb)
                stc_algorithm(enb)
        optimized = True

    return optimized


# noinspection PyShadowingNames
def pci_optimization_for_gnb():
    optimized = False
    # PCI Collision resolution
    if (optimizationTarget == "0" or optimizationTarget == "g30") and pciCollisionCnt[
        "gNBMod30"
    ] > 0:
        for row in globalGNBList:
            for gnb in row:
                if gnb is None:
                    continue
                pci_collision_resolution(gnb, 30)
        optimized = True

    if (optimizationTarget == "0" or optimizationTarget == "g4") and pciCollisionCnt[
        "gNBMod4"
    ] > 0:
        for row in globalGNBList:
            for gnb in row:
                if gnb is None:
                    continue
                pci_collision_resolution(gnb, 4)
        optimized = True

    if (optimizationTarget == "0" or optimizationTarget == "g3") and pciCollisionCnt[
        "gNBMod3"
    ] > 0:
        for row in globalGNBList:
            for gnb in row:
                if gnb is None:
                    continue
                pci_collision_resolution(gnb, 3)
        optimized = True

    # PCI Confusion resolution
    if (optimizationTarget == "0" or optimizationTarget == "g0") and pciConfusionCnt[
        "gNB"
    ] > 0:
        for row in globalGNBList:
            for gnb in row:
                if gnb is None:
                    continue
                # Symmetrical Comparison algorithm
                sc_algorithm(gnb)
                stc_algorithm(gnb)
        optimized = True

    return optimized


def cnt_dict_init():
    pci_confusion_cnt = {"eNB": 0.0, "gNB": 0.0}

    pci_collision_cnt = {
        "eNBMod30": 0.0,
        "eNBMod6": 0.0,
        "eNBMod3": 0.0,
        "gNBMod30": 0.0,
        "gNBMod4": 0.0,
        "gNBMod3": 0.0,
    }

    return pci_confusion_cnt, pci_collision_cnt


# noinspection PyShadowingNames
def confusion_collision_list_reset():
    for row in eNBList:
        for enb in row:
            if enb is None:
                continue
            enb.confusionList.clear()
            enb.mod30CollisionList.clear()
            enb.mod6CollisionList.clear()
            enb.mod3CollisionList.clear()

    for row in globalGNBList:
        for gnb in row:
            if gnb is None:
                continue
            gnb.confusionList.clear()
            gnb.mod30CollisionList.clear()
            gnb.mod4CollisionList.clear()
            gnb.mod3CollisionList.clear()


def optimization_list_init(optimization_target):
    optimization_cnt_list = np.zeros(MAX_EPISODES)

    if optimization_target == "e0":
        eNB_confusion_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "g0":
        gNB_confusion_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "e30":
        eNB_mod30_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "e6":
        eNB_mod6_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "e3":
        eNB_mod3_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            gNB_mod30_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "g30":
        gNB_mod30_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod4_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "g4":
        gNB_mod4_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod3_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "g3":
        gNB_mod3_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        (
            eNB_confusion_list,
            gNB_confusion_list,
            eNB_mod30_list,
            eNB_mod6_list,
            eNB_mod3_list,
            gNB_mod30_list,
            gNB_mod4_list,
        ) = ([], [], [], [], [], [], [])
    elif optimization_target == "0":
        eNB_confusion_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        gNB_confusion_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        eNB_mod30_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        eNB_mod6_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        eNB_mod3_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        gNB_mod30_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        gNB_mod4_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
        gNB_mod3_list = np.array(np.zeros((MAX_EPISODES, MAX_TRIES + 1)))
    else:
        print("!!! Invalid optimization target !!!")
        sys.exit(1)

    return (
        optimization_cnt_list,
        eNB_confusion_list,
        gNB_confusion_list,
        eNB_mod30_list,
        eNB_mod6_list,
        eNB_mod3_list,
        gNB_mod30_list,
        gNB_mod4_list,
        gNB_mod3_list,
    )


# noinspection PyShadowingNames
def optimization_list_update(optimization_target: str, *args):
    if not args:
        raise ValueError("!!! No arguments !!!")
    else:
        if optimization_target == "e0":
            eNBConfusionList[args[0]][args[1]] = pciConfusionCnt["eNB"]
        elif optimization_target == "g0":
            gNBConfusionList[args[0]][args[1]] = pciConfusionCnt["gNB"]
        elif optimization_target == "e30":
            eNBMod30List[args[0]][args[1]] = pciCollisionCnt["eNBMod30"]
        elif optimization_target == "e6":
            eNBMod6List[args[0]][args[1]] = pciCollisionCnt["eNBMod6"]
        elif optimization_target == "e3":
            eNBMod3List[args[0]][args[1]] = pciCollisionCnt["eNBMod3"]
        elif optimization_target == "g30":
            gNBMod30List[args[0]][args[1]] = pciCollisionCnt["gNBMod30"]
        elif optimization_target == "g4":
            gNBMod4List[args[0]][args[1]] = pciCollisionCnt["gNBMod4"]
        elif optimization_target == "g3":
            gNBMod3List[args[0]][args[1]] = pciCollisionCnt["gNBMod3"]
        elif optimization_target == "0":
            eNBConfusionList[args[0]][args[1]] = pciConfusionCnt["eNB"]
            gNBConfusionList[args[0]][args[1]] = pciConfusionCnt["gNB"]
            eNBMod30List[args[0]][args[1]] = pciCollisionCnt["eNBMod30"]
            eNBMod6List[args[0]][args[1]] = pciCollisionCnt["eNBMod6"]
            eNBMod3List[args[0]][args[1]] = pciCollisionCnt["eNBMod3"]
            gNBMod30List[args[0]][args[1]] = pciCollisionCnt["gNBMod30"]
            gNBMod4List[args[0]][args[1]] = pciCollisionCnt["gNBMod4"]
            gNBMod3List[args[0]][args[1]] = pciCollisionCnt["gNBMod3"]
        else:
            raise ValueError("!!! Invalid optimization target !!!")


def optimization_finished(optimization_target: str):
    if optimization_target == "e0":
        if pciConfusionCnt["eNB"] == 0:
            print("!!! All eNB PCI confusions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! eNB PCI confusion resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "g0":
        if pciConfusionCnt["gNB"] == 0:
            print("!!! All gNB PCI confusions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! gNB PCI confusion resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "e30":
        if pciCollisionCnt["eNBMod30"] == 0:
            print("!!! All eNB PCI mod30 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! eNB PCI mod30 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "e6":
        if pciCollisionCnt["eNBMod6"] == 0:
            print("!!! All eNB PCI mod6 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! eNB PCI mod6 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "e3":
        if pciCollisionCnt["eNBMod3"] == 0:
            print("!!! All eNB PCI mod3 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! eNB PCI mod3 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "g30":
        if pciCollisionCnt["gNBMod30"] == 0:
            print("!!! All gNB PCI mod30 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! gNB PCI mod30 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "g4":
        if pciCollisionCnt["gNBMod4"] == 0:
            print("!!! All gNB PCI mod4 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! gNB PCI mod4 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "g3":
        if pciCollisionCnt["gNBMod3"] == 0:
            print("!!! All gNB PCI mod3 collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! gNB PCI mod3 collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    elif optimization_target == "0":
        if (
            pciConfusionCnt["eNB"] == 0
            and pciConfusionCnt["gNB"] == 0
            and pciCollisionCnt["eNBMod30"] == 0
            and pciCollisionCnt["eNBMod6"] == 0
            and pciCollisionCnt["gNBMod30"] == 0
            and pciCollisionCnt["gNBMod4"] == 0
        ):
            print("!!! All PCI confusions and collisions have been resolved !!!")
            optimizationCntList[i] = j + 1
            return True
        elif j == MAX_TRIES - 1:
            print("!!! PCI confusion and collision resolution failed !!!")
            optimizationCntList[i] = j + 1
            return True
    else:
        print("!!! Invalid optimization target !!!")
        sys.exit(1)

    return False


# Main function
if __name__ == "__main__":
    numOfPci = int(input("Input the number of PCI: "))

    numOfEnb = int(input("Input the number of eNB: "))
    sideLength = int(np.ceil(np.sqrt(numOfEnb)))
    debug(sideLength)
    # x_max = 4*(n-1)+3, y_max = 6*(n-1)+5
    dimension_x_max = 4 * (sideLength - 1) + 3
    dimension_x_min = 1
    dimension_y_max = 6 * (sideLength - 1) + 5
    dimension_y_min = 1
    print("dimension_x_max: ", dimension_x_max)
    print("dimension_x_min: ", dimension_x_min)
    print("dimension_y_max: ", dimension_y_max)
    print("dimension_y_min: ", dimension_y_min)

    numOfGnbPerCluster = int(input("Input the number of gNB per cluster (0 ~ 6): "))
    if numOfGnbPerCluster < 0 or numOfGnbPerCluster > 6:
        print("Invalid input! Set gNB per cluster to 6.")

    runMode = int(input("Input the run mode (0: Configuration, 1: Optimization): "))
    if runMode == 0:
        OPTIMIZATION_MODE = False
        MAX_EPISODES = 100
        MAX_TRIES = 1
    elif runMode == 1:
        OPTIMIZATION_MODE = True
        MAX_EPISODES = 100
        MAX_TRIES = 100

    if not OPTIMIZATION_MODE:
        pciConfusionCnt, pciCollisionCnt = cnt_dict_init()

        i = 0
        for i in range(MAX_EPISODES):
            oamPciList = oam_pci_list_init(numOfPci)
            eNBList = enb_list_init(
                numOfEnb, sideLength, dimension_x_max, dimension_y_max
            )
            globalGNBList = gnb_list_per_cluster_init(
                numOfGnbPerCluster, dimension_x_max, dimension_y_max
            )

            enb_nr_list_generation()
            gnb_nr_list_generation()

            pci_self_configuration()

            pci_confusion_counting(pciConfusionCnt)
            pci_collision_counting(pciCollisionCnt)
            # print(f"PCI confusion and collision after {i + 1} episodes: {pciConfusionCnt}, {pciCollisionCnt}")

        pciConfusionCnt["eNB"] /= MAX_EPISODES
        pciConfusionCnt["gNB"] /= MAX_EPISODES
        pciCollisionCnt["eNBMod30"] /= MAX_EPISODES
        pciCollisionCnt["eNBMod6"] /= MAX_EPISODES
        pciCollisionCnt["eNBMod3"] /= MAX_EPISODES
        pciCollisionCnt["gNBMod30"] /= MAX_EPISODES
        pciCollisionCnt["gNBMod4"] /= MAX_EPISODES
        pciCollisionCnt["gNBMod3"] /= MAX_EPISODES
        print(
            f"PCI confusion and collision in average with "
            f"{numOfPci} PCIs and {numOfEnb} eNBs in {i + 1} episodes: {pciConfusionCnt}, {pciCollisionCnt}"
        )

    elif OPTIMIZATION_MODE:
        optimizationTarget = input(
            "Input the optimization target (e0: eNB Confusion, g0: gNB Confusion, "
            "e30: eNB mod30 collision, e6: eNB mod6 collision, e3: eNB mod3 collision,"
            "g30: gNB mod30 collision, g4: gNB mod4 collision, g3: gNB mod3 collision,"
            "0: all): "
        )

        (
            optimizationCntList,
            eNBConfusionList,
            gNBConfusionList,
            eNBMod30List,
            eNBMod6List,
            eNBMod3List,
            gNBMod30List,
            gNBMod4List,
            gNBMod3List,
        ) = optimization_list_init(optimizationTarget)

        # Count for total optimization times
        optimizationCntTotal = 0
        i = 0

        for i in range(MAX_EPISODES):
            pciConfusionCnt, pciCollisionCnt = cnt_dict_init()

            oamPciList = oam_pci_list_init(numOfPci)
            eNBList = enb_list_init(
                numOfEnb, sideLength, dimension_x_max, dimension_y_max
            )
            globalGNBList = gnb_list_per_cluster_init(
                numOfGnbPerCluster, dimension_x_max, dimension_y_max
            )

            enb_nr_list_generation()
            gnb_nr_list_generation()

            pci_self_configuration()

            pci_confusion_counting(pciConfusionCnt)
            pci_collision_counting(pciCollisionCnt)
            print(
                f"PCI confusion and collision before optimization: {pciConfusionCnt}, {pciCollisionCnt}"
            )

            optimization_list_update(optimizationTarget, i, 0)

            for j in range(MAX_TRIES):
                r_value_enb = pci_optimization_for_enb()
                r_value_gnb = pci_optimization_for_gnb()

                # print("r_value_enb: ", r_value_enb)
                # print("r_value_gnb: ", r_value_gnb)
                if r_value_enb is False and r_value_gnb is False:
                    break
                else:
                    optimizationCntTotal += 1

                    pciConfusionCnt, pciCollisionCnt = cnt_dict_init()
                    confusion_collision_list_reset()
                    pci_confusion_counting(pciConfusionCnt)
                    pci_collision_counting(pciCollisionCnt)
                    print(
                        # f"PCI confusion and collision after optimization {j + 1}: "
                        # f"{pciConfusionCnt}, {pciCollisionCnt}"
                    )

                    optimization_list_update(optimizationTarget, i, j + 1)

                    if optimization_finished(optimizationTarget):
                        print(optimizationCntList)
                        break

            for row in eNBList:
                for enb in row:
                    if enb is None:
                        continue
                    debug(enb)
            for row in globalGNBList:
                for gnb in row:
                    if gnb is None:
                        continue
                    debug(gnb)

        print(
            f"Average optimization times for {optimizationTarget} with "
            f"{numOfPci} PCIs and {numOfEnb} eNBs in {i + 1} episodes: {optimizationCntTotal / MAX_EPISODES}"
        )
        print(f"optimizationCntList: {optimizationCntList}")
        print(f"eNBConfusionList: {eNBConfusionList}")
        print(f"gNBConfusionList: {gNBConfusionList}")
        print(f"eNBMod30List: {eNBMod30List}")
        print(f"eNBMod6List: {eNBMod6List}")
        print(f"eNBMod3List: {eNBMod3List}")
        print(f"gNBMod30List: {gNBMod30List}")
        print(f"gNBMod4List: {gNBMod4List}")
        print(f"gNBMod3List: {gNBMod3List}")

# 不同eNB数量，不同PCI数量，需要多少轮优化达到全局最优
