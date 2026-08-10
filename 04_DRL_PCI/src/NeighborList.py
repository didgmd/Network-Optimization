import numpy as np
from DebugPrint import *


def neighbor_list_construct(enb_list, gnb_list, r_sum_multiplier):
    if len(enb_list) == 0 and len(gnb_list) == 0:
        return

    # eNB仅与eNB建立邻区关系
    for center_enb in enb_list:
        for enb in enb_list:
            if center_enb == enb:
                continue
            if (
                np.sqrt(
                    (enb.posX - center_enb.posX) ** 2
                    + (enb.posY - center_enb.posY) ** 2
                )
                < (center_enb.radius + enb.radius) * r_sum_multiplier
            ):
                center_enb.neighborList.append(enb)
        debug(f"enb_lenNeighborlist is  {len(center_enb.neighborList)}")

    # gNB仅与gNB建立邻区关系
    for center_gnb in gnb_list:
        for gnb in gnb_list:
            if center_gnb == gnb:
                continue
            if (
                np.sqrt(
                    (gnb.posX - center_gnb.posX) ** 2
                    + (gnb.posY - center_gnb.posY) ** 2
                )
                < (center_gnb.radius + gnb.radius) * r_sum_multiplier
            ):
                center_gnb.neighborList.append(gnb)
        debug(f"gnb_lenNeighborlist is  {len(center_gnb.neighborList)}")
