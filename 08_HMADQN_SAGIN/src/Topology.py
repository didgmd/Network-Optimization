# -*- coding: utf-8 -*-
import numpy as np
from Classes import SatelliteBS, MacroBS, SmallBS, UavBS, User
from Parameters import (
    MACRO_BS_1_X,
    MACRO_BS_1_Y,
    MACRO_BS_2_X,
    MACRO_BS_2_Y,
    SMALL_BS_1_X,
    SMALL_BS_1_Y,
    SMALL_BS_2_X,
    SMALL_BS_2_Y,
    SMALL_BS_3_X,
    SMALL_BS_3_Y,
    SMALL_BS_4_X,
    SMALL_BS_4_Y,
    UAV_INIT_X_1,
    UAV_INIT_Y_1,
    UAV_INIT_X_2,
    UAV_INIT_Y_2,
    UAV_INIT_HEIGHT,
    NUM_OF_USER,
    AREA_SIZE_X,
    AREA_SIZE_Y,
    MACRO_BS_HEIGHT,
    SMALL_BS_HEIGHT,
)


def define_topology():
    sat_bs = SatelliteBS(0)
    macro_bs_1 = MacroBS(1, MACRO_BS_1_X, MACRO_BS_1_Y, MACRO_BS_HEIGHT)
    macro_bs_2 = MacroBS(2, MACRO_BS_2_X, MACRO_BS_2_Y, MACRO_BS_HEIGHT)

    small_bs_1 = SmallBS(3, SMALL_BS_1_X, SMALL_BS_1_Y, SMALL_BS_HEIGHT)
    small_bs_2 = SmallBS(4, SMALL_BS_2_X, SMALL_BS_2_Y, SMALL_BS_HEIGHT)
    small_bs_3 = SmallBS(5, SMALL_BS_3_X, SMALL_BS_3_Y, SMALL_BS_HEIGHT)
    small_bs_4 = SmallBS(6, SMALL_BS_4_X, SMALL_BS_4_Y, SMALL_BS_HEIGHT)

    uav_bs_1 = UavBS(7, UAV_INIT_X_1, UAV_INIT_Y_1, UAV_INIT_HEIGHT)
    uav_bs_2 = UavBS(8, UAV_INIT_X_2, UAV_INIT_Y_2, UAV_INIT_HEIGHT)

    return (
        [sat_bs],
        [macro_bs_1, macro_bs_2],
        [small_bs_1, small_bs_2, small_bs_3, small_bs_4],
        [uav_bs_1, uav_bs_2],
    )


def define_users():
    user_list = []
    occupied_positions = set()

    num_embb = NUM_OF_USER // 3
    num_mmtc = NUM_OF_USER // 3
    num_urllc = NUM_OF_USER - num_embb - num_mmtc

    user_types = ["eMBB"] * num_embb + ["mMTC"] * num_mmtc + ["uRLLC"] * num_urllc
    np.random.shuffle(user_types)

    def generate_unique_position():
        max_attempts = 1000
        for _ in range(max_attempts):
            x = round(np.random.uniform(0, AREA_SIZE_X), 2)
            y = round(np.random.uniform(0, AREA_SIZE_Y), 2)
            if (x, y) not in occupied_positions:
                occupied_positions.add((x, y))
                return x, y
        grid_size = int(np.ceil(np.sqrt(NUM_OF_USER)))
        grid_x = (len(occupied_positions) % grid_size) * (AREA_SIZE_X / grid_size)
        grid_y = (len(occupied_positions) // grid_size) * (AREA_SIZE_Y / grid_size)
        return grid_x, grid_y

    for i in range(NUM_OF_USER):
        x, y = generate_unique_position()
        user_type = user_types[i]
        user_list.append(User(i, x, y, user_type))

    return user_list
