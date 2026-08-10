import numpy as np
from DebugPrint import *
from StateCalculator import calculate_rsrp_by_distance
from Parameters import *


def nn1_reward_calculation(
    user, levy_step, nn1_curr_state, bs_node_list, nn1_next_state
):
    # 初始化奖励
    reward = 0.0

    # 确定源基站对象
    org_node = bs_node_list[nn1_curr_state[5]]

    # 计算用户处于当前位置时与源基站之间的距离
    org_node_curr_distance = np.sqrt(
        (user.levyPositionList[levy_step][0] - org_node.posX) ** 2
        + (user.levyPositionList[levy_step][1] - org_node.posY) ** 2
    )

    # 根据距离计算当前位置到源基站的RSRP
    org_node_curr_rsrp = calculate_rsrp_by_distance(org_node_curr_distance)
    debug(f"org_node_curr_rsrp is {org_node_curr_rsrp}")

    # 计算用户处于下个位置时与源基站之间的距离
    org_node_next_distance = np.sqrt(
        (user.levyPositionList[levy_step + 1][0] - org_node.posX) ** 2
        + (user.levyPositionList[levy_step + 1][1] - org_node.posY) ** 2
    )

    # 根据距离计算下个位置到源基站的RSRP
    org_node_next_rsrp = calculate_rsrp_by_distance(org_node_next_distance)
    debug(f"org_node_next_rsrp is {org_node_next_rsrp}")

    # 根据当前位置到源基站的RSRP与下个位置到源基站的RSRP计算RSRP的变化，从而判断离源基站变远还是变近
    org_node_rsrp_diff = org_node_curr_rsrp - org_node_next_rsrp  # 正数：变远；负数：变近
    debug(f"org_node_rsrp_diff is {org_node_rsrp_diff}")

    # 由于NN1的输入包括了当前位置与移动方向，因此有一定能力判断下个位置到当前连接基站的RSRP变化，
    # 进而也可以判定与A2阈值的关系，所以可将奖惩与其联系起来。
    # 根据下个状态的S5是否为0，判断是否发生切换：0为不切换，1为切换
    # 最终决定（下个状态的S5）为不切换（即并不涉及下个连接基站/目标基站）时，奖励/惩罚的计算
    if nn1_next_state[4] == 0:
        # 下个位置到源基站的RSRP大于RSRP阈值（A2，无需切换）
        if org_node_next_rsrp > SOURCE_RSRP_THRESHOLD_A2:
            # （奖励）NN1决定不切换
            if user.action["NN1"] == 0:
                reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
            # （惩罚）NN1决定切换
            elif user.action["NN1"] == 1:
                reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
        # 下个位置到源基站的RSRP等于RSRP阈值（A2）
        elif org_node_next_rsrp == SOURCE_RSRP_THRESHOLD_A2:
            # 下个位置到源基站的RSRP大于当前位置到源基站的RSRP（变近）
            if org_node_rsrp_diff < 0:
                # （奖励）NN1决定不切换
                if user.action["NN1"] == 0:
                    reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
                # （惩罚）NN1决定切换
                elif user.action["NN1"] == 1:
                    reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
            # 下个位置到源基站的RSRP等于当前位置到源基站的RSRP
            elif org_node_rsrp_diff == 0:
                # （无奖惩）NN1决定切换/不切换
                reward = 0.0
            # 下个位置到源基站的RSRP小于当前位置到源基站的RSRP（变远）
            elif org_node_rsrp_diff > 0:
                # （奖励）NN1决定切换
                if user.action["NN1"] == 1:
                    reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
                # （惩罚）NN1决定不切换
                elif user.action["NN1"] == 0:
                    reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
        # 下个位置到源基站的RSRP小于RSRP阈值（A2，需要切换）
        elif org_node_next_rsrp < SOURCE_RSRP_THRESHOLD_A2:
            # （奖励）NN1决定切换
            if user.action["NN1"] == 1:
                reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
            # （惩罚）NN1决定不切换
            elif user.action["NN1"] == 0:
                reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
    # 最终决定（下个状态的S5）为切换（即与下个连接基站/目标基站有关，不管离源基站变远还是变近，
    # 只要到源基站的RSRP大于A2阈值，则无需切换；需要切换时，则判断与到目标基站RSRP的关系）时，奖励/惩罚的计算
    elif nn1_next_state[4] == 1:
        # 确定目标基站对象
        target_node = bs_node_list[nn1_next_state[5]]

        # 计算用户处于下个位置时与目标基站之间的距离
        target_node_distance = np.sqrt(
            (user.levyPositionList[levy_step + 1][0] - target_node.posX) ** 2
            + (user.levyPositionList[levy_step + 1][1] - target_node.posY) ** 2
        )

        # 根据距离计算下个位置到目标基站的RSRP
        target_node_rsrp = calculate_rsrp_by_distance(target_node_distance)
        debug(f"target_node_rsrp is {target_node_rsrp}")

        # 下个位置到源基站的RSRP大于RSRP阈值（A2，无需切换）
        if org_node_next_rsrp > SOURCE_RSRP_THRESHOLD_A2:
            # （奖励）NN1决定不切换
            if user.action["NN1"] == 0:
                reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
            # （惩罚）NN1决定切换
            elif user.action["NN1"] == 1:
                reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
        # 下个位置到源基站的RSRP小于等于RSRP阈值（A2，如果目标基站更好，则需要切换）
        elif org_node_next_rsrp <= SOURCE_RSRP_THRESHOLD_A2:
            # 下个位置到目标基站的RSRP大于下个位置到源基站的RSRP（应该切换）
            if target_node_rsrp > org_node_next_rsrp:
                # （奖励）NN1决定切换
                if user.action["NN1"] == 1:
                    reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
                # （惩罚）NN1决定不切换
                elif user.action["NN1"] == 0:
                    reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
            # 下个位置到目标基站的RSRP等于下个位置到源基站的RSRP
            elif target_node_rsrp == org_node_next_rsrp:
                # （无奖惩）NN1决定切换/不切换
                reward = 0.0
            # 下个位置到目标基站的RSRP小于下个位置到源基站的RSRP（不应该切换）
            elif target_node_rsrp < org_node_next_rsrp:
                # （奖励）NN1决定不切换
                if user.action["NN1"] == 0:
                    reward = abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR
                # （惩罚）NN1决定切换
                elif user.action["NN1"] == 1:
                    reward = (-1) * abs(org_node_rsrp_diff) * REWARD_SCALE_FACTOR

    return reward


def nn2_reward_calculation(user):
    # 初始化奖励
    reward = 0.0

    # 根据切换状态给予NN2奖惩
    if user.handoverStatus >= 0:
        reward_a4 = reward + user.handoverStatus * REWARD_SCALE_FACTOR
    else:
        reward_a4 = reward + user.handoverStatus * REWARD_SCALE_FACTOR * 10
        debug_print(f"Penalty for NN2 due to Handover Failure is {reward_a4}")


    # 根据无线链路状态给予NN2奖惩
    if user.radioLinkStatus >= 0:
        reward_a2 = reward + user.radioLinkStatus * REWARD_SCALE_FACTOR
    else:
        reward_a2 = reward + user.radioLinkStatus * REWARD_SCALE_FACTOR * 10
        debug_print(f"Penalty for NN2 due to Radio Link Failure is {reward_a2}")

    # 以下为原思路
    """
    # 获取当前状态RSRP
    curr_rsrp = user.currState["NN2"][3]

    # 获取下个状态RSRP
    next_rsrp = user.nextState["NN2"][3]

    # 计算当前状态RSRP与下个状态RSRP之差
    rsrp_diff = curr_rsrp - next_rsrp
    debug(f"rsrp_diff is {rsrp_diff}")

    # 由于NN2的输入仅包括当前连接基站的坐标和与当前位置的距离等信息，
    # 因此可根据当前连接基站到当前位置的RSRP与A2阈值的关系进行奖惩。
    # 当前连接基站到当前位置的RSRP大于RSRP阈值（A2，无需切换）
    if curr_rsrp > SOURCE_RSRP_THRESHOLD_A2:
        # （奖励）NN2决定不切换
        if user.action["NN2"] == 0:
            reward = abs(rsrp_diff) * REWARD_SCALE_FACTOR
        # （惩罚）NN2决定切换
        elif user.action["NN2"] == 1:
            reward = (-1) * abs(rsrp_diff) * REWARD_SCALE_FACTOR
    # 当前连接基站到当前位置的RSRP等于RSRP阈值（A2）
    elif curr_rsrp == SOURCE_RSRP_THRESHOLD_A2:
        # （无奖惩）NN2决定切换/不切换
        reward = 0.0
    # 当前连接基站到当前位置的RSRP小于RSRP阈值（A2，需要切换）
    elif curr_rsrp < SOURCE_RSRP_THRESHOLD_A2:
        # （奖励）NN2决定切换
        if user.action["NN2"] == 1:
            reward = abs(rsrp_diff) * REWARD_SCALE_FACTOR
        # （惩罚）NN2决定不切换
        elif user.action["NN2"] == 0:
            reward = (-1) * abs(rsrp_diff) * REWARD_SCALE_FACTOR
    """

    return reward_a2, reward_a4


def nn3_reward_calculation(user, levy_step):
    # 初始化奖励
    reward = 0.0

    # 不应根据选择的基站是否为离当前位置最近基站进行奖惩，因为其目的是选择适用于下个位置的基站；
    # 根据选择的基站是否与源基站相同、切换成功与否、乒乓切换发生与否进行奖惩。
    # 根据选择的目标基站是否与源基站相同给予惩罚或奖励
    flag_same_node_optimization = True
    if flag_same_node_optimization:
        reward_same_node = 0.0
        # （奖励）选择的目标基站与源基站不同
        if user.nn3SameNodeSelectedNum == 0:
            reward_same_node = 1.0 * REWARD_SCALE_FACTOR
        # （惩罚）选择的目标基站与源基站相同
        elif user.nn3SameNodeSelectedNum > 0:
            reward_same_node = (-1) * user.nn3SameNodeSelectedNum * REWARD_SCALE_FACTOR
            debug_print(
                f"Penalty for NN3 due to same node selected is {reward_same_node}"
            )

        reward += reward_same_node

    # NN2的部分奖惩由切换是否成功决定
    # 根据切换是否成功给予惩罚或奖励
    flag_handover_success_optimization = True
    if flag_handover_success_optimization:
        reward_handover_success = 0.0
        # （奖励）切换成功
        if user.handoverResult:
            reward_handover_success = 1.0 * REWARD_SCALE_FACTOR
        # （惩罚）切换失败
        elif not user.handoverResult:
            reward_handover_success = (-1) * 1.0 * REWARD_SCALE_FACTOR * 10
            debug_print(
                f"Penalty for NN3 due to handover failure is {reward_handover_success}"
            )

        reward += reward_handover_success

    # 根据是否发生乒乓切换给予奖励或惩罚
    flag_ping_pong_handover_optimization = True
    if flag_ping_pong_handover_optimization:
        reward_ping_pong_handover = 0.0
        if levy_step >= 2:
            # （惩罚）发生乒乓切换
            if (user.s6EpochList[levy_step - 2] == user.s6EpochList[levy_step]) and (
                user.s6EpochList[levy_step - 1] != user.s6EpochList[levy_step]
            ):
                reward_ping_pong_handover = (-1) * 1.0 * REWARD_SCALE_FACTOR
                debug_print(
                    f"Penalty for NN3 due to Ping-Pong handover is {reward_ping_pong_handover}"
                )

            # （奖励）乒乓切换未发生。由于乒乓切换极少发生，因此移除奖励
            # elif user.s6EpochList[levy_step - 2] != user.s6EpochList[levy_step]:
            #     reward_ping_pong_handover = 1.0 * REWARD_SCALE_FACTOR

        reward += reward_ping_pong_handover

    return reward
