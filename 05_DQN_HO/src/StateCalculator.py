import math

# 自定义模块
from DebugPrint import *
from ActionChooser import *
from Parameters import *
from Normalization import *


def nn1_current_state_calculation(levy_step, user, bs_node_list):
    if levy_step == 0:
        # 计算用户到每个基站的距离，从中选择距离最近的基站
        nearest_index = 0  # 初始化索引值，即S6
        nearest_distance = 0.0
        for i in range(len(bs_node_list)):
            levy_x = user.levyPositionList[levy_step][0]
            levy_y = user.levyPositionList[levy_step][1]
            node_x = bs_node_list[i].posX
            node_y = bs_node_list[i].posY

            if (
                not np.isfinite(levy_x)
                or not np.isfinite(levy_y)
                or not np.isfinite(node_x)
                or not np.isfinite(node_y)
            ):
                debug_print(f"Invalid value encountered at step {levy_step}, node {i}")
                continue

            distance = np.sqrt((levy_x - node_x) ** 2 + (levy_y - node_y) ** 2)

            # distance = np.sqrt(bs_node_list[i].posX ** 2 + bs_node_list[i].posY ** 2)
            debug(f"distance is {distance}")
            # 初次运行时，以第一个基站离原点的距离作为最短距离
            if nearest_distance == 0.0:
                nearest_distance = distance
                nearest_index = i
            # 后续比较当前索引到的基站与最短距离的大小，如果比最短距离小，则设置为新的最短距离，相应更新最短距离基站的索引
            elif distance < nearest_distance:
                nearest_distance = distance
                nearest_index = i

        # 计算NN1当前状态
        nn1_curr_state = [
            user.levyPositionList[levy_step][0],
            user.levyPositionList[levy_step][1],
            user.levyVelocityList[levy_step][0],
            user.levyVelocityList[levy_step][1],
            0,  # 初始状态不需要切换
            nearest_index,  # 初始状态设置离原点最近基站的索引值为S6
            bs_node_list[nearest_index].posX,
            bs_node_list[nearest_index].posY,
            calculate_rsrp_by_distance(nearest_distance),
        ]

    else:
        nn1_curr_state = user.currState["NN1"].copy()
        nn1_curr_state[4] = 0  # 当前状态一直将切换标签置零

    debug(f"nn1_curr_state is {nn1_curr_state}")
    user.currState["NN1"] = nn1_curr_state.copy()  # 将NN1当前状态保存到用户对象中
    debug(f"user.currState['NN1'] is {user.currState['NN1']}")

    return nn1_curr_state


def calculate_rsrp_by_distance(distance_m):
    # rsrp = -26.34 - 30 * math.log10(distance)
    frequency_hz = 1.86e9  # in Hertz (1.86 GHz)
    tx_power_dbm = 50.0  # in dBm
    speed_of_light_m_s = 3e8  # Speed of light in meters/second

    # Calculate FSPL
    fspl_db = (
        20.0 * math.log10(distance_m)
        + 20.0 * math.log10(frequency_hz)
        + 20.0 * math.log10((4.0 * math.pi) / speed_of_light_m_s)
    )
    # fspl_db = 20.0 * math.log10(distance_m) + 20.0 * math.log10(frequency_hz) + 32.44

    # Calculate received power
    rx_power_dbm = tx_power_dbm - fspl_db

    # print(f"FSPL for {distance_m} meters: {fspl_db} dB")
    # print(f"Received power for {distance_m} meters: {rx_power_dbm} dBm")
    return rx_power_dbm


def nn2_sinr_calculation(distance):
    sinr = 10 * math.log10(35 / distance * 3)
    return sinr


def nn2_sinr_calculation2(levy_step, user, bs_node_list, curr_node_index, distance):
    # 信号
    signal_power_in_mw = 10 ** (calculate_rsrp_by_distance(distance) / 10)

    # 干扰
    interference_power_in_mw = 0.0
    for i in range(len(bs_node_list)):
        if i != curr_node_index:
            dis = np.sqrt(
                (user.levyPositionList[levy_step][0] - bs_node_list[i].posX) ** 2
                + (user.levyPositionList[levy_step][1] - bs_node_list[i].posY) ** 2
            )

            rsrp = calculate_rsrp_by_distance(dis)  # dBm
            interference_power_in_mw += 10 ** (rsrp / 10)

    # 噪声
    bandwidth_in_hz = 20e6  # 20MHz 带宽
    power_spectral_density_in_dbm_per_hz = (
        -174
    )  # dBm/Hz 功率谱密度 PSD (Power Spectral Density)
    power_spectral_density_in_mw_per_hz = 10 ** (
        power_spectral_density_in_dbm_per_hz / 10
    )
    noise_power_in_mw = power_spectral_density_in_mw_per_hz * bandwidth_in_hz

    return 10 * math.log10(
        signal_power_in_mw / (interference_power_in_mw + noise_power_in_mw)
    )


def nn2_curr_state_calculation(nn1_curr_state, levy_step, user, bs_node_list):
    # 首先确定当前连接基站的索引
    curr_node_index = nn1_curr_state[5]

    # 其次确定当前基站对象
    curr_node = bs_node_list[curr_node_index]

    # 计算用户与当前连接基站之间的距离
    distance = np.sqrt(
        (user.levyPositionList[levy_step][0] - curr_node.posX) ** 2
        + (user.levyPositionList[levy_step][1] - curr_node.posY) ** 2
    )

    # 基于距离计算RSRP
    rsrp = calculate_rsrp_by_distance(distance)

    # 基于距离计算SINR
    # sinr = nn2_sinr_calculation(distance)
    sinr = nn2_sinr_calculation2(
        levy_step, user, bs_node_list, curr_node_index, distance
    )
    debug(f"sinr is {sinr}")

    # 计算NN2当前状态：SINR，RSRP，Distance
    # nn2_curr_state = [curr_node.posX, curr_node.posY, sinr, rsrp, distance]
    nn2_curr_state = [
        curr_node.posX,
        curr_node.posY,
        sinr,
        rsrp,
        user.hoSuccessProbability,
    ]

    debug(f"nn2_curr_state is {nn2_curr_state}")
    user.currState["NN2"] = nn2_curr_state.copy()  # 将NN2当前状态保存到用户对象中
    debug(f"user.currState['NN2'] is {user.currState['NN2']}")

    return nn2_curr_state


def nn3_curr_state_calculation(levy_step, user, bs_node_list):
    # NN3的输入特征应基于当前位置确定
    nn3_curr_state = []

    # 获取用户当前位置
    pos_x = user.levyPositionList[levy_step][0]
    pos_y = user.levyPositionList[levy_step][1]

    for node in bs_node_list:
        nn3_curr_state.append(node.posX)
        nn3_curr_state.append(node.posY)
        distance = np.sqrt((pos_x - node.posX) ** 2 + (pos_y - node.posY) ** 2)
        rsrp = calculate_rsrp_by_distance(distance)
        nn3_curr_state.append(rsrp)
    debug(f"nn3_curr_state is {nn3_curr_state}")
    user.currState["NN3"] = nn3_curr_state.copy()  # 将NN3当前状态保存到用户对象中
    debug(f"user.currState['NN3'] is {user.currState['NN3']}")

    return nn3_curr_state


# 初始化储存切换失败次数的表格
lFHOALLEPOCH = []


def nn1_next_state_calculation(
    nn1_curr_state,
    nn1_action,
    levy_step,
    user,
    bs_node_list,
    nn2_epsilon,
    nn2_actions_a2,
    nn2_actions_a4,
    device,
    nn3_epsilon,
    nn3_actions,
    anchor_list,
):
    # 将切换失败的次数初始化为0
    n_fho_per_epoch = 0

    # 计算NN2当前状态
    nn2_curr_state = nn2_curr_state_calculation(
        nn1_curr_state, levy_step, user, bs_node_list
    )
    user.currState["NN2"] = nn2_curr_state.copy()  # 将NN2当前状态保存到用户对象中

    # 计算NN2的动作，NN2对应qNetList[1], nn2Action确定是否切换
    normalized_nn2_curr_state = nn2_state_normalization(nn2_curr_state)
    # print(normalized_nn2_curr_state)
    nn2_action_a2, nn2_action_a4 = nn2_choose_action(
        normalized_nn2_curr_state,
        user.qNetList[1],
        nn2_epsilon,
        nn2_actions_a2,
        nn2_actions_a4,
        device,
    )
    debug(f"nn2_action_a2 is {nn2_action_a2}, nn2_action_a4 is {nn2_action_a4}")
    user.action["NN2_A2"] = nn2_action_a2  # 将NN2动作保存到用户对象中
    user.action["NN2_A4"] = nn2_action_a4  # 将NN2动作保存到用户对象中

    # 初始化S5
    s5 = 0

    # 初始化s6为当前基站
    s6 = nn1_curr_state[5]

    # 根据NN2的动作确定当前A2和A4的阈值
    source_rsrp_threshold_a2 = nn2_action_a2
    target_rsrp_threshold_a4 = nn2_action_a4
    debug(
        f"source_rsrp_threshold_a2 is {source_rsrp_threshold_a2}, "
        f"target_rsrp_threshold_a4 is {target_rsrp_threshold_a4}"
    )

    # 初始化切换状态标志与无线链路状态标志，作为NN2奖惩的依据。为保证在未发生切换时切换状态标志为0，因此初始化的位置需在NN3运行前。
    user.handoverStatus = 0.0
    user.radioLinkStatus = 0.0

    # 根据NN1的动作与当前位置的RSRP，即nn1_curr_state[8]，
    # 当NN1决定切换，或者RSRP小于等于A2阈值时，触发NN3。注意此时触发NN3并不一定会发生切换，还要看A4
    if (nn1_action == 1) or (nn1_curr_state[8] <= source_rsrp_threshold_a2):
        # 计算NN3当前状态
        nn3_curr_state = nn3_curr_state_calculation(levy_step, user, bs_node_list)
        user.currState["NN3"] = nn3_curr_state.copy()  # 将NN3当前状态保存到用户对象中

        # 根据当前与下个基站RSRP与各自阈值的关系，最终决定是否切换。即NN3为最终决策者
        # 首先获取下个位置
        next_pos_x = user.levyPositionList[levy_step + 1][0]
        next_pos_y = user.levyPositionList[levy_step + 1][1]
        # 获取下个位置到当前基站的RSRP
        curr_node_index = user.currState["NN1"][5]
        curr_node = bs_node_list[curr_node_index]
        curr_distance = np.sqrt(
            (next_pos_x - curr_node.posX) ** 2 + (next_pos_y - curr_node.posY) ** 2
        )
        curr_rsrp = calculate_rsrp_by_distance(curr_distance)

        # NN3必须选择出与当前节点不同的基站
        nn3_diff_node_selected = False

        while not nn3_diff_node_selected:
            # 计算NN3的动作，NN3对应qNetList[2], nn3Action对应一个基站的索引
            nn3_action = nn3_choose_action(
                nn3_curr_state, user.qNetList[2], nn3_epsilon, nn3_actions, device
            )

            if nn3_action != curr_node_index:
                nn3_diff_node_selected = True
            else:
                user.nn3SameNodeSelectedNum += 1

        debug(f"nn3_action is {nn3_action}")
        user.action["NN3"] = nn3_action  # 将NN3动作保存到用户对象中

        # 获取下个位置到下个基站的RSRP
        next_node_index = nn3_action
        next_node = bs_node_list[next_node_index]
        next_distance = np.sqrt(
            (next_pos_x - next_node.posX) ** 2 + (next_pos_y - next_node.posY) ** 2
        )
        next_rsrp = calculate_rsrp_by_distance(next_distance)

        # 分别判断当前基站RSRP与阈值的关系，以及下个基站RSRP与阈值的关系
        debug(
            f"curr_rsrp with curr_node_index {curr_node_index} is {curr_rsrp}, "
            f"next_rsrp with next_node_index {next_node_index} is {next_rsrp}"
        )

        # 当确定切换时，根据目标基站的RSRP计算切换成功概率
        if (
            (curr_rsrp < next_rsrp)
            and (
                curr_rsrp < source_rsrp_threshold_a2
                or next_rsrp > target_rsrp_threshold_a4
            )
            and (curr_node_index != next_node_index)
        ) or (curr_rsrp < RADIO_LINK_NORMAL_THRESHOLD and curr_rsrp < next_rsrp):
            # 新策略 start
            # 判断用户在哪个锚点附近
            for anchor in anchor_list:
                if (
                    anchor.x - anchor.radius
                    <= user.levyPositionList[levy_step][0]
                    <= anchor.x + anchor.radius
                    and anchor.y - anchor.radius
                    <= user.levyPositionList[levy_step][1]
                    <= anchor.y + anchor.radius
                ):
                    debug(f"User is within the radius of anchor.index {anchor.index}")
                    break  # 找到锚点后跳出循环

            print(
                f"{anchor.index}[{curr_node_index}][{next_node_index}] is "
                f"{anchor.hoSuccessRateMatrix[curr_node_index][next_node_index]} in "
                f"{anchor.hoSuccessRateMatrix[curr_node_index]} "
            )
            # 记忆矩阵初始均为0.0，首次执行切换时，将其初值更改为0.5
            perform_handover = 0
            if anchor.hoSuccessRateMatrix[curr_node_index][next_node_index] == 0.0:
                perform_handover = 1
                anchor.hoSuccessRateMatrix[curr_node_index][next_node_index] = 0.5
                print(
                    f"First time, set to 0.5 -> "
                    f"{anchor.hoSuccessRateMatrix[curr_node_index]}"
                )
            elif (
                next_node_index == anchor.hoSuccessRateMatrix[curr_node_index].argmax()
            ):
                perform_handover = 1
                # print(
                #     f"{next_node_index}: {anchor.hoSuccessRateMatrix[curr_node_index][next_node_index]} "
                #     f"is max in {anchor.index}[{curr_node_index}]: {anchor.hoSuccessRateMatrix[curr_node_index]}"
                # )
            # else:
            #     perform_handover = np.random.choice(
            #         [0, 1],
            #         p=[
            #             1
            #             - anchor.hoSuccessRateMatrix[curr_node_index][next_node_index],
            #             anchor.hoSuccessRateMatrix[curr_node_index][next_node_index],
            #         ],
            #     )

            if perform_handover == 0:
                user.nn3Triggered = False
                # False 时，不切换，后面的代码都不运行
            else:
                user.nn3Triggered = True
                # 新策略 end

                # user.nn3Triggered = True
                print(
                    "user.id %2d (%5d, %5d) try to handover from %2d (%3d dBm @ %5d, %5d) to %2d (%3d dBm @ %5d, %5d) "
                    % (
                        user.id,
                        user.levyPositionList[levy_step][0],
                        user.levyPositionList[levy_step][1],
                        curr_node_index,
                        curr_rsrp,
                        curr_node.posX,
                        curr_node.posY,
                        next_node_index,
                        next_rsrp,
                        next_node.posX,
                        next_node.posY,
                    ),
                    end="",
                )
                user.handoverResult = True  # 默认切换成功

                # 根据RSRP确定切换成功概率
                if next_rsrp >= HANDOVER_SUCCESS_THRESHOLD:
                    handover_success_probability = 1.0
                elif next_rsrp <= HANDOVER_FAILURE_THRESHOLD:
                    handover_success_probability = 0.0
                else:
                    handover_success_probability = (
                        next_rsrp - HANDOVER_FAILURE_THRESHOLD
                    ) / (HANDOVER_SUCCESS_THRESHOLD - HANDOVER_FAILURE_THRESHOLD)

                # 根据SINR调整切换成功概率 20240317
                handover_success_probability = (
                    handover_success_probability + nn2_curr_state[2] * 0.01
                )

                # 根据切换成功概率确定是否切换成功
                flag_handover_success = np.random.choice(
                    [0, 1],
                    p=[1 - handover_success_probability, handover_success_probability],
                )

                # 根据切换成功与否确定最终切换结果
                if flag_handover_success == 1:
                    # 切换成功
                    # 读取本次切换用户的位置信息
                    success_x = user.levyPositionList[levy_step][0]
                    success_y = user.levyPositionList[levy_step][1]

                    # 判断用户在哪个锚点附近
                    for anchor in anchor_list:
                        if (
                            anchor.x - anchor.radius
                            <= success_x
                            <= anchor.x + anchor.radius
                            and anchor.y - anchor.radius
                            <= success_y
                            <= anchor.y + anchor.radius
                        ):
                            debug(
                                f"\nUser is within the radius of anchor.index {anchor.index}"
                            )
                            break

                    # 读取本次切换基站的ID信息
                    success_curr_node = curr_node_index
                    success_next_node = next_node_index

                    # 将此锚点附近从当前基站切换到下个基站的值乘1.05
                    anchor.hoSuccessRateMatrix[success_curr_node][
                        success_next_node
                    ] *= HO_SUCCESS_RATE_MATRIX_PLUS_MULTIPLIER
                    # 最大值不能超过1
                    if (
                        anchor.hoSuccessRateMatrix[success_curr_node][success_next_node]
                        > 1
                    ):
                        anchor.hoSuccessRateMatrix[success_curr_node][
                            success_next_node
                        ] = 1

                    s5 = 1
                    s6 = nn3_action
                    print(
                        f"Handover success with A2 {source_rsrp_threshold_a2} A4 {target_rsrp_threshold_a4} "
                        f"  Anchor {anchor.index} Source {success_curr_node} Target {success_next_node} "
                        f"Rate {anchor.hoSuccessRateMatrix[success_curr_node][success_next_node]} "
                        f"HO Success Prob. {handover_success_probability} SINR {nn2_curr_state[2]}dB "
                    )
                    user.handoverStatus = 1.0

                    # 更新Epoch切换成功计数器
                    user.hoSuccessInEpoch += 1
                else:
                    # 切换失败
                    # 读取本次切换用户的位置信息
                    failure_x = user.levyPositionList[levy_step][0]
                    failure_y = user.levyPositionList[levy_step][1]

                    # 判断用户在哪个锚点附近
                    for anchor in anchor_list:
                        if (
                            anchor.x - anchor.radius
                            <= failure_x
                            <= anchor.x + anchor.radius
                            and anchor.y - anchor.radius
                            <= failure_y
                            <= anchor.y + anchor.radius
                        ):
                            debug(
                                f"User is within the radius of anchor.index {anchor.index}"
                            )
                            break

                    # 读取本次切换基站的ID信息
                    failure_curr_node = curr_node_index
                    failure_next_node = next_node_index

                    # 将此锚点附近从当前基站切换到下个基站的值乘0.9
                    anchor.hoSuccessRateMatrix[failure_curr_node][
                        failure_next_node
                    ] *= HO_SUCCESS_RATE_MATRIX_MINUS_MULTIPLIER

                    s5 = 0
                    s6 = curr_node_index
                    print(
                        f"Handover failure with A2 {source_rsrp_threshold_a2} A4 {target_rsrp_threshold_a4}"
                        f" * Anchor {anchor.index} Source {failure_curr_node} Target {failure_next_node} "
                        f"Rate {anchor.hoSuccessRateMatrix[failure_curr_node][failure_next_node]} "
                        f"HO Success Prob. {handover_success_probability} SINR {nn2_curr_state[2]}dB "
                    )
                    user.handoverResult = False  # 切换失败
                    user.handoverStatus = (-1) * 1.0
                    n_fho_per_epoch += 1  # 切换失败次数加1

                    # 更新Epoch切换失败计数器
                    user.hoFailInEpoch += 1

                # 将切换成功概率保存到用户对象中
                user.hoSuccessProbability = handover_success_probability

                # 更新Epoch切换计数器
                user.hoInitiatedInEpoch += 1
        else:
            # 不切换
            s5 = 0
            s6 = curr_node_index
            debug(f"Keep connection to {curr_node_index}")

            # 未发生切换时，令切换成功概率为1.0
            user.hoSuccessProbability = 1.0

    # 将切换失败次数保存到列表中
    lFHOALLEPOCH.append(n_fho_per_epoch)

    # 无论是否切换，都应判断是否会发生RLF。
    # 根据RSRP确定无线链路正常概率
    if nn1_curr_state[8] >= RADIO_LINK_NORMAL_THRESHOLD:
        radio_link_normal_probability = 1.0
    elif nn1_curr_state[8] <= RADIO_LINK_FAILURE_THRESHOLD:
        radio_link_normal_probability = 0.0
    else:
        radio_link_normal_probability = (
            nn1_curr_state[8] - RADIO_LINK_FAILURE_THRESHOLD
        ) / (RADIO_LINK_NORMAL_THRESHOLD - RADIO_LINK_FAILURE_THRESHOLD)

    debug(f"radio_link_normal_probability is {radio_link_normal_probability}")

    # 根据无线链路正常概率确定是否发生RLF
    flag_radio_link_normal = np.random.choice(
        [0, 1],
        p=[1 - radio_link_normal_probability, radio_link_normal_probability],
    )

    if flag_radio_link_normal == 1:
        user.radioLinkStatus = 0.1  # 无线链路正常
        debug(f"Radio Link Normal")
    elif flag_radio_link_normal == 0:
        user.radioLinkStatus = (
            -1
        ) * 1.0  # 无线链路中断，即RLF。为了简化，此处只考虑是否发生RLF，在发生RLF时，并不进行小区重选。
        debug_print("Radio Link Failure")
        user.rlfInEpoch += 1

    # 计算NN1下个状态
    nn1_next_state = [
        user.levyPositionList[levy_step + 1][0],
        user.levyPositionList[levy_step + 1][1],
        user.levyVelocityList[levy_step + 1][0],
        user.levyVelocityList[levy_step + 1][1],
        s5,  # NN2动作即为S5
        s6,  # NN3动作即为S6
        bs_node_list[s6].posX,
        bs_node_list[s6].posY,
        calculate_rsrp_by_distance(
            np.sqrt(
                (user.levyPositionList[levy_step + 1][0] - bs_node_list[s6].posX) ** 2
                + (user.levyPositionList[levy_step + 1][1] - bs_node_list[s6].posY) ** 2
            )
        ),
    ]

    user.nextState["NN1"] = nn1_next_state.copy()  # 将NN1下个状态保存到用户对象中

    return nn1_next_state


def nn2_next_state_calculation(levy_step, user, bs_node_list):
    # 确定下个状态服务基站的索引与基站对象
    next_node_index = user.nextState["NN1"][5]
    next_node = bs_node_list[next_node_index]

    # 根据S5与S6计算下个状态的距离、RSRP、SINR
    next_distance = np.sqrt(
        (user.levyPositionList[levy_step + 1][0] - next_node.posX) ** 2
        + (user.levyPositionList[levy_step + 1][1] - next_node.posY) ** 2
    )
    next_rsrp = calculate_rsrp_by_distance(next_distance)
    next_sinr = nn2_sinr_calculation(next_distance)

    # 计算NN2下个状态
    nn2_next_state = [
        next_node.posX,
        next_node.posY,
        next_sinr,
        next_rsrp,
        # next_distance,
        user.hoSuccessProbability,
    ]
    user.nextState["NN2"] = nn2_next_state.copy()  # 将NN2下个状态保存到用户对象中

    return nn2_next_state


def nn3_next_state_calculation(user, levy_step, bs_node_list):
    # NN3的下个状态应基于下个位置确定
    nn3_next_state = []

    # 获取用户下个位置
    pos_x = user.levyPositionList[levy_step + 1][0]
    pos_y = user.levyPositionList[levy_step + 1][1]

    for node in bs_node_list:
        nn3_next_state.append(node.posX)
        nn3_next_state.append(node.posY)
        distance = np.sqrt((pos_x - node.posX) ** 2 + (pos_y - node.posY) ** 2)
        rsrp = calculate_rsrp_by_distance(distance)
        nn3_next_state.append(rsrp)
    debug(f"nn3_next_state is {nn3_next_state}")
    user.nextState["NN3"] = nn3_next_state.copy()  # 将NN3下个状态保存到用户对象中
    debug(f"user.nextState['NN3'] is {user.nextState['NN3']}")

    return nn3_next_state
