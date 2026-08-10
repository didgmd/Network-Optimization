from DebugPrint import *
from Classes import DrxState
from Parameters import *


def calculate_steady_state_distribution_probability():
    # 计算稳态分布概率 Steady-state distribution probability
    # 稳态分布概率
    R = (
        (
            (1 - P67 * P76)
            * (P31 + P31 * P12 + (1 - P11) + P12 * P24 * P31 + (1 - P11) * P35)
        )
        + ((1 - P11) * P35 * P56 * (1 + P67 + P67 * P78))
        + (P35 * P56 * P67 * P78 * P81 * (1 + P12 + P12 * P24))
    )
    debug(f"R = {R}")
    pi1 = (P31 * (1 - P67 * P76) + P35 * P56 * P67 * P78 * P81) / R
    pi2 = (P31 * P12 * (1 - P67 * P76) + P35 * P56 * P67 * P78 * P81 * P12) / R
    pi3 = ((1 - P67 * P76) * (1 - P11)) / R
    pi4 = (
        P12 * P24 * P31 * (1 - P67 * P76) + P12 * P24 * P35 * P56 * P67 * P78 * P81
    ) / R
    pi5 = (P35 * (1 - P67 * P76) * (1 - P11)) / R
    pi6 = (P35 * P56 * (1 - P11)) / R
    pi7 = (P67 * (1 - P11) * P35 * P56) / R
    pi8 = (P67 * (1 - P11) * P35 * P56 * P78) / R
    debug(
        f"pi1 = {pi1}, pi2 = {pi2}, pi3 = {pi3}, pi4 = {pi4}, pi5 = {pi5}, pi6 = {pi6}, pi7 = {pi7}, pi8 = {pi8}"
    )

    param_for_reward["pi2"] = pi2
    param_for_reward["pi4"] = pi4
    param_for_reward["pi6"] = pi6

    return pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8


def calculate_power_consumption_and_delay(pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8):
    P1 = Ppc * (1 - np.exp(-LAMBDA_ipc * sum(long_term_dsCounter))) + Ps * (
        1 - np.exp(-LAMBDA_is * sum(long_term_dsCounter))
    )
    P2 = Ppc * np.exp(-LAMBDA_ipc * sum(long_term_dsCounter)) + Ps * np.exp(
        -LAMBDA_is * sum(long_term_dsCounter)
    )
    N_ds = (Ppc / (1 - np.exp(-LAMBDA_ipc * t_DS))) + (
        Ps / (1 - np.exp(-LAMBDA_is * t_DS))
    )
    N_e2 = (P2 * N_DS + P1 * N_ds) * (t_DS - t)
    N_t = (
        (t_DS - t)
        - (Ppc / LAMBDA_ipc * (1 - np.exp(-LAMBDA_ipc * (t_DS - t))))
        - (Ps / LAMBDA_is * (1 - np.exp(-LAMBDA_is * (t_DS - t))))
    )

    debug(f"N_e2 = {N_e2}, N_t = {N_t}")

    # 系统模型的功耗
    e1 = (
        ((Wp - 1) / LAMBDA_ip)
        + (Ppc * (1 - np.exp(-LAMBDA_ipc * t_I)) / LAMBDA_ipc)
        + (Ps * (1 - np.exp(-LAMBDA_is * t_I)) / LAMBDA_is)
    )

    e3 = (Ppc * (1 - np.exp(-LAMBDA_ipc * tbs)) / LAMBDA_ipc) + (
        Ps * (1 - np.exp(-LAMBDA_is * tbs)) / LAMBDA_is
    )
    # e4 = (
    #     (Ppc * (1 - np.exp(-LAMBDA_ipc * t_DL)) / LAMBDA_ipc)
    #     + (Ps * (1 - np.exp(-LAMBDA_is * t_DL)) / LAMBDA_is)
    # ) * (t_DL - t)
    e4 = t_DL - t
    e5 = trc
    e6 = tpc
    e7 = (1 - np.exp(-LAMBDA * tbs)) / LAMBDA
    e8 = tpc
    debug(f"e1, e3, e4, e5, e6, e7, e8 = {e1}, {e3}, {e4}, {e5}, {e6}, {e7}, {e8}")

    param_for_reward["N_e2"] = N_e2
    param_for_reward["e4"] = e4
    param_for_reward["e6"] = e6

    P = (
        pi1 * e1
        + pi2 * N_e2
        + pi3 * e3
        + pi4 * e4
        + pi5 * e5
        + pi6 * e6
        + pi7 * e7
        + pi8 * e8
    )  # 系统模型的总功耗
    PS = (pi2 * N_e2 + pi4 * e4 + pi6 * e6) / P  # 功耗节省因子
    debug(f"PS = {PS}, P = {P}")

    return N_t, PS


# 初始化各状态概率列表
g2List, g3List, g4List, g5List, g6List, g7List, g8List = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)


def calculate_for_t(dsCounter):
    temp = 0
    for i in range(len(state_history)):
        if i < len(state_history) - 1:
            match state_history[i]:
                case DrxState.S1:
                    temp += t_I
                case DrxState.S2:
                    temp = temp + t + dsCounter
                case DrxState.S3:
                    temp += tbs
                case DrxState.S4:
                    temp += t_DL
                case DrxState.S5:
                    temp += trc
                case DrxState.S6:
                    temp += tpc
                case DrxState.S7:
                    temp += tbs
                case DrxState.S8:
                    temp += tec
        else:
            # 对于最后一个状态，即当前到达的状态，需要做减法
            match state_history[i]:
                case DrxState.S1:
                    temp -= t_I
                case DrxState.S2:
                    temp = temp + t - dsCounter
                case DrxState.S3:
                    temp -= tbs
                case DrxState.S4:
                    temp -= t_DL
                case DrxState.S5:
                    temp -= trc
                case DrxState.S6:
                    temp -= tpc
                case DrxState.S7:
                    temp -= tbs
                case DrxState.S8:
                    temp -= tec

    return temp


def calculate_average_delay(from_state, to_state, N_t, dsCounter):
    debug(
        f"In calculate_average_delay, from_state = {from_state}, to_state = {to_state}"
    )

    overall_delay = 0

    # 系统模型的平均时延
    # 1各状态平均时延

    u2 = N_t
    u3 = (
        tbs
        - (Ppc / LAMBDA_ipc * (1 - np.exp(-LAMBDA_ipc * tbs)))
        - (Ps / LAMBDA_is * (1 - np.exp(-LAMBDA_is * tbs)))
    )
    u4 = (
        (t_DL - t)
        - (Ppc / LAMBDA_ipc * (1 - np.exp(-LAMBDA_ipc * (t_DL - t))))
        - (Ps / LAMBDA_is * (1 - np.exp(-LAMBDA_is * (t_DL - t))))
    )
    u5 = trc - (1 - np.exp(-LAMBDA * trc)) / LAMBDA
    u6 = tpc - (1 - np.exp(-LAMBDA * tpc)) / LAMBDA
    u7 = tbs - (1 - np.exp(-LAMBDA * tbs)) / LAMBDA
    u8 = tec - (1 - np.exp(-LAMBDA * tec)) / LAMBDA
    # print(u2, u3, u4, u5, u6, u7, u8)
    # 2各状态到达概率
    if from_state == DrxState.S1 and to_state == DrxState.S2:
        # S1-S2
        apc = (
            Ppc
            * (1 - np.exp(-LAMBDA_ipc * (t_DS - t)))
            * np.exp(-LAMBDA_ipc * calculate_for_t(dsCounter))
        )

        ais = (
            Ps
            * (1 - np.exp(-LAMBDA_is * (t_DS - t)))
            * np.exp(-LAMBDA_is * calculate_for_t(dsCounter))
        )
        g2 = (
            apc
            * np.exp(-LAMBDA_ipc * t_DS)
            * ((1 - np.exp(-LAMBDA_ipc * dsCounter)) / (1 - np.exp(-LAMBDA_ipc * t_DS)))
        ) + (
            ais
            * np.exp(-LAMBDA_is * t_DS)
            * ((1 - np.exp(-LAMBDA_is * dsCounter)) / (1 - np.exp(-LAMBDA_is * t_DS)))
        )
        # debug(f"g2 = {g2}, apc= {apc}, ais = {ais}")
        g2List.append(g2)

    elif to_state == DrxState.S3:
        # S3
        bpc2 = (
            Ppc
            * (1 - np.exp(-LAMBDA_ipc * tbs))
            * (np.exp(-LAMBDA_ipc * calculate_for_t(dsCounter)))
        )
        bis2 = (
            Ps
            * (1 - np.exp(-LAMBDA_is * tbs))
            * np.exp(-LAMBDA_is * calculate_for_t(dsCounter))
        )
        g3 = (bpc2 * (np.exp(-LAMBDA_ipc * tbs) / (1 - np.exp(-LAMBDA_ipc * tbs)))) + (
            bis2 * (np.exp(-LAMBDA_is * tbs) / (1 - np.exp(-LAMBDA_is * tbs)))
        )
        # debug(f"g3 = {g3}, bpc2= {apc2}, bis2 = {ais2}")
        g3List.append(g3)

    elif from_state == DrxState.S2 and to_state == DrxState.S4:
        # S2-S4
        cpc = (
            Ppc
            * (1 - np.exp(-LAMBDA_ipc * (t_DL - t)))
            * np.exp(-LAMBDA_ipc * calculate_for_t(dsCounter))
        )
        cis = (
            Ps
            * (1 - np.exp(-LAMBDA_is * (t_DL - t)))
            * np.exp(-LAMBDA_is * calculate_for_t(dsCounter))
        )
        g4 = (cpc * (np.exp(-LAMBDA_ipc * t_DL) / (1 - np.exp(-LAMBDA_ipc * t_DL)))) + (
            cis * (np.exp(-LAMBDA_is * t_DL) / (1 - np.exp(-LAMBDA_is * t_DL)))
        )
        g4List.append(g4)

    elif from_state == DrxState.S3 and to_state == DrxState.S5:
        # S3-S5
        d = (1 - np.exp(-LAMBDA * trc)) * (np.exp(-LAMBDA * calculate_for_t(dsCounter)))
        g5 = d * (np.exp(-LAMBDA * trc) / (1 - np.exp(-LAMBDA * trc)))
        # debug(f"g5 = {g5}, d = {d}")
        g5List.append(g5)

    elif to_state == DrxState.S6:
        # S6
        h = (1 - np.exp(-LAMBDA * tpc)) * (np.exp(-LAMBDA * calculate_for_t(dsCounter)))
        g6 = h * (np.exp(-LAMBDA * tpc) / (1 - np.exp(-LAMBDA * tpc)))
        # debug(f"g6 = {g6}, h= {h}")
        g6List.append(g6)

    elif from_state == DrxState.S6 and to_state == DrxState.S7:
        # S6-S7
        f = (1 - np.exp(-LAMBDA * tbs)) * (np.exp(-LAMBDA * calculate_for_t(dsCounter)))
        g7 = f * (np.exp(-LAMBDA * tbs) / (1 - np.exp(-LAMBDA * tbs)))
        # debug(f"g7 = {g7}, f = {f}")
        g7List.append(g7)

    elif from_state == DrxState.S7 and to_state == DrxState.S8:
        # S7-S8
        d = (1 - np.exp(-LAMBDA * tec)) * (np.exp(-LAMBDA * calculate_for_t(dsCounter)))
        g8 = d * (np.exp(-LAMBDA * tec) / (1 - np.exp(-LAMBDA * tec)))
        # debug(f"g8 = {g8}, d = {d}")
        g8List.append(g8)

        # S8-S1 无须计算时延

    # 总时延
    if (
        len(g2List) > 0
        and len(g3List) > 0
        and len(g4List) > 0
        and len(g5List) > 0
        and len(g6List) > 0
        and len(g7List) > 0
        and len(g8List) > 0
    ):
        t1 = (
            u2 * sum(g2List) / len(g2List)
            + u3 * sum(g3List) / len(g3List)
            + u4 * sum(g4List) / len(g4List)
            + u5 * sum(g5List) / len(g5List)
            + u6 * sum(g6List) / len(g6List)
            + u7 * sum(g7List) / len(g7List)
            + u8 * sum(g8List) / len(g8List)
        )

        debug(f"t1 = {t1}")
        overall_delay = t1

    return overall_delay


def calculate_delay_and_power_consumption_coefficient(dsCounter):
    P4 = Ppc * np.exp(-LAMBDA_ipc * dsCounter) + Ps * np.exp(-LAMBDA_is * dsCounter)

    # 计算S2的稳态分布概率
    R = (
        (
            (1 - P67 * P76)
            * (P31 + P31 * P12 + (1 - P11) + P12 * P24 * P31 + (1 - P11) * P35)
        )
        + ((1 - P11) * P35 * P56 * (1 + P67 + P67 * P78))
        + (P35 * P56 * P67 * P78 * P81 * (1 + P12 + P12 * P24))
    )
    pi2 = (P31 * P12 * (1 - P67 * P76) + P35 * P56 * P67 * P78 * P81 * P12) / R

    # 计算S2的平均时延和能耗系数
    P3 = Ppc * (1 - np.exp(-LAMBDA_ipc * dsCounter)) + Ps * (
        1 - np.exp(-LAMBDA_is * dsCounter)
    )
    P4 = Ppc * np.exp(-LAMBDA_ipc * dsCounter) + Ps * np.exp(-LAMBDA_is * dsCounter)
    N_ds = (Ppc / (1 - np.exp(-LAMBDA_ipc * t_DS))) + (
        Ps / (1 - np.exp(-LAMBDA_is * t_DS))
    )
    N_e2 = (P4 * N_DS + P3 * N_ds) * (t_DS - t)
    N_t = (
        (t_DS - t)
        - (Ppc / LAMBDA_ipc * (1 - np.exp(-LAMBDA_ipc * (t_DS - t))))
        - (Ps / LAMBDA_is * (1 - np.exp(-LAMBDA_is * (t_DS - t))))
    )

    debug(f"N_e2 = {N_e2}, N_t = {N_t}")
    power_consumption_coefficient = N_e2
    u2 = N_t

    # 计算到达概率
    apc = (
        Ppc
        * (1 - np.exp(-LAMBDA_ipc * (t_DS - t)))
        * np.exp(-LAMBDA_ipc * calculate_for_t(dsCounter))
    )

    ais = (
        Ps
        * (1 - np.exp(-LAMBDA_is * (t_DS - t)))
        * np.exp(-LAMBDA_is * calculate_for_t(dsCounter))
    )
    g2 = (
        apc
        * np.exp(-LAMBDA_ipc * t_DS)
        * ((1 - np.exp(-LAMBDA_ipc * t_DS)) / (1 - np.exp(-LAMBDA_ipc * t_DS)))
    ) + (
        ais
        * np.exp(-LAMBDA_is * t_DS)
        * ((1 - np.exp(-LAMBDA_is * t_DS)) / (1 - np.exp(-LAMBDA_is * t_DS)))
    )

    # 计算平均时延
    average_delay_s2 = u2 * g2

    return average_delay_s2, power_consumption_coefficient


def drx_state_change(curr_drx_state, dsCounter):
    next_drx_state = 0
    N_t = 0
    average_delay_s2, power_consumption_coefficient = 0, 0
    overall_delay = 0
    # 首先先查状态历史列表是否为空，如果为空，则将当前状态加入列表
    if len(state_history) == 0:
        state_history.append(curr_drx_state)

    match curr_drx_state:
        case DrxState.S1:
            next_drx_state = np.random.choice(
                np.array([DrxState.S1, DrxState.S2]), p=np.array([P11, P12])
            )

        case DrxState.S2:
            debug(f"Current DRX state is {curr_drx_state}, P23 = {P23}, P24 = {P24}")
            next_drx_state = np.random.choice(
                np.array([DrxState.S3, DrxState.S4]), p=np.array([P23, P24])
            )

        case DrxState.S3:
            debug(f"Current DRX state is {curr_drx_state}, P31 = {P31}, P35 = {P35}")
            next_drx_state = np.random.choice(
                np.array([DrxState.S1, DrxState.S5]), p=np.array([P31, P35])
            )

        case DrxState.S4:
            debug(f"Current DRX state is {curr_drx_state}")
            next_drx_state = DrxState.S3

        case DrxState.S5:
            debug(f"Current DRX state is {curr_drx_state}")
            next_drx_state = DrxState.S6

        case DrxState.S6:
            debug(f"Current DRX state is {curr_drx_state}")
            next_drx_state = DrxState.S7

        case DrxState.S7:
            debug(f"Current DRX state is {curr_drx_state}, P76 = {P76}, P78 = {P78}")
            next_drx_state = np.random.choice(
                np.array([DrxState.S6, DrxState.S8]), p=np.array([P76, P78])
            )

        case DrxState.S8:
            debug(f"Current DRX state is {curr_drx_state}")

            next_drx_state = DrxState.S1
        case _:
            debug("Unknown current DRX state!")

    # 计算稳态分布概率 Steady-state distribution probability
    (
        pi1,
        pi2,
        pi3,
        pi4,
        pi5,
        pi6,
        pi7,
        pi8,
    ) = calculate_steady_state_distribution_probability()

    # 计算系统模型功耗 Calculate system model power consumption
    N_t, power_consumption_coefficient = calculate_power_consumption_and_delay(
        pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8
    )

    # 计算总时延
    overall_delay = calculate_average_delay(
        curr_drx_state, next_drx_state, N_t, dsCounter
    )

    # 当离开S2状态时，计算相应的时延和能耗系数
    if curr_drx_state == DrxState.S2 and next_drx_state != DrxState.S2:
        (
            average_delay_s2,
            power_consumption_coefficient,
        ) = calculate_delay_and_power_consumption_coefficient(dsCounter)

    # 在计算S1到S8的时延和能耗之后，清空状态历史列表
    if curr_drx_state == DrxState.S8:
        state_history.clear()

    # 将下个状态加入状态历史列表
    state_history.append(next_drx_state)
    debug(f"State history is {state_history}")

    return (
        next_drx_state,
        average_delay_s2,
        power_consumption_coefficient,
        overall_delay,
    )


def ql_state_change(state_space, action):
    next_state = state_space[action]

    return next_state


def ql_reward_calculation(delay, pwrConsumCoeff, state):
    # 初始化奖励
    reward = 0.0
    debug(f"param_for_reward: {param_for_reward}")
    # 计算奖励  # ToDo @LianXiaoHui: 奖励需要微调
    if state == "S2":
        if len(short_term_delay) > 0:
            if delay < sum(short_term_delay) / len(short_term_delay):
                # reward += 0.001
                reward += 0.1 * (1 / LAMBDA_ip + LAMBDA) / 2
                # reward += 0
            elif delay > sum(short_term_delay) / len(short_term_delay):
                # reward -= 0.01
                reward -= (1 / LAMBDA_ip + LAMBDA) / 2
                # reward -= 0

        short_term_delay.append(delay)
    elif state == "S8":
        if len(long_term_delay) > 0:
            if delay < sum(long_term_delay) / len(long_term_delay):
                # reward += 0.001
                reward += 0.1 * (1 / LAMBDA_ip + LAMBDA) / 2
                # reward += 0
            elif delay > sum(long_term_delay) / len(long_term_delay):
                # reward -= 0.01
                reward -= (1 / LAMBDA_ip + LAMBDA) / 2
                # reward -= 0

        if len(long_term_pwrConsumCoeff) > 0:
            if pwrConsumCoeff > sum(long_term_pwrConsumCoeff) / len(
                long_term_pwrConsumCoeff
            ):
                # reward += 0.001
                reward += (
                    param_for_reward["N_e2"]
                    * param_for_reward["pi2"]
                    / (
                        param_for_reward["N_e2"] * param_for_reward["pi2"]
                        + param_for_reward["e4"] * param_for_reward["pi4"]
                        + param_for_reward["e6"] * param_for_reward["pi6"]
                    )
                )
                # reward += 0
            elif pwrConsumCoeff < sum(long_term_pwrConsumCoeff) / len(
                long_term_pwrConsumCoeff
            ):
                # reward -= 0.01
                reward -= (
                    param_for_reward["N_e2"]
                    * param_for_reward["pi2"]
                    / (
                        param_for_reward["N_e2"] * param_for_reward["pi2"]
                        + param_for_reward["e4"] * param_for_reward["pi4"]
                        + param_for_reward["e6"] * param_for_reward["pi6"]
                    )
                )
                # reward -= 0

        long_term_delay.append(delay)
        long_term_pwrConsumCoeff.append(pwrConsumCoeff)

    return reward
