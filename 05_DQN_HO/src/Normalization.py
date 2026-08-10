from Parameters import *


def normalization_rsrp(rsrp):
    # RSRP考虑范围为-140到0，后续可考虑采用高斯白噪声参考功率
    rsrp_min = -140
    rsrp_max = 0
    corrected_rsrp = (
        rsrp_min if rsrp < rsrp_min else (rsrp_max if rsrp > rsrp_max else rsrp)
    )
    normalized_rsrp = corrected_rsrp / (rsrp_max - rsrp_min) + 1.0  # 0.0 ~ 1.0

    return normalized_rsrp


def nn1_state_normalization(state):
    # Step X, Step Y, Velocity X, Velocity Y, HO, Node Index, Node Pos X, Node Pos Y, RSRP
    normalized_step_x = state[0] / X_MAX
    normalized_step_y = state[1] / Y_MAX
    normalized_velocity_x = state[2] / X_MAX + 0.5
    normalized_velocity_y = state[3] / Y_MAX + 0.5
    normalized_ho = state[4]
    normalized_node_index = state[5] / NUM_BS
    normalized_pos_x = state[6] / X_MAX
    normalized_pos_y = state[7] / Y_MAX
    normalized_rsrp = normalization_rsrp(state[8])

    return [
        normalized_step_x,
        normalized_step_y,
        normalized_velocity_x,
        normalized_velocity_y,
        normalized_ho,
        normalized_node_index,
        normalized_pos_x,
        normalized_pos_y,
        normalized_rsrp,
    ]


def nn2_state_normalization(state):
    # node.posX, node.posY, sinr, rsrp, user.hoSuccessProbability
    normalized_pos_x = state[0] / X_MAX
    normalized_pos_y = state[1] / Y_MAX

    # SINR考虑范围为-15到0
    sinr_min = -15
    sinr_max = 5
    corrected_sinr = (
        sinr_min
        if state[2] < sinr_min
        else (sinr_max if state[2] > sinr_max else state[2])
    )
    normalized_sinr = corrected_sinr / (sinr_max - sinr_min) + 1.0  # 0.0 ~ 1.0

    # RSRP考虑范围为-140到0
    rsrp_min = -140
    rsrp_max = 0
    corrected_rsrp = (
        rsrp_min
        if state[3] < rsrp_min
        else (rsrp_max if state[3] > rsrp_max else state[3])
    )
    normalized_rsrp = corrected_rsrp / (rsrp_max - rsrp_min) + 1.0  # 0.0 ~ 1.0

    return [
        normalized_pos_x,
        normalized_pos_y,
        normalized_sinr,
        normalized_rsrp,
        state[4],
    ]


def nn3_state_normalization(state):
    normalized_state = []
    # Node Pos X, Node Pos Y, RSRP
    for i in range(NUM_BS):
        normalized_pos_x = state[3 * i] / X_MAX
        normalized_pos_y = state[3 * i + 1] / Y_MAX
        normalized_rsrp = normalization_rsrp(state[3 * i + 2])
        normalized_state.append(normalized_pos_x)
        normalized_state.append(normalized_pos_y)
        normalized_state.append(normalized_rsrp)

    return normalized_state
