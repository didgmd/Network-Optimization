import math

from rl_training_config import (
    OUTCOME_REWARD_PROFILE,
    OUTCOME_REWARD_PROFILE_V1,
    OUTCOME_REWARD_PROFILE_V2_SMALL_DECIMAL,
    OUTCOME_REWARD_CLIP_MAX,
    OUTCOME_REWARD_CLIP_MIN,
    OUTCOME_REWARD_EXEC_WEIGHT,
    OUTCOME_REWARD_PPHO_WEIGHT,
    OUTCOME_REWARD_RSRP_SCALE_DB,
    OUTCOME_REWARD_RSRP_WEIGHT,
    OUTCOME_REWARD_SINR_WEIGHT,
    OUTCOME_V2_CLIP_MAX,
    OUTCOME_V2_CLIP_MIN,
    OUTCOME_V2_EXEC_GOOD_REWARD,
    OUTCOME_V2_EXEC_NEG_GAIN_PENALTY,
    OUTCOME_V2_EXEC_RLF_PENALTY,
    OUTCOME_V2_PPHO_PENALTY,
    OUTCOME_V2_RSRP_SCALE,
    OUTCOME_V2_SINR_OUTAGE_PENALTY,
)


def outcome_based_reward(
    prev_rsrp,
    post_rsrp,
    handover_executed=False,
    ppho_event=False,
    post_sinr=None,
    rlf_sinr_threshold=-5.0,
    reward_profile=None,
    return_components=False,
):
    rsrp_gain = float(post_rsrp) - float(prev_rsrp)
    profile = reward_profile or OUTCOME_REWARD_PROFILE
    post_sinr_value = None if post_sinr is None else float(post_sinr)
    is_rlf = (
        post_sinr_value is not None
        and post_sinr_value < float(rlf_sinr_threshold)
    )

    if profile == OUTCOME_REWARD_PROFILE_V2_SMALL_DECIMAL:
        r_rsrp = (
            OUTCOME_V2_RSRP_SCALE * math.tanh(rsrp_gain / OUTCOME_REWARD_RSRP_SCALE_DB)
            if handover_executed
            else 0.0
        )
        r_sinr = OUTCOME_V2_SINR_OUTAGE_PENALTY if is_rlf else 0.0
        if not handover_executed:
            r_exec = 0.0
        elif is_rlf:
            r_exec = OUTCOME_V2_EXEC_RLF_PENALTY
        elif rsrp_gain >= 0:
            r_exec = OUTCOME_V2_EXEC_GOOD_REWARD
        else:
            r_exec = OUTCOME_V2_EXEC_NEG_GAIN_PENALTY
        r_ppho = OUTCOME_V2_PPHO_PENALTY if ppho_event else 0.0
        clip_min = OUTCOME_V2_CLIP_MIN
        clip_max = OUTCOME_V2_CLIP_MAX
    elif profile == OUTCOME_REWARD_PROFILE_V1:
        base_r_rsrp = (
            math.tanh(rsrp_gain / OUTCOME_REWARD_RSRP_SCALE_DB)
            if handover_executed
            else 0.0
        )
        if post_sinr_value is None:
            base_r_sinr = 0.0
        else:
            base_r_sinr = 1.0 if post_sinr_value >= float(rlf_sinr_threshold) else -1.0
        if not handover_executed:
            base_r_exec = -0.1
        elif is_rlf:
            base_r_exec = -0.3
        elif rsrp_gain >= 0:
            base_r_exec = 0.2
        else:
            base_r_exec = -0.2
        base_r_ppho = -1.0 if ppho_event else 0.0
        r_rsrp = OUTCOME_REWARD_RSRP_WEIGHT * base_r_rsrp
        r_sinr = OUTCOME_REWARD_SINR_WEIGHT * base_r_sinr
        r_exec = OUTCOME_REWARD_EXEC_WEIGHT * base_r_exec
        r_ppho = OUTCOME_REWARD_PPHO_WEIGHT * base_r_ppho
        clip_min = OUTCOME_REWARD_CLIP_MIN
        clip_max = OUTCOME_REWARD_CLIP_MAX
    else:
        raise ValueError(f"Unknown outcome reward profile: {profile}")

    reward_unclipped = r_rsrp + r_sinr + r_exec + r_ppho
    reward_raw = max(clip_min, min(clip_max, reward_unclipped))
    reward = reward_raw
    if return_components:
        return {
            "profile": profile,
            "rsrp": r_rsrp,
            "sinr": r_sinr,
            "execution": r_exec,
            "ppho": r_ppho,
            "raw": reward_raw,
            "reward": reward,
            "handover_executed": int(bool(handover_executed)),
            "ppho_event": int(bool(ppho_event)),
            "rsrp_gain_db": rsrp_gain,
        }
    return reward


# Legacy expert-guided reward. Retained for historical variants only; primary
# CG-as-reference variants use outcome_based_reward and do not read labels.
def calculate_reward(
    prev_rsrp,
    post_rsrp,
    predicted_label=None,
    previous_action=None,
    action=None,
    a3_satisfied=False,
    handover_executed=False,
    flag_reward_rsrp=False,  # 是否启用奖励细节
):
    reward = 0.0
    """
    奖励函数：预测标签值、动作合理性、切换结果、 A3
    :param prev_rsrp: 切换前RSRP
    :param post_rsrp: 切换后RSRP
    :param predicted_label: 预测标签（0=过早，1=过晚，2=理想）
    :param previous_action: 上一次动作 (TTT, HOM)
    :param action: 当前动作 (TTT, HOM)
    :param a3_satisfied: 当前动作是否满足 A3 条件
    :param handover_executed: 是否真实发生切换
    :return: 奖励值 ∈ [-1.0, 1.0]
    :flag_reward_rsrp: 是否启用奖励细节
    """

    if action is None or previous_action is None or predicted_label is None:
        return 0.0
    prev_ttt, prev_hom = previous_action
    curr_ttt, curr_hom = action
    ttt_diff = curr_ttt - prev_ttt
    hom_diff = curr_hom - prev_hom

    if predicted_label == 0:
        if ttt_diff > 0 and hom_diff > 0:  # 最高优先级
            reward += 0.25
        elif ttt_diff > 0 and hom_diff == 0:  # 第二优先级
            reward += 0.20
        elif ttt_diff == 0 and hom_diff > 0 and not a3_satisfied:  # 第三优先级
            reward += 0.15
        elif ttt_diff > 0 and hom_diff < 0:  # 第四优先级
            reward += 0.1
        else:  # 其他情况均为不合理动作
            reward -= 0.1

    elif predicted_label == 1:
        if ttt_diff < 0 and hom_diff < 0:  # 最高优先级
            reward += 0.25
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif ttt_diff < 0 and hom_diff == 0:  # 第二优先级
            reward += 0.20
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif ttt_diff == 0 and hom_diff < 0:  # 第三优先级
            reward += 0.15
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif (
            ttt_diff < 0 and hom_diff > 0 and a3_satisfied
        ):  # 第四优先级，适合轻度过晚情况，较保守
            reward += 0.1
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif (
            ttt_diff == 0 and hom_diff > 0 and a3_satisfied
        ):  # 第五优先级，非常保守的一种策略
            reward += 0.05
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        else:  # 其他情况均为不合理动作
            reward -= 0.1

    elif predicted_label == 2:
        if ttt_diff == 0 and hom_diff == 0:  # 最高优先级
            reward += 0.25
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif (
            -200 <= ttt_diff <= 200 and hom_diff > 0 and a3_satisfied
        ):  # 第二优先级，适合探索调参边界
            reward += 0.20
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        elif (
            -200 <= ttt_diff <= 200 and hom_diff < 0 and a3_satisfied
        ):  # 第三优先级，适合探索调参边界
            reward += 0.15
            if flag_reward_rsrp:
                if handover_executed:
                    if post_rsrp > prev_rsrp:
                        reward += 0.05
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05
        else:
            # 其他情况均为不合理动作
            reward -= 0.1

    return max(min(reward, 1.0), -1.0)


# V1
# def calculate_reward(user, dl_predicted_label, prev_rsrp, post_rsrp):
#     """
#     参数说明：
#     user: 当前User对象，需具有切换结果信息（如handover_result）
#     dl_predicted_label: CatBoost模型预测的标签（0:过早，1:过晚，2:理想）
#     prev_rsrp: 切换前的source_bs_rsrp
#     post_rsrp: 切换后的source_bs_rsrp
#     """
#     reward = 0.0
#
#     # 奖励项1：切换类型是否理想
#     if dl_predicted_label == 2:  # 理想切换
#         reward += 1.0
#     else:
#         reward -= 0.5  # 惩罚过早/过晚
#
#     # 奖励项2：RSRP是否提升
#     if post_rsrp >= prev_rsrp:
#         reward += 0.5
#     else:
#         reward -= 0.5
#
#     # 奖励项3：是否成功切换
#     if hasattr(user, "handover_result"):
#         if user.handover_result:  # 成功切换
#             reward += 1.0
#         else:  # 失败切换
#             reward -= 10.0
#
#     return reward

# V2
# def calculate_reward(prev_rsrp, post_rsrp):  # 20250502晚增
#     reward = 0.0
#
# # 1. 切换后 RSRP 变化
# if post_rsrp >= prev_rsrp:
#     reward += 0.5
# else:
#     reward -= 0.5
#
#     return reward
#

# V3
# from Parameters import (INIT_TTT, INIT_HOM, )
#
#
# def calculate_reward(prev_rsrp, post_rsrp, predicted_label=None, action=None):
#     """
#     :param prev_rsrp: 切换前RSRP
#     :param post_rsrp: 切换后RSRP
#     :param predicted_label: 深度学习预测标签（0=过早，1=过晚，2=理想）
#     :param action: 当前动作 (TTT, HOM)
#     :return: 综合奖励值
#     """
#     reward = 0.0
#
#     # 1. 切换后RSRP变化
#     if post_rsrp >= prev_rsrp:
#         reward += 0.5
#     else:
#         reward -= 0.5
#
#     # 2. 预测标签和动作匹配奖惩逻辑
#     if predicted_label is not None and action is not None:
#         ttt, hom = action
#         ttt_diff = ttt - INIT_TTT  # 与初始值对比
#         hom_diff = hom - INIT_HOM
#
#         # 分类奖励逻辑
#         if predicted_label == 0:  # 过早
#             if ttt_diff > 0 and hom_diff == 0:
#                 reward += 0.3  # 优先级最高奖励
#             elif ttt_diff > 0 and hom_diff > 0:
#                 reward += 0.2
#             elif ttt_diff == 0 and hom_diff > 0:
#                 reward += 0.2
#             elif ttt_diff == 0 and hom_diff == 0:
#                 reward -= 0.3
#             elif ttt_diff < 0 and hom_diff <= 0:
#                 reward -= 0.3
#         elif predicted_label == 1:  # 过晚
#             if ttt_diff < 0 and hom_diff == 0:
#                 reward += 0.3
#             elif ttt_diff < 0 and hom_diff < 0:
#                 reward += 0.2
#             elif ttt_diff == 0 and hom_diff < 0:
#                 reward += 0.2
#             elif ttt_diff == 0 and hom_diff == 0:
#                 reward += 0.2
#             elif ttt_diff > 0 and hom_diff >= 0:
#                 reward -= 0.3
#         elif predicted_label == 2:  # 理想
#             if ttt_diff == 0 and hom_diff == 0:
#                 reward += 0.3
#             elif (ttt_diff > 0 and hom_diff < 0) or (ttt_diff < 0 and hom_diff > 0):
#                 # 限制极端动作偏移范围，允许小范围调整
#                 if abs(ttt_diff) <= 500 and abs(hom_diff) <= 0.5:
#                     reward += 0.0
#                 else:
#                     reward -= 0.3
#             else:
#                 reward -= 0.3
#
#     reward = max(min(reward, 5.0), -5.0)  # 限幅
#     return reward / 5.0

# # V4
#
# def is_reasonable_action(predicted_label, ttt_diff, hom_diff, a3_satisfied):
#     if predicted_label == 0:  # 过早，应延迟切换
#         return (ttt_diff > 0 and hom_diff >= 0) or (ttt_diff == 0 and hom_diff > 0)
#
#     elif predicted_label == 1:  # 过晚，应促进切换
#         return ttt_diff < 0 and a3_satisfied
#
#     elif predicted_label == 2:  # 理想，应维持或微调策略
#         return ttt_diff <= 0 and hom_diff <= 0 and a3_satisfied
#
#     return False
#
#
# def calculate_reward(prev_rsrp, post_rsrp, predicted_label=None,
#                      previous_action=None, action=None,
#                      a3_satisfied=False, handover_executed=False):
#     """
#     奖励函数：预测标签值、动作合理性、切换结果、 A3
#     :param prev_rsrp: 切换前RSRP
#     :param post_rsrp: 切换后RSRP
#     :param predicted_label: 预测标签（0=过早，1=过晚，2=理想）
#     :param previous_action: 上一次动作 (TTT, HOM)
#     :param action: 当前动作 (TTT, HOM)
#     :param a3_satisfied: 当前动作是否满足 A3 条件
#     :param handover_executed: 是否真实发生切换
#     :return: 奖励值 ∈ [-1.0, 1.0]
#     """
#     reward = 0.0
#
#     if action is None or previous_action is None or predicted_label is None:
#         return 0.0
#
#     prev_ttt, prev_hom = previous_action
#     curr_ttt, curr_hom = action
#     ttt_diff = curr_ttt - prev_ttt
#     hom_diff = curr_hom - prev_hom
#
#     is_reasonable = is_reasonable_action(predicted_label, ttt_diff, hom_diff, a3_satisfied)
#
#     # 标签驱动评估
#     if predicted_label == 0:  # 过早
#         if is_reasonable:
#             if not handover_executed:
#                 reward += 0.1
#             else:
#                 reward -= 0.1  # 合理动作却未避免切换
#         else:
#             reward -= 0.1
#             if handover_executed:
#                 reward -= 0.1
#
#     elif predicted_label == 1:  # 过晚
#         if is_reasonable:
#             if handover_executed:
#                 if post_rsrp > prev_rsrp:
#                     reward += 0.1
#                 else:
#                     reward += 0.1
#             else:
#                 reward -= 0.1
#         else:
#             reward -= 0.1
#             if not handover_executed:
#                 reward -= 0.1
#
#     elif predicted_label == 2:  # 理想
#         if is_reasonable:
#             if handover_executed:
#                 if post_rsrp > prev_rsrp:
#                     reward += 0.1
#                 else:
#                     reward += 0.1
#             else:
#                 reward -= 0.1  # 本应发生但没发生
#         else:
#             reward -= 0.1
#             if not handover_executed:
#                 reward -= 0.1
#
#     return max(min(reward, 1.0), -1.0)

# V5


# def is_reasonable_action(predicted_label, ttt_diff, hom_diff, a3_satisfied):
#     if predicted_label == 0:  # 过早，应延迟切换
#         return (ttt_diff > 0 and hom_diff >= 0) or (ttt_diff == 0 and hom_diff > 0)
#
#     elif predicted_label == 1:  # 过晚，应促进切换
#         return ttt_diff < 0 and a3_satisfied
#
#     elif predicted_label == 2:  # 理想，应维持或微调策略
#         return ttt_diff <= 0 and hom_diff <= 0 and a3_satisfied
#
#     return False
#
#
# def calculate_reward(
#     prev_rsrp,
#     post_rsrp,
#     predicted_label=None,
#     previous_action=None,
#     action=None,
#     a3_satisfied=False,
#     handover_executed=False,
# ):
#     """
#     奖励函数：预测标签值、动作合理性、切换结果、 A3
#     :param prev_rsrp: 切换前RSRP
#     :param post_rsrp: 切换后RSRP
#     :param predicted_label: 预测标签（0=过早，1=过晚，2=理想）
#     :param previous_action: 上一次动作 (TTT, HOM)
#     :param action: 当前动作 (TTT, HOM)
#     :param a3_satisfied: 当前动作是否满足 A3 条件
#     :param handover_executed: 是否真实发生切换
#     :return: 奖励值 ∈ [-1.0, 1.0]
#     """
#     reward = 0.0
#
#     if action is None or previous_action is None or predicted_label is None:
#         return 0.0
#
#     prev_ttt, prev_hom = previous_action
#     curr_ttt, curr_hom = action
#     ttt_diff = curr_ttt - prev_ttt
#     hom_diff = curr_hom - prev_hom
#
#     is_reasonable = is_reasonable_action(
#         predicted_label, ttt_diff, hom_diff, a3_satisfied
#     )
#
#     # 标签驱动评估
#     if predicted_label == 0:  # 过早
#         if is_reasonable:
#             reward += 0.15
#             # if not handover_executed:
#             #     reward += 0.1
#             # else:
#             #     reward -= 0.1  # 合理动作却未避免切换
#         else:
#             reward -= 0.1
#             # if handover_executed:
#             #     reward -= 0.1
#
#     elif predicted_label == 1:  # 过晚
#         if is_reasonable:
#             reward += 0.15
#             # if handover_executed:
#             #     if post_rsrp > prev_rsrp:
#             #         reward += 0.1
#             #     else:
#             #         reward += 0.1
#             # else:
#             #     reward -= 0.1
#         else:
#             reward -= 0.1
#             # if not handover_executed:
#             #     reward -= 0.1
#
#     elif predicted_label == 2:  # 理想
#         if is_reasonable:
#             reward += 0.15
#             # if handover_executed:
#             #     if post_rsrp > prev_rsrp:
#             #         reward += 0.1
#             #     else:
#             #         reward += 0.1
#             # else:
#             #     reward -= 0.1  # 本应发生但没发生
#         else:
#             reward -= 0.1
#             # if not handover_executed:
#             #     reward -= 0.1
#
#     return max(min(reward, 1.0), -1.0)

# V6
# def is_reasonable_action(predicted_label, ttt_diff, hom_diff, a3_satisfied):
#     if predicted_label == 0:  # 过早，应延迟切换
#         return (ttt_diff > 0 and hom_diff >= 0) or (ttt_diff == 0 and hom_diff > 0)
#
#     elif predicted_label == 1:  # 过晚，应促进切换
#         return ttt_diff <= 0 and a3_satisfied
#
#     elif predicted_label == 2:  # 理想，应维持或微调策略
#         return ttt_diff <= 0 and hom_diff <= 0 and a3_satisfied
#
#     return False


# def is_reasonable_action(predicted_label, ttt_diff, hom_diff, a3_satisfied):
#     if predicted_label == 0:  # 过早，应延迟切换
#         if ttt_diff > 0 and hom_diff > 0:
#             return True
#         elif ttt_diff > 0 and hom_diff == 0:
#             return True
#         elif ttt_diff == 0 and hom_diff > 0:
#             return True
#         else:
#             return False
#
#     elif predicted_label == 1:  # 过晚，应促进切换
#         if ttt_diff < 0 and a3_satisfied:
#             return True
#         elif ttt_diff == 0 and a3_satisfied:
#             return True
#         else:
#             return False
#
#     elif predicted_label == 2:  # 理想，应维持或微调策略
#
#         if a3_satisfied:
#             if ttt_diff < 0 and hom_diff < 0:
#                 return True
#             elif ttt_diff < 0 and hom_diff == 0:
#                 return True
#             elif ttt_diff == 0 and hom_diff < 0:
#                 return True
#             elif ttt_diff == 0 and hom_diff == 0:
#                 return True
#             else:
#                 # TTT 或 HOM 增大
#                 return False
#         else:
#             # 不满足 A3，直接不合理
#             return False
#     # fallback: 非法 predicted_label
#     return False


# def calculate_reward(
#     prev_rsrp,
#     post_rsrp,
#     predicted_label=None,
#     previous_action=None,
#     action=None,
#     a3_satisfied=False,
#     handover_executed=False,
#     flag_reward_rsrp=False,  # 是否启用奖励细节
# ):
#     reward = 0.0
#
#     if action is None or previous_action is None or predicted_label is None:
#         return 0.0
#
#     prev_ttt, prev_hom = previous_action
#     curr_ttt, curr_hom = action
#     ttt_diff = curr_ttt - prev_ttt
#     hom_diff = curr_hom - prev_hom
#
#     is_reasonable = is_reasonable_action(
#         predicted_label, ttt_diff, hom_diff, a3_satisfied
#     )
#
#     if predicted_label == 0:  # 过早
#         if is_reasonable:
#             reward += 0.15
#             if flag_reward_rsrp:
#                 if not handover_executed:
#                     reward += 0.05
#                 else:
#                     reward -= 0.05
#         else:
#             reward -= 0.1
#             if flag_reward_rsrp:
#                 if handover_executed:
#                     reward -= 0.05
#
#     elif predicted_label == 1:  # 过晚
#         if is_reasonable:
#             reward += 0.15
#             if flag_reward_rsrp:
#                 if handover_executed:
#                     if post_rsrp > prev_rsrp:
#                         reward += 0.05
#                     else:
#                         reward -= 0.05
#                 else:
#                     reward -= 0.05
#         else:
#             reward -= 0.1
#             if flag_reward_rsrp:
#                 if not handover_executed:
#                     reward -= 0.05
#
#     elif predicted_label == 2:  # 理想
#         if is_reasonable:
#             reward += 0.15
#             if flag_reward_rsrp:
#                 if handover_executed:
#                     if post_rsrp > prev_rsrp:
#                         reward += 0.05
#                     else:
#                         reward -= 0.05
#                 else:
#                     reward -= 0.05
#         else:
#             reward -= 0.1
#             if flag_reward_rsrp:
#                 if not handover_executed:
#                     reward -= 0.05
#
#     return max(min(reward, 1.0), -1.0)
