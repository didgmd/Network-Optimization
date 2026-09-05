# -*- coding: utf-8 -*-
import numpy as np
from Parameters import (
    SAT_BS_CAPACITY,
    SAT_NUM_OF_SC,
    SAT_TOTAL_BW_HZ,
    MACRO_BS_CAPACITY,
    MACRO_NUM_OF_SC,
    MACRO_TOTAL_BW_HZ,
    SMALL_BS_CAPACITY,
    SMALL_NUM_OF_SC,
    SMALL_TOTAL_BW_HZ,
    UAV_BS_CAPACITY,
    UAV_NUM_OF_SC,
    UAV_TOTAL_BW_HZ,
    CLOUD_REWARD_WEIGHTS,
    EDGE_REWARD_WEIGHTS,
    CLOUD_THR_NORM_Mbps,
    EDGE_THR_NORM_MACRO_Mbps,
    EDGE_THR_NORM_SMALL_Mbps,
    EDGE_THR_NORM_UAV_Mbps,
    EDGE_THR_REQ_MACRO_Mbps,
    EDGE_THR_REQ_SMALL_Mbps,
    EDGE_THR_REQ_UAV_Mbps,
    SAT_THR_REQ_Mbps,
    NOISE_PSD_dBm_perHz,
    NOISE_FIGURE_dB,
)


def calculate_rate_mbps_helper(
    rsrp_dbm, num_sc, total_sc, total_bw_hz, power_split=1.0
):
    if num_sc <= 0 or rsrp_dbm is None:
        return 0.0
    rsrp_dbm = max(rsrp_dbm, -140.0)
    power_split = max(power_split, 1e-6)

    b_sc = total_bw_hz / total_sc
    noise_sc = NOISE_PSD_dBm_perHz + 10 * np.log10(b_sc) + NOISE_FIGURE_dB
    effective_rsrp = rsrp_dbm + 10 * np.log10(power_split)
    snr_db = effective_rsrp - noise_sc
    snr_linear = 10 ** (snr_db / 10)

    rate_mbps = (num_sc * b_sc * np.log2(1 + snr_linear)) / 1_000_000
    return rate_mbps


def tx_power_per_sc_dbm(total_tx_power_dbm, total_sc):
    total_sc = max(total_sc, 1)
    return total_tx_power_dbm - 10 * np.log10(total_sc)


def saturation_rate_mbps(rsrp_dbm, total_sc, total_bw_hz):
    return calculate_rate_mbps_helper(
        rsrp_dbm, total_sc, total_sc, total_bw_hz, power_split=1.0
    )


def required_rate_mbps(rsrp_dbm, total_sc, total_bw_hz, capacity):
    sat_rate = saturation_rate_mbps(rsrp_dbm, total_sc, total_bw_hz)
    capacity = max(capacity, 1)
    return max(sat_rate / capacity, 1.0)


def calculate_jain_index_helper(satisfaction_ratios):
    if not satisfaction_ratios:
        return 0.0
    n = len(satisfaction_ratios)
    sum_vals = sum(satisfaction_ratios)
    if sum_vals == 0:
        return 0.0
    sum_sq = sum(v**2 for v in satisfaction_ratios)
    if sum_sq == 0:
        return 0.0
    return (sum_vals**2) / (n * sum_sq)


def normalize_reward_ratio(value, mode=None, scale=9.0):
    if mode == "log":
        safe = max(value, 0.0)
        return float(np.log1p(safe * scale) / np.log1p(scale))
    return value


def stretch_reward_ratio(value, gamma=None):
    if gamma is None:
        return value
    gamma = float(gamma)
    if gamma <= 0:
        return value
    safe = max(min(value, 1.0), 0.0)
    return float(safe**gamma)


def cloud_reward_calculation(
    cloud_action,
    user,
    predicted_loads,
    rsrp_list,
    cloud_weights=None,
    norm_mode=None,
    use_required_rate=True,
):
    assigned_bs_idx = int(cloud_action)

    if assigned_bs_idx == 0:
        num_sc = SAT_NUM_OF_SC
        total_bw = SAT_TOTAL_BW_HZ
        bs_capacity = SAT_BS_CAPACITY
        required_rate = SAT_THR_REQ_Mbps if use_required_rate else user.qos["rate"]
    elif 1 <= assigned_bs_idx <= 2:
        num_sc = MACRO_NUM_OF_SC
        total_bw = MACRO_TOTAL_BW_HZ
        bs_capacity = MACRO_BS_CAPACITY
        required_rate = (
            EDGE_THR_REQ_MACRO_Mbps if use_required_rate else user.qos["rate"]
        )
    elif 3 <= assigned_bs_idx <= 6:
        num_sc = SMALL_NUM_OF_SC
        total_bw = SMALL_TOTAL_BW_HZ
        bs_capacity = SMALL_BS_CAPACITY
        required_rate = (
            EDGE_THR_REQ_SMALL_Mbps if use_required_rate else user.qos["rate"]
        )
    else:
        num_sc = UAV_NUM_OF_SC
        total_bw = UAV_TOTAL_BW_HZ
        bs_capacity = UAV_BS_CAPACITY
        required_rate = (
            EDGE_THR_REQ_UAV_Mbps if use_required_rate else user.qos["rate"]
        )

    max_capacity = max(
        SAT_BS_CAPACITY, MACRO_BS_CAPACITY, SMALL_BS_CAPACITY, UAV_BS_CAPACITY
    )
    r_cap = bs_capacity / max_capacity

    current_load = (
        predicted_loads[assigned_bs_idx] if predicted_loads is not None else 0.0
    )
    est_users = max(current_load * bs_capacity, 1.0)
    sc_per_user = max(num_sc / est_users, 1.0)

    rsrp_dbm = rsrp_list[assigned_bs_idx]
    estimated_rate = calculate_rate_mbps_helper(rsrp_dbm, sc_per_user, num_sc, total_bw)

    r_qos = min(estimated_rate / required_rate, 1.0)

    if current_load <= 0.5:
        r_load = 1.0 - current_load
    else:
        r_load = (1.0 - current_load) ** 2

    r_thr = min(estimated_rate / CLOUD_THR_NORM_Mbps, 1.0)

    r_qos = normalize_reward_ratio(r_qos, norm_mode)
    r_thr = normalize_reward_ratio(r_thr, norm_mode)

    weights = cloud_weights if cloud_weights is not None else CLOUD_REWARD_WEIGHTS
    total = (
        weights["qos"] * r_qos
        + weights["load"] * r_load
        + weights["thr"] * r_thr
        + weights["cap"] * r_cap
    )

    return {
        "total": total,
        "qos": r_qos,
        "load": r_load,
        "thr": r_thr,
        "cap": r_cap,
    }


def macro_reward_calculation(
    action_list,
    user_list,
    frequency,
    tx_power,
    bs_object=None,
    edge_weights=None,
    norm_mode=None,
    return_details=False,
    use_required_rate=True,
    use_sc_budget=True,
    use_total_power=False,
):
    if not action_list or not user_list:
        return (0.0, None) if return_details else 0.0

    satisfaction_ratios = []
    ratio_uncapped = []
    actual_rates = []
    remaining_sc = MACRO_NUM_OF_SC
    for i, user in enumerate(user_list):
        if i < len(action_list):
            sc = action_list[i][0]
            mu = action_list[i][1]
            tx_power_dbm = (
                tx_power if use_total_power else tx_power_per_sc_dbm(tx_power, MACRO_NUM_OF_SC)
            )
            rsrp = user.calculate_rsrp(
                "curr",
                bs_object,
                frequency,
                tx_power_dbm,
            )
            req = EDGE_THR_REQ_MACRO_Mbps if use_required_rate else user.qos["rate"]
            sc = max(int(sc), 0)
            if use_sc_budget:
                alloc_sc = min(sc, remaining_sc)
                remaining_sc -= alloc_sc
            else:
                alloc_sc = sc
            rate = calculate_rate_mbps_helper(
                rsrp, alloc_sc, MACRO_NUM_OF_SC, MACRO_TOTAL_BW_HZ, mu
            )
            ratio = rate / req
            ratio_uncapped.append(ratio)
            satisfaction_ratios.append(min(ratio, 1.0))
            actual_rates.append(rate)
        else:
            satisfaction_ratios.append(0.0)
            actual_rates.append(0.0)

    avg_qos = np.mean(satisfaction_ratios) if satisfaction_ratios else 0.0
    fair = calculate_jain_index_helper(satisfaction_ratios)
    avg_thr = np.mean(actual_rates) if actual_rates else 0.0
    r_thr_raw = avg_thr / EDGE_THR_NORM_MACRO_Mbps
    r_thr = min(r_thr_raw, 1.0)

    avg_qos_norm = normalize_reward_ratio(avg_qos, norm_mode)
    r_thr_norm = normalize_reward_ratio(r_thr, norm_mode)

    weights = edge_weights if edge_weights is not None else EDGE_REWARD_WEIGHTS
    total = (
        weights["qos"] * avg_qos_norm
        + weights["fair"] * fair
        + weights["thr"] * r_thr_norm
    )

    if not return_details:
        return total

    ratio_uncapped_arr = np.asarray(ratio_uncapped, dtype=float)
    actual_rates_arr = np.asarray(actual_rates, dtype=float)
    details = {
        "avg_qos": float(avg_qos),
        "fair": float(fair),
        "avg_thr_mbps": float(avg_thr),
        "r_thr_raw": float(r_thr_raw),
        "r_thr_capped": float(r_thr),
        "avg_qos_norm": float(avg_qos_norm),
        "r_thr_norm": float(r_thr_norm),
        "ratio_uncapped_mean": (
            float(np.mean(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "ratio_uncapped_max": (
            float(np.max(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "rate_max_mbps": (
            float(np.max(actual_rates_arr)) if actual_rates_arr.size > 0 else 0.0
        ),
        "reward_total": float(total),
    }
    return total, details


def small_reward_calculation(
    action_list,
    user_list,
    frequency,
    tx_power,
    bs_object=None,
    edge_weights=None,
    norm_mode=None,
    return_details=False,
    use_required_rate=True,
    use_sc_budget=True,
    use_total_power=False,
):
    if not action_list or not user_list:
        return (0.0, None) if return_details else 0.0

    satisfaction_ratios = []
    ratio_uncapped = []
    actual_rates = []
    remaining_sc = SMALL_NUM_OF_SC
    for i, user in enumerate(user_list):
        if i < len(action_list):
            sc = action_list[i][0]
            mu = action_list[i][1]
            tx_power_dbm = (
                tx_power if use_total_power else tx_power_per_sc_dbm(tx_power, SMALL_NUM_OF_SC)
            )
            rsrp = user.calculate_rsrp(
                "curr",
                bs_object,
                frequency,
                tx_power_dbm,
            )
            req = EDGE_THR_REQ_SMALL_Mbps if use_required_rate else user.qos["rate"]
            sc = max(int(sc), 0)
            if use_sc_budget:
                alloc_sc = min(sc, remaining_sc)
                remaining_sc -= alloc_sc
            else:
                alloc_sc = sc
            rate = calculate_rate_mbps_helper(
                rsrp, alloc_sc, SMALL_NUM_OF_SC, SMALL_TOTAL_BW_HZ, mu
            )
            ratio = rate / req
            ratio_uncapped.append(ratio)
            satisfaction_ratios.append(min(ratio, 1.0))
            actual_rates.append(rate)
        else:
            satisfaction_ratios.append(0.0)
            actual_rates.append(0.0)

    avg_qos = np.mean(satisfaction_ratios) if satisfaction_ratios else 0.0
    fair = calculate_jain_index_helper(satisfaction_ratios)
    avg_thr = np.mean(actual_rates) if actual_rates else 0.0
    r_thr_raw = avg_thr / EDGE_THR_NORM_SMALL_Mbps
    r_thr = min(r_thr_raw, 1.0)

    avg_qos_norm = normalize_reward_ratio(avg_qos, norm_mode)
    r_thr_norm = normalize_reward_ratio(r_thr, norm_mode)

    weights = edge_weights if edge_weights is not None else EDGE_REWARD_WEIGHTS
    total = (
        weights["qos"] * avg_qos_norm
        + weights["fair"] * fair
        + weights["thr"] * r_thr_norm
    )

    if not return_details:
        return total

    ratio_uncapped_arr = np.asarray(ratio_uncapped, dtype=float)
    actual_rates_arr = np.asarray(actual_rates, dtype=float)
    details = {
        "avg_qos": float(avg_qos),
        "fair": float(fair),
        "avg_thr_mbps": float(avg_thr),
        "r_thr_raw": float(r_thr_raw),
        "r_thr_capped": float(r_thr),
        "avg_qos_norm": float(avg_qos_norm),
        "r_thr_norm": float(r_thr_norm),
        "ratio_uncapped_mean": (
            float(np.mean(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "ratio_uncapped_max": (
            float(np.max(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "rate_max_mbps": (
            float(np.max(actual_rates_arr)) if actual_rates_arr.size > 0 else 0.0
        ),
        "reward_total": float(total),
    }
    return total, details


def uav_reward_calculation(
    user_list,
    uav_bs,
    frequency,
    tx_power,
    edge_weights=None,
    norm_mode=None,
    reward_gamma=None,
    reward_scale=None,
    return_details=False,
    use_required_rate=True,
    use_sc_budget=True,
    use_total_power=False,
):
    if not user_list:
        return (0.0, None) if return_details else 0.0

    num_users = len(user_list)
    sc_per_user = UAV_NUM_OF_SC / num_users if num_users > 0 else 0
    remaining_sc = UAV_NUM_OF_SC

    satisfaction_ratios = []
    ratio_uncapped = []
    actual_rates = []
    for user in user_list:
        tx_power_dbm = (
            tx_power if use_total_power else tx_power_per_sc_dbm(tx_power, UAV_NUM_OF_SC)
        )
        rsrp = user.calculate_rsrp(
            "curr",
            uav_bs,
            frequency,
            tx_power_dbm,
        )
        req = EDGE_THR_REQ_UAV_Mbps if use_required_rate else user.qos["rate"]
        if use_sc_budget:
            alloc_sc = min(int(sc_per_user), remaining_sc)
            remaining_sc -= alloc_sc
        else:
            alloc_sc = int(sc_per_user)
        rate = calculate_rate_mbps_helper(
            rsrp, alloc_sc, UAV_NUM_OF_SC, UAV_TOTAL_BW_HZ, 1.0
        )
        ratio = rate / req
        ratio_uncapped.append(ratio)
        satisfaction_ratios.append(min(ratio, 1.0))
        actual_rates.append(rate)

    avg_qos = np.mean(satisfaction_ratios) if satisfaction_ratios else 0.0
    fair = calculate_jain_index_helper(satisfaction_ratios)
    avg_thr = np.mean(actual_rates) if actual_rates else 0.0
    r_thr_raw = avg_thr / EDGE_THR_NORM_UAV_Mbps
    r_thr = min(r_thr_raw, 1.0)

    avg_qos_norm = normalize_reward_ratio(avg_qos, norm_mode)
    r_thr_norm = normalize_reward_ratio(r_thr, norm_mode)
    avg_qos_final = stretch_reward_ratio(avg_qos_norm, reward_gamma)
    r_thr_final = stretch_reward_ratio(r_thr_norm, reward_gamma)
    if reward_scale is not None:
        scale = max(min(float(reward_scale), 1.0), 0.0)
        avg_qos_final = avg_qos_final * scale
        r_thr_final = r_thr_final * scale

    weights = edge_weights if edge_weights is not None else EDGE_REWARD_WEIGHTS
    total = (
        weights["qos"] * avg_qos_final
        + weights["fair"] * fair
        + weights["thr"] * r_thr_final
    )

    if not return_details:
        return total

    ratio_uncapped_arr = np.asarray(ratio_uncapped, dtype=float)
    actual_rates_arr = np.asarray(actual_rates, dtype=float)
    details = {
        "avg_qos": float(avg_qos),
        "fair": float(fair),
        "avg_thr_mbps": float(avg_thr),
        "r_thr_raw": float(r_thr_raw),
        "r_thr_capped": float(r_thr),
        "avg_qos_norm": float(avg_qos_norm),
        "r_thr_norm": float(r_thr_norm),
        "avg_qos_final": float(avg_qos_final),
        "r_thr_final": float(r_thr_final),
        "ratio_uncapped_mean": (
            float(np.mean(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "ratio_uncapped_max": (
            float(np.max(ratio_uncapped_arr)) if ratio_uncapped_arr.size > 0 else 0.0
        ),
        "rate_max_mbps": (
            float(np.max(actual_rates_arr)) if actual_rates_arr.size > 0 else 0.0
        ),
        "reward_total": float(total),
    }
    return total, details


def calculate_global_reward(cloud_reward, macro_reward, small_reward, uav_reward):
    return cloud_reward + macro_reward + small_reward + uav_reward
