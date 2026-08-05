# -*- coding: utf-8 -*-
import numpy as np


def path_loss_calculation(distance, frequency):
    # pl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.44
    pl = 36.7 * np.log10(distance) + 26 * np.log10(frequency) + 22.7

    return pl


def sinr_calculation(
    signal_bs_rsrp,
    signal_bs_id,
    user,
    bs_list,
    frequency,
    tx_power,
    shadow_sigma_db=np.sqrt(2),
):
    # 计算信号强度 mw
    signal_in_mw = 10 ** (signal_bs_rsrp / 10)

    # 计算干扰 mw
    total_interference_in_mw = 0

    for interference_bs in bs_list:
        if interference_bs.bs_id == signal_bs_id:
            continue

        interference_distance = np.sqrt(
            (user.curr_x - interference_bs.bs_x) ** 2
            + (user.curr_y - interference_bs.bs_y) ** 2
        )

        interference_pl = path_loss_calculation(interference_distance, frequency)

        interference_rsrp = tx_power - interference_pl - np.random.normal(
            loc=0, scale=shadow_sigma_db
        )

        total_interference_in_mw += 10 ** (interference_rsrp / 10)

    # 计算噪声 mw
    noise_in_mw = 10 ** (-174 / 10) * 1e6  # 噪声功率谱密度 -174 dBm/Hz

    # 计算 SINR dB
    sinr_in_db = 10 * np.log10(signal_in_mw / (total_interference_in_mw + noise_in_mw))

    return sinr_in_db


def average_rsrp_calculation(dl_source_bs_rsrp_list):
    if len(dl_source_bs_rsrp_list) == 0:
        return 0
    else:
        total_rsrp_in_mw = 0
        for rsrp_in_dbm in dl_source_bs_rsrp_list:
            total_rsrp_in_mw += 10 ** (rsrp_in_dbm / 10)

        average_rsrp_in_mw = total_rsrp_in_mw / len(dl_source_bs_rsrp_list)

        return 10 * np.log10(average_rsrp_in_mw)


def estimate_throughput(sinr_db, bandwidth=20e6):
    sinr_linear = 10 ** (sinr_db / 10)
    return bandwidth * np.log2(1 + sinr_linear) if sinr_linear > 0 else 0
