# -*- coding: utf-8 -*-
import random
import os
import sys
import csv
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from ActionChooser import cloud_choose_action, bs_choose_action, uav_choose_action
from Classes import CloudAgent, BsAgent, UavAgent
from DebugPrint import set_debug_mode, debug_print
from Parameters import (
    print_parameters,
    CLOUD_INPUT_DIM,
    CLOUD_OUTPUT_DIM,
    CLOUD_HIDDEN_DIM,
    CLOUD_LEARNING_RATE,
    CLOUD_GAMMA,
    CLOUD_EPSILON_START,
    CLOUD_EPSILON_END,
    CLOUD_EPSILON_DECAY_EPOCHS,
    CLOUD_ASSOCIATION_PERIOD,
    SAT_BS_CAPACITY,
    SAT_NUM_OF_SC,
    SAT_FREQUENCY,
    SAT_TX_POWER,
    SAT_TOTAL_BW_HZ,
    MACRO_INPUT_DIM,
    MACRO_OUTPUT_DIM,
    MACRO_HIDDEN_DIM,
    MACRO_LEARNING_RATE,
    MACRO_GAMMA,
    MACRO_EPSILON_START,
    MACRO_EPSILON_END,
    MACRO_EPSILON_DECAY_EPOCHS,
    MACRO_NUM_OF_SC,
    MACRO_TX_POWER,
    MACRO_FREQUENCY,
    MACRO_BS_CAPACITY,
    MACRO_TOTAL_BW_HZ,
    SMALL_INPUT_DIM,
    SMALL_OUTPUT_DIM,
    SMALL_HIDDEN_DIM,
    SMALL_LEARNING_RATE,
    SMALL_EPSILON_START,
    SMALL_EPSILON_END,
    SMALL_EPSILON_DECAY_EPOCHS,
    SMALL_NUM_OF_SC,
    SMALL_GAMMA,
    SMALL_TX_POWER,
    SMALL_FREQUENCY,
    SMALL_BS_CAPACITY,
    SMALL_TOTAL_BW_HZ,
    UAV_INPUT_DIM,
    UAV_OUTPUT_DIM,
    UAV_HIDDEN_DIM,
    UAV_LEARNING_RATE,
    UAV_GAMMA,
    UAV_TOTAL_BW_HZ,
    UAV_EPSILON_START,
    UAV_EPSILON_END,
    UAV_EPSILON_DECAY_EPOCHS,
    UAV_BS_CAPACITY,
    UAV_NUM_OF_SC,
    UAV_FREQUENCY,
    UAV_TX_POWER,
    AREA_SIZE_X,
    AREA_SIZE_Y,
    UAV_MAX_HEIGHT,
    MAX_EPOCHS,
    TAU_SOFT_UPDATE,
    MOVING_AVG_WINDOW,
    CLOUD_SWITCH_PENALTY,
    CLOUD_STICKINESS_BONUS,
    CLOUD_REWARD_WEIGHTS,
    EDGE_REWARD_WEIGHTS,
    EDGE_THR_REQ_MACRO_Mbps,
    EDGE_THR_REQ_SMALL_Mbps,
    EDGE_THR_REQ_UAV_Mbps,
    SAT_THR_REQ_Mbps,
)
from RewardCalculator import (
    cloud_reward_calculation,
    macro_reward_calculation,
    small_reward_calculation,
    uav_reward_calculation,
    calculate_global_reward,
    calculate_rate_mbps_helper as calculate_rate_mbps,
    calculate_jain_index_helper as calculate_jain_fairness,
)
from Topology import define_topology, define_users


def normalize_rsrp(rsrp_dbm):
    normalized = (rsrp_dbm + 100) / 50.0
    return np.clip(normalized, 0.0, 1.0)


def tx_power_per_sc_dbm(total_tx_power_dbm, total_sc):
    total_sc = max(total_sc, 1)
    return total_tx_power_dbm - 10 * np.log10(total_sc)


def get_tx_power_dbm(total_tx_power_dbm, total_sc, use_total_power):
    if use_total_power:
        return total_tx_power_dbm
    return tx_power_per_sc_dbm(total_tx_power_dbm, total_sc)


def build_sdm_order_indices(
    user_count, mode="fixed_order", seed=0, cloud_epoch=0
):
    order_indices = list(range(user_count))
    if mode == "fixed_order":
        return order_indices
    if mode == "random_per_cloud_epoch_order":
        rng = random.Random(int(seed) + int(cloud_epoch) * 1000003)
        rng.shuffle(order_indices)
        return order_indices
    raise ValueError(f"Unsupported sdm_order_mode: {mode}")


def saturation_rate_mbps(rsrp_dbm, total_sc, total_bw_hz):
    return calculate_rate_mbps(rsrp_dbm, total_sc, total_sc, total_bw_hz, 1.0)


def required_rate_mbps(rsrp_dbm, total_sc, total_bw_hz, capacity):
    sat_rate = saturation_rate_mbps(rsrp_dbm, total_sc, total_bw_hz)
    capacity = max(capacity, 1)
    return max(sat_rate / capacity, 1.0)


def renormalize_weights(weights):
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return weights
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def moving_average(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def moving_average_full(values, window):
    arr = np.asarray(values, dtype=float)
    out = []
    for i in range(len(arr)):
        if i + 1 < window:
            out.append(np.nan)
        else:
            seg = arr[i + 1 - window : i + 1]
            out.append(float(np.mean(seg)))
    return out


def average_detail_dicts(detail_list):
    if not detail_list:
        return None
    keys = detail_list[0].keys()
    averaged = {}
    for key in keys:
        averaged[key] = float(np.mean([d.get(key, 0.0) for d in detail_list]))
    return averaged


def compute_epsilon(epoch, start, end, decay_epochs, schedule="linear"):
    if decay_epochs <= 0:
        return end
    progress = min(max(epoch, 0), decay_epochs) / decay_epochs
    if schedule == "log":
        log_progress = np.log1p(progress * (np.e - 1.0))
        return end + (start - end) * (1.0 - log_progress)
    if schedule == "exp2":
        return max(end, start / (2 ** (epoch / 1000.0)))
    return start - (start - end) * progress


def build_epsilon_curve(epochs, start, end, decay_epochs, schedule="linear"):
    return [compute_epsilon(e, start, end, decay_epochs, schedule) for e in epochs]


def compute_rate_cv(rates):
    if not rates:
        return 0.0
    arr = np.asarray(rates, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def compute_rate_gini(rates):
    if not rates:
        return 0.0
    arr = np.sort(np.asarray(rates, dtype=float))
    total = np.sum(arr)
    if total == 0:
        return 0.0
    n = len(arr)
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / total) / n)


def compute_load_cv(loads):
    if not loads:
        return 0.0
    arr = np.asarray(loads, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def get_bs_capacity(bid):
    if bid == 0:
        return SAT_BS_CAPACITY
    if bid in [1, 2]:
        return MACRO_BS_CAPACITY
    if bid in [3, 4, 5, 6]:
        return SMALL_BS_CAPACITY
    return UAV_BS_CAPACITY


def get_target_bs(bid, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list):
    if bid == 0:
        return sat_bs_list[0]
    if 1 <= bid <= 2:
        return macro_bs_list[bid - 1]
    if 3 <= bid <= 6:
        return small_bs_list[bid - 3]
    if 7 <= bid <= 8:
        return uav_bs_list[bid - 7]
    return None


def _validate_cloud_action_contract(
    cloud_agent, available_mask, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
):
    output_dim = getattr(getattr(cloud_agent, "fc2", None), "out_features", None)
    if CLOUD_OUTPUT_DIM != 9 or output_dim != 9:
        raise RuntimeError(
            "R4-4 requires configured and runtime cloud output dimensions of 9; "
            f"got {CLOUD_OUTPUT_DIM} and {output_dim}."
        )
    if not sat_bs_list or get_target_bs(
        0, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
    ) is not sat_bs_list[0]:
        raise RuntimeError("R4-4 satellite action mapping requires index 0.")
    if available_mask is None:
        return None
    if not isinstance(available_mask, (list, tuple, np.ndarray)):
        raise TypeError("cloud_action_available_mask must be a list, tuple, or array.")
    mask = np.asarray(available_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise TypeError("cloud_action_available_mask must be one-dimensional and boolean.")
    if len(mask) != 9:
        raise ValueError("cloud_action_available_mask must contain exactly 9 entries.")
    if not bool(mask.any()):
        raise ValueError("cloud_action_available_mask must enable at least one action.")
    return mask


def _validate_sdm_order_indices(order_indices, user_count):
    if any(not isinstance(index, (int, np.integer)) for index in order_indices):
        raise TypeError("SDM order entries must be integer user positions.")
    if len(order_indices) != user_count or set(order_indices) != set(range(user_count)):
        raise RuntimeError("SDM order must cover every user position exactly once.")
    return order_indices


def _action_is_available(action, available_mask):
    return available_mask is None or bool(available_mask[int(action)])


def _commit_cloud_action_profile(
    current_cloud_actions, processed_user_keys, expected_user_keys, available_mask
):
    if (
        processed_user_keys != expected_user_keys
        or set(current_cloud_actions) != expected_user_keys
        or len(current_cloud_actions) != len(expected_user_keys)
    ):
        raise RuntimeError("Cloud action profile is incomplete and cannot be committed.")
    if any(
        not _action_is_available(action, available_mask)
        for action in current_cloud_actions.values()
    ):
        raise RuntimeError("Cloud action profile contains an unavailable action.")
    return dict(current_cloud_actions)


def _read_historical_action(cloud_historical_actions, user_key, available_mask):
    action = cloud_historical_actions[user_key]
    if not _action_is_available(action, available_mask):
        raise RuntimeError("Historical cloud action is unavailable.")
    return action


def _append_cloud_experience(
    current_slot_experiences, curr_state, action, reward, next_state, device
):
    current_slot_experiences.append(
        (
            torch.tensor(curr_state, dtype=torch.float32).to(device),
            action,
            reward,
            torch.tensor(next_state, dtype=torch.float32).to(device),
        )
    )


def _masked_cloud_q_max(q_values, available_mask):
    mask = torch.as_tensor(available_mask, dtype=torch.bool, device=q_values.device)
    if mask.ndim != 1 or len(mask) != q_values.shape[0] or not bool(mask.any()):
        raise RuntimeError("Invalid cloud availability mask for TD target.")
    return torch.max(q_values.masked_fill(~mask, -torch.inf))


def build_eiap_state(bs, capacity, frequency, tx_power):
    roster = []
    state = []
    for user in bs.serv_user_list:
        if len(roster) >= capacity:
            break
        roster.append(user)
        rsrp = user.calculate_rsrp("curr", bs, frequency, tx_power)
        state.append(
            [normalize_rsrp(rsrp)]
            + [1 if user.user_type == t else 0 for t in ["eMBB", "mMTC", "uRLLC"]]
        )

    active_count = len(roster)
    if active_count < capacity:
        state.extend([[0.0, 0, 0, 0]] * (capacity - active_count))

    return roster, state, active_count


def plot_training_summary(
    losses,
    rewards,
    reward_sum_history,
    throughput_history,
    qos_history,
    jain_history,
    epsilon_histories,
    full_exploration_metrics=None,
    show=False,
    filename="training_summary.png",
):
    ma_window = MOVING_AVG_WINDOW
    fig, axes = plt.subplots(7, 4, figsize=(20, 18))

    loss_titles = ["Cloud Loss", "Macro BS Loss", "Small BS Loss", "UAV Loss"]
    agent_keys = ["cloud", "macro", "small", "uav"]
    for idx, key in enumerate(agent_keys):
        ax = axes[0, idx]
        series = losses.get(key, [])
        if series:
            ax.plot(np.arange(len(series)), series, label=key)
        ax.set_title(loss_titles[idx])
        ax.set_xlabel("Epoch")
        ax.grid(True)

        ax_ma = axes[1, idx]
        if series:
            ma = moving_average(series, ma_window)
            ax_ma.plot(np.arange(len(ma)), ma, label=f"{key} MA")
        ax_ma.set_title(f"{loss_titles[idx]} (MA {ma_window})")
        ax_ma.set_xlabel("Epoch")
        ax_ma.grid(True)

    reward_titles = [
        "Cloud Reward",
        "Macro Reward",
        "Small Reward",
        "UAV Reward",
    ]
    for idx, key in enumerate(agent_keys):
        ax = axes[2, idx]
        series = rewards.get(key, [])
        if series:
            ax.plot(np.arange(len(series)), series, label=key)
        ax.set_title(reward_titles[idx])
        ax.set_xlabel("Epoch")
        ax.grid(True)

        ax_ma = axes[3, idx]
        if series:
            ma = moving_average(series, ma_window)
            ax_ma.plot(np.arange(len(ma)), ma, label=f"{key} MA")
        ax_ma.set_title(f"{reward_titles[idx]} (MA {ma_window})")
        ax_ma.set_xlabel("Epoch")
        ax_ma.grid(True)

    kpi_titles = ["Total Reward", "Total Throughput", "Total QoS", "Total Jain"]
    kpi_series = [reward_sum_history, throughput_history, qos_history, jain_history]
    for idx, series in enumerate(kpi_series):
        ax = axes[4, idx]
        if series:
            ax.plot(np.arange(len(series)), series, label="current")
        if full_exploration_metrics:
            full_series = full_exploration_metrics.get(
                ["reward_sum", "throughput", "qos", "jain"][idx], []
            )
            if full_series:
                ax.plot(np.arange(len(full_series)), full_series, label="full")
        ax.set_title(kpi_titles[idx])
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax_ma = axes[5, idx]
        if series:
            ma = moving_average(series, ma_window)
            ax_ma.plot(np.arange(len(ma)), ma, label="current")
        if full_exploration_metrics:
            full_series = full_exploration_metrics.get(
                ["reward_sum", "throughput", "qos", "jain"][idx], []
            )
            if full_series:
                full_ma = moving_average(full_series, ma_window)
                ax_ma.plot(np.arange(len(full_ma)), full_ma, label="full")
        ax_ma.set_title(f"{kpi_titles[idx]} (MA {ma_window})")
        ax_ma.set_xlabel("Epoch")
        ax_ma.grid(True)
        ax_ma.legend(loc="best", fontsize=8)

    eps_titles = ["Cloud Epsilon", "Macro Epsilon", "Small Epsilon", "UAV Epsilon"]
    for idx, key in enumerate(agent_keys):
        ax = axes[6, idx]
        series = epsilon_histories.get(key, []) if epsilon_histories else []
        if series:
            ax.plot(np.arange(len(series)), series)
        ax.set_title(eps_titles[idx])
        ax.set_xlabel("Epoch")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    if show:
        plt.show()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_csv_matrix(path, expect_cols=None):
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if expect_cols is not None and data.shape[1] < expect_cols:
        return None
    return data


def load_csv_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) <= 1:
        return None
    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def rolling_std(values, window):
    arr = np.asarray(values, dtype=float)
    out = []
    for i in range(len(arr)):
        if i + 1 < window:
            out.append(np.nan)
        else:
            seg = arr[i + 1 - window : i + 1]
            out.append(float(np.std(seg)))
    return out


def last_fraction_mean(values, fraction=0.1):
    if not values:
        return 0.0
    count = max(int(len(values) * fraction), 1)
    return float(np.mean(values[-count:]))


def save_system_metrics(
    logs_dir,
    reward_sum_history,
    throughput_history,
    qos_history,
    jain_history,
    filename,
):
    os.makedirs(logs_dir, exist_ok=True)
    system_path = os.path.join(logs_dir, filename)
    with open(system_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "reward_sum", "throughput", "qos", "jain"])
        for i in range(len(reward_sum_history)):
            writer.writerow(
                [
                    i,
                    reward_sum_history[i],
                    throughput_history[i],
                    qos_history[i],
                    jain_history[i],
                ]
            )


def save_agent_losses(logs_dir, losses):
    os.makedirs(logs_dir, exist_ok=True)
    loss_path = os.path.join(logs_dir, "agent_losses.csv")
    with open(loss_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "cloud", "macro", "small", "uav"])
        for i in range(len(losses["cloud"])):
            writer.writerow(
                [
                    i,
                    losses["cloud"][i],
                    losses["macro"][i],
                    losses["small"][i],
                    losses["uav"][i],
                ]
            )


def save_agent_rewards(logs_dir, rewards):
    os.makedirs(logs_dir, exist_ok=True)
    reward_path = os.path.join(logs_dir, "agent_rewards.csv")
    with open(reward_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "cloud", "macro", "small", "uav"])
        for i in range(len(rewards["cloud"])):
            writer.writerow(
                [
                    i,
                    rewards["cloud"][i],
                    rewards["macro"][i],
                    rewards["small"][i],
                    rewards["uav"][i],
                ]
            )


def rl(
    sat_bs_list,
    macro_bs_list,
    small_bs_list,
    uav_bs_list,
    user_list,
    device,
    epsilon_mode="normal",
    config=None,
):
    config = config or {}
    training_enabled = config.get("training", True)
    max_epochs = config.get("max_epochs", MAX_EPOCHS)
    agent_bundle = config.get("agent_bundle")
    reward_norm_mode = config.get("reward_norm_mode")
    macro_lr = config.get("macro_lr")
    uav_reward_gamma = config.get("uav_reward_gamma")
    uav_reward_scale = config.get("uav_reward_scale")
    uav_reward_log = config.get("uav_reward_log", False)
    use_sc_budget = config.get("use_sc_budget", True)
    use_total_power = config.get("use_total_power", False)
    use_required_rate = config.get("use_required_rate", True)
    convergence_check = config.get("convergence_check", True)
    convergence_window = config.get("convergence_window", 1000)
    convergence_std_ratio = config.get("convergence_std_ratio", 0.05)
    min_convergence_epochs = config.get("min_convergence_epochs", 2000)
    cloud_weights = config.get("cloud_weights")
    edge_weights = config.get("edge_weights")
    switch_penalty = config.get("switch_penalty", CLOUD_SWITCH_PENALTY)
    stickiness_bonus = config.get("stickiness_bonus", CLOUD_STICKINESS_BONUS)
    fixed_epsilon = config.get("fixed_epsilon")
    freeze_uav = config.get("freeze_uav", False)
    use_predicted_loads = config.get("use_predicted_loads", True)
    epsilon_schedule = config.get("epsilon_schedule", "linear")
    epsilon_decay_epochs = config.get("epsilon_decay_epochs")
    shadowing_sigma_db = config.get("shadowing_sigma_db", 0.0)
    shadowing_seed = config.get("shadowing_seed", 0)
    sdm_order_mode = config.get("sdm_order_mode", "fixed_order")
    sdm_order_seed = config.get("sdm_order_seed", config.get("seed", 0))
    cloud_action_available_mask = config.get("cloud_action_available_mask")
    for user in user_list:
        user.configure_shadowing(shadowing_sigma_db, shadowing_seed)
    if agent_bundle:
        cloud_agent = agent_bundle["cloud"].to(device)
        macro_bs_agent = agent_bundle["macro"].to(device)
        small_bs_agent = agent_bundle["small"].to(device)
        uav_bs_agent = agent_bundle["uav"].to(device)
    else:
        cloud_agent = CloudAgent(
            CLOUD_INPUT_DIM, CLOUD_HIDDEN_DIM, CLOUD_OUTPUT_DIM
        ).to(device)
        macro_bs_agent = BsAgent(
            MACRO_INPUT_DIM, MACRO_HIDDEN_DIM, MACRO_OUTPUT_DIM
        ).to(device)
        small_bs_agent = BsAgent(
            SMALL_INPUT_DIM, SMALL_HIDDEN_DIM, SMALL_OUTPUT_DIM
        ).to(device)
        uav_bs_agent = UavAgent(UAV_INPUT_DIM, UAV_HIDDEN_DIM, UAV_OUTPUT_DIM).to(
            device
        )

    cloud_action_available_mask = _validate_cloud_action_contract(
        cloud_agent,
        cloud_action_available_mask,
        sat_bs_list,
        macro_bs_list,
        small_bs_list,
        uav_bs_list,
    )
    expected_user_keys = {user.global_index for user in user_list}
    if len(expected_user_keys) != len(user_list):
        raise RuntimeError("User global_index values must be unique within an R4-4 run.")

    cloud_optimizer = None
    macro_bs_optimizer = None
    small_bs_optimizer = None
    uav_bs_optimizer = None
    if training_enabled:
        cloud_optimizer = torch.optim.Adam(
            cloud_agent.parameters(), lr=CLOUD_LEARNING_RATE
        )
        macro_lr_value = macro_lr if macro_lr is not None else MACRO_LEARNING_RATE
        macro_bs_optimizer = torch.optim.Adam(
            macro_bs_agent.parameters(), lr=macro_lr_value
        )
        small_bs_optimizer = torch.optim.Adam(
            small_bs_agent.parameters(), lr=SMALL_LEARNING_RATE
        )
        uav_bs_optimizer = torch.optim.Adam(
            uav_bs_agent.parameters(), lr=UAV_LEARNING_RATE
        )

    cloud_target = None
    macro_target = None
    small_target = None
    uav_target = None
    if training_enabled:
        cloud_target = deepcopy(cloud_agent).eval()
        macro_target = deepcopy(macro_bs_agent).eval()
        small_target = deepcopy(small_bs_agent).eval()
        uav_target = deepcopy(uav_bs_agent).eval()

        for p in cloud_target.parameters():
            p.requires_grad = False
        for p in macro_target.parameters():
            p.requires_grad = False
        for p in small_target.parameters():
            p.requires_grad = False
        for p in uav_target.parameters():
            p.requires_grad = False

    cloud_replay_buffer = []
    cloud_priorities = []
    replay_capacity = 4000
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA = 0.4
    PRIORITY_BETA_INCREMENT = 0.001
    PRIORITY_BETA_MAX = 1.0
    PRIORITY_EPSILON = 1e-6

    cloud_historical_actions = {}

    cloud_losses = []
    macro_losses = []
    small_losses = []
    uav_losses = []
    cloud_rewards = []
    macro_rewards = []
    small_rewards = []
    uav_rewards = []
    throughput_history = []
    uav_trajectories = [[], []]
    reward_sum_history = []
    qos_history = []
    jain_history = []
    user_rate_history = []
    user_bs_history = []
    bs_load_history = []
    switch_count_history = []
    cloud_eps_history = []
    macro_eps_history = []
    small_eps_history = []
    uav_eps_history = []

    for epoch in range(max_epochs):
        if fixed_epsilon is not None:
            current_cloud_epsilon = fixed_epsilon
            current_macro_epsilon = fixed_epsilon
            current_small_epsilon = fixed_epsilon
            current_uav_epsilon = fixed_epsilon
        elif epsilon_mode == "full_exploration":
            current_cloud_epsilon = 1.0
            current_macro_epsilon = 1.0
            current_small_epsilon = 1.0
            current_uav_epsilon = 1.0
        else:
            decay_epochs = epsilon_decay_epochs or CLOUD_EPSILON_DECAY_EPOCHS
            current_cloud_epsilon = compute_epsilon(
                epoch,
                CLOUD_EPSILON_START,
                CLOUD_EPSILON_END,
                decay_epochs,
                schedule=epsilon_schedule,
            )
            decay_epochs = epsilon_decay_epochs or MACRO_EPSILON_DECAY_EPOCHS
            current_macro_epsilon = compute_epsilon(
                epoch,
                MACRO_EPSILON_START,
                MACRO_EPSILON_END,
                decay_epochs,
                schedule=epsilon_schedule,
            )
            decay_epochs = epsilon_decay_epochs or SMALL_EPSILON_DECAY_EPOCHS
            current_small_epsilon = compute_epsilon(
                epoch,
                SMALL_EPSILON_START,
                SMALL_EPSILON_END,
                decay_epochs,
                schedule=epsilon_schedule,
            )
            decay_epochs = epsilon_decay_epochs or UAV_EPSILON_DECAY_EPOCHS
            current_uav_epsilon = compute_epsilon(
                epoch,
                UAV_EPSILON_START,
                UAV_EPSILON_END,
                decay_epochs,
                schedule=epsilon_schedule,
            )

        cloud_eps_history.append(current_cloud_epsilon)
        macro_eps_history.append(current_macro_epsilon)
        small_eps_history.append(current_small_epsilon)
        uav_eps_history.append(current_uav_epsilon)

        num_slots = config.get("cloud_association_period", CLOUD_ASSOCIATION_PERIOD)
        epoch_total_throughput_mbps = 0.0
        epoch_user_rate_sum = [0.0] * len(user_list)
        epoch_user_satisfaction_sum = [0.0] * len(user_list)
        epoch_qos_sum = 0.0
        epoch_jain_sum = 0.0
        epoch_switch_count = 0

        epoch_macro_reward_sum = 0.0
        epoch_small_reward_sum = 0.0
        epoch_uav_reward_sum = 0.0
        epoch_macro_loss_sum = 0.0
        epoch_small_loss_sum = 0.0
        epoch_uav_loss_sum = 0.0
        epoch_macro_loss_count = 0
        epoch_small_loss_count = 0
        epoch_uav_loss_count = 0
        cloud_loss = 0.0
        cloud_reward_avg = 0.0

        for slot in range(num_slots):
            for user in user_list:
                user.calculate_next_step()

            initial_bs_loads = [0.0] * 9
            sat_load = (
                len(sat_bs_list[0].serv_user_list) / SAT_BS_CAPACITY
                if sat_bs_list
                else 0.0
            )
            initial_bs_loads[0] = sat_load
            for i, bs in enumerate(macro_bs_list):
                initial_bs_loads[i + 1] = len(bs.serv_user_list) / MACRO_BS_CAPACITY
            for i, bs in enumerate(small_bs_list):
                initial_bs_loads[i + 3] = len(bs.serv_user_list) / SMALL_BS_CAPACITY
            for i, bs in enumerate(uav_bs_list):
                initial_bs_loads[i + 7] = len(bs.serv_user_list) / UAV_BS_CAPACITY

            predicted_bs_loads = initial_bs_loads.copy()
            is_cloud_action_slot = slot == 0
            current_slot_experiences = []

            if is_cloud_action_slot:
                order_indices = _validate_sdm_order_indices(
                    build_sdm_order_indices(
                        len(user_list),
                        mode=sdm_order_mode,
                        seed=sdm_order_seed,
                        cloud_epoch=epoch,
                    ),
                    len(user_list),
                )
                current_cloud_actions = {}
                processed_user_keys = set()
                for user_idx in order_indices:
                    user = user_list[user_idx]
                    user_key = user.global_index
                    if user_key not in expected_user_keys or user_key in processed_user_keys:
                        raise RuntimeError(
                            "SDM cloud pass contains an unknown or duplicate user identifier."
                        )
                    processed_user_keys.add(user_key)
                    current_rsrp = []
                    for bs in sat_bs_list:
                        current_rsrp.append(
                            user.calculate_rsrp(
                                "curr",
                                bs,
                                SAT_FREQUENCY,
                                get_tx_power_dbm(
                                    SAT_TX_POWER, SAT_NUM_OF_SC, use_total_power
                                ),
                            )
                        )
                    for bs in macro_bs_list:
                        current_rsrp.append(
                            user.calculate_rsrp(
                                "curr",
                                bs,
                                MACRO_FREQUENCY,
                                get_tx_power_dbm(
                                    MACRO_TX_POWER,
                                    MACRO_NUM_OF_SC,
                                    use_total_power,
                                ),
                            )
                        )
                    for bs in small_bs_list:
                        current_rsrp.append(
                            user.calculate_rsrp(
                                "curr",
                                bs,
                                SMALL_FREQUENCY,
                                get_tx_power_dbm(
                                    SMALL_TX_POWER,
                                    SMALL_NUM_OF_SC,
                                    use_total_power,
                                ),
                            )
                        )
                    for bs in uav_bs_list:
                        current_rsrp.append(
                            user.calculate_rsrp(
                                "curr",
                                bs,
                                UAV_FREQUENCY,
                                get_tx_power_dbm(
                                    UAV_TX_POWER, UAV_NUM_OF_SC, use_total_power
                                ),
                            )
                        )

                    normalized_rsrp = [normalize_rsrp(r) for r in current_rsrp[1:]]
                    curr_state = (
                        normalized_rsrp
                        + list(predicted_bs_loads)
                        + [
                            1 if user.user_type == t else 0
                            for t in ["eMBB", "mMTC", "uRLLC"]
                        ]
                    )

                    cloud_action, _ = cloud_choose_action(
                        cloud_agent,
                        curr_state,
                        device,
                        epsilon=current_cloud_epsilon,
                        available_mask=cloud_action_available_mask,
                    )
                    if not _action_is_available(cloud_action, cloud_action_available_mask):
                        raise RuntimeError("Cloud chooser returned an unavailable action.")
                    current_cloud_actions[user_key] = cloud_action
                    user.serv_bs_id_by_cloud_agent = cloud_action

                    selected_bs_idx = int(cloud_action)
                    cap = get_bs_capacity(selected_bs_idx)

                    if use_predicted_loads and 0 <= selected_bs_idx < len(
                        predicted_bs_loads
                    ):
                        predicted_bs_loads[selected_bs_idx] = min(
                            predicted_bs_loads[selected_bs_idx] + 1.0 / cap, 1.0
                        )

                    cloud_reward = cloud_reward_calculation(
                        cloud_action,
                        user,
                        predicted_bs_loads,
                        rsrp_list=current_rsrp,
                        cloud_weights=cloud_weights,
                        norm_mode=reward_norm_mode,
                        use_required_rate=use_required_rate,
                    )

                    if user.previous_bs is not None:
                        if cloud_action != user.previous_bs:
                            cloud_reward["total"] -= switch_penalty
                            epoch_switch_count += 1
                        else:
                            cloud_reward["total"] += stickiness_bonus
                    user.previous_bs = cloud_action

                    next_rsrp = []
                    for bs in macro_bs_list:
                        next_rsrp.append(
                            user.calculate_rsrp(
                                "next",
                                bs,
                                MACRO_FREQUENCY,
                                get_tx_power_dbm(
                                    MACRO_TX_POWER,
                                    MACRO_NUM_OF_SC,
                                    use_total_power,
                                ),
                            )
                        )
                    for bs in small_bs_list:
                        next_rsrp.append(
                            user.calculate_rsrp(
                                "next",
                                bs,
                                SMALL_FREQUENCY,
                                get_tx_power_dbm(
                                    SMALL_TX_POWER,
                                    SMALL_NUM_OF_SC,
                                    use_total_power,
                                ),
                            )
                        )
                    for bs in uav_bs_list:
                        next_rsrp.append(
                            user.calculate_rsrp(
                                "next",
                                bs,
                                UAV_FREQUENCY,
                                get_tx_power_dbm(
                                    UAV_TX_POWER, UAV_NUM_OF_SC, use_total_power
                                ),
                            )
                        )

                    normalized_next_rsrp = [normalize_rsrp(r) for r in next_rsrp]
                    next_state = (
                        normalized_next_rsrp
                        + list(predicted_bs_loads)
                        + [
                            1 if user.user_type == t else 0
                            for t in ["eMBB", "mMTC", "uRLLC"]
                        ]
                    )

                    _append_cloud_experience(
                        current_slot_experiences,
                        curr_state,
                        cloud_action,
                        cloud_reward["total"],
                        next_state,
                        device,
                    )

                cloud_historical_actions = _commit_cloud_action_profile(
                    current_cloud_actions,
                    processed_user_keys,
                    expected_user_keys,
                    cloud_action_available_mask,
                )

                if training_enabled and current_slot_experiences:
                    max_priority = max(cloud_priorities) if cloud_priorities else 1.0
                    cloud_replay_buffer.extend(current_slot_experiences)
                    cloud_priorities.extend(
                        [max_priority] * len(current_slot_experiences)
                    )
                    if len(cloud_replay_buffer) > replay_capacity:
                        remove_count = len(cloud_replay_buffer) - replay_capacity
                        cloud_replay_buffer = cloud_replay_buffer[remove_count:]
                        cloud_priorities = cloud_priorities[remove_count:]

                if training_enabled and cloud_replay_buffer:
                    cloud_optimizer.zero_grad()
                    current_beta = min(
                        PRIORITY_BETA + epoch * PRIORITY_BETA_INCREMENT,
                        PRIORITY_BETA_MAX,
                    )
                    priorities = np.array(cloud_priorities, dtype=np.float32)
                    priorities = np.maximum(priorities, PRIORITY_EPSILON)
                    sampling_probs = priorities**PRIORITY_ALPHA
                    sampling_probs = sampling_probs / np.sum(sampling_probs)

                    batch_size = min(256, len(cloud_replay_buffer))
                    indices = np.random.choice(
                        len(cloud_replay_buffer),
                        batch_size,
                        p=sampling_probs,
                        replace=False,
                    )
                    sampled_experiences = [cloud_replay_buffer[i] for i in indices]
                    weights = (len(cloud_replay_buffer) * sampling_probs[indices]) ** (
                        -current_beta
                    )
                    weights = weights / np.max(weights)

                    td_errors = []
                    batch_loss = 0.0
                    for i, (state, action, reward, next_state) in enumerate(
                        sampled_experiences
                    ):
                        q_values = cloud_agent(state)
                        q_estimate = q_values[int(action)]
                        with torch.no_grad():
                            next_q_values = cloud_target(next_state)
                            if cloud_action_available_mask is None:
                                next_q_max = torch.max(next_q_values)
                            else:
                                next_q_max = _masked_cloud_q_max(
                                    next_q_values, cloud_action_available_mask
                                )
                            q_target = reward + CLOUD_GAMMA * next_q_max
                        td_error = abs(q_estimate.item() - q_target.item())
                        td_errors.append(td_error)
                        weighted_loss = (
                            torch.nn.functional.mse_loss(q_estimate, q_target)
                            * weights[i]
                        )
                        batch_loss += weighted_loss

                    batch_loss = batch_loss / batch_size
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(cloud_agent.parameters(), 1.0)
                    cloud_optimizer.step()
                    cloud_loss = batch_loss.item()
                    for idx, td_error in zip(indices, td_errors):
                        cloud_priorities[idx] = td_error + PRIORITY_EPSILON

                if current_slot_experiences:
                    cloud_reward_avg = sum(
                        exp[2] for exp in current_slot_experiences
                    ) / len(current_slot_experiences)
            else:
                for user in user_list:
                    user.serv_bs_id_by_cloud_agent = _read_historical_action(
                        cloud_historical_actions,
                        user.global_index,
                        cloud_action_available_mask,
                    )

            for bs in sat_bs_list + macro_bs_list + small_bs_list + uav_bs_list:
                bs.serv_user_list.clear()

            slot_connected_users_count = 0
            for user in user_list:
                bid = user.serv_bs_id_by_cloud_agent
                target_bs = get_target_bs(
                    bid, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
                )

                if target_bs:
                    cap = get_bs_capacity(bid)
                    if len(target_bs.serv_user_list) < cap:
                        user.id_in_serv_bs = len(target_bs.serv_user_list)
                        target_bs.serv_user_list.append(user)
                        slot_connected_users_count += 1

            slot_total_throughput = 0.0
            slot_user_satisfaction_list = [0.0] * len(user_list)
            slot_user_rate_list = [0.0] * len(user_list)
            slot_satisfied_users_count = 0

            slot_macro_reward_total = 0.0
            slot_macro_loss_total = 0.0
            slot_macro_loss_count = 0
            for bs in macro_bs_list:
                if not bs.serv_user_list:
                    continue
                roster, macro_state, active_count = build_eiap_state(
                    bs,
                    MACRO_BS_CAPACITY,
                    MACRO_FREQUENCY,
                    get_tx_power_dbm(MACRO_TX_POWER, MACRO_NUM_OF_SC, use_total_power),
                )
                action_list, q_est = bs_choose_action(
                    macro_bs_agent,
                    macro_state,
                    device,
                    current_macro_epsilon,
                    MACRO_NUM_OF_SC,
                    active_count,
                    MACRO_BS_CAPACITY,
                )
                macro_reward = macro_reward_calculation(
                    action_list,
                    roster,
                    MACRO_FREQUENCY,
                    MACRO_TX_POWER,
                    bs_object=bs,
                    edge_weights=edge_weights,
                    norm_mode=reward_norm_mode,
                    use_required_rate=use_required_rate,
                    use_sc_budget=use_sc_budget,
                    use_total_power=use_total_power,
                )
                slot_macro_reward_total += macro_reward

                remaining_sc = MACRO_NUM_OF_SC
                for i, user in enumerate(roster):
                    if i < len(action_list):
                        rsrp = user.calculate_rsrp(
                            "curr",
                            bs,
                            MACRO_FREQUENCY,
                            get_tx_power_dbm(
                                MACRO_TX_POWER,
                                MACRO_NUM_OF_SC,
                                use_total_power,
                            ),
                        )
                        alloc_sc = max(int(action_list[i][0]), 0)
                        if use_sc_budget:
                            alloc_sc = min(alloc_sc, remaining_sc)
                            remaining_sc -= alloc_sc
                        rate = calculate_rate_mbps(
                            rsrp,
                            alloc_sc,
                            MACRO_NUM_OF_SC,
                            MACRO_TOTAL_BW_HZ,
                            action_list[i][1],
                        )
                        slot_total_throughput += rate
                        slot_user_rate_list[user.global_index] = rate
                        req = (
                            EDGE_THR_REQ_MACRO_Mbps
                            if use_required_rate
                            else user.qos["rate"]
                        )
                        ratio = min(rate / req, 1.0)
                        slot_user_satisfaction_list[user.global_index] = ratio
                        if rate >= req:
                            slot_satisfied_users_count += 1

                macro_next_state = []
                for user in roster:
                    rsrp = user.calculate_rsrp(
                        "next",
                        bs,
                        MACRO_FREQUENCY,
                        get_tx_power_dbm(
                            MACRO_TX_POWER,
                            MACRO_NUM_OF_SC,
                            use_total_power,
                        ),
                    )
                    macro_next_state.append(
                        [normalize_rsrp(rsrp)]
                        + [
                            1 if user.user_type == t else 0
                            for t in ["eMBB", "mMTC", "uRLLC"]
                        ]
                    )
                if len(macro_next_state) < MACRO_BS_CAPACITY:
                    macro_next_state.extend(
                        [[0.0, 0, 0, 0]] * (MACRO_BS_CAPACITY - len(macro_next_state))
                    )

                if training_enabled:
                    with torch.no_grad():
                        q_max = torch.max(
                            macro_target(
                                torch.tensor(macro_next_state, dtype=torch.float32)
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .to(device)
                            )
                        )
                    q_target = torch.tensor(
                        [macro_reward + MACRO_GAMMA * q_max.item()],
                        dtype=torch.float32,
                    ).to(device)

                    if isinstance(q_est, torch.Tensor):
                        if q_est.numel() > 1:
                            q_est = q_est.mean()
                        q_est = q_est.reshape(1)
                    else:
                        q_est = torch.tensor(q_est, dtype=torch.float32).to(device)

                    macro_loss = torch.nn.functional.mse_loss(q_est, q_target)
                    macro_bs_optimizer.zero_grad()
                    macro_loss.backward()
                    torch.nn.utils.clip_grad_norm_(macro_bs_agent.parameters(), 1.0)
                    macro_bs_optimizer.step()
                    slot_macro_loss_total += macro_loss.item()
                    slot_macro_loss_count += 1

            slot_small_reward_total = 0.0
            slot_small_loss_total = 0.0
            slot_small_loss_count = 0
            for bs in small_bs_list:
                if not bs.serv_user_list:
                    continue
                roster, small_state, active_count = build_eiap_state(
                    bs,
                    SMALL_BS_CAPACITY,
                    SMALL_FREQUENCY,
                    get_tx_power_dbm(SMALL_TX_POWER, SMALL_NUM_OF_SC, use_total_power),
                )
                action_list, q_est = bs_choose_action(
                    small_bs_agent,
                    small_state,
                    device,
                    current_small_epsilon,
                    SMALL_NUM_OF_SC,
                    active_count,
                    SMALL_BS_CAPACITY,
                )
                small_reward = small_reward_calculation(
                    action_list,
                    roster,
                    SMALL_FREQUENCY,
                    SMALL_TX_POWER,
                    bs_object=bs,
                    edge_weights=edge_weights,
                    norm_mode=reward_norm_mode,
                    use_required_rate=use_required_rate,
                    use_sc_budget=use_sc_budget,
                    use_total_power=use_total_power,
                )
                slot_small_reward_total += small_reward

                remaining_sc = SMALL_NUM_OF_SC
                for i, user in enumerate(roster):
                    if i < len(action_list):
                        rsrp = user.calculate_rsrp(
                            "curr",
                            bs,
                            SMALL_FREQUENCY,
                            get_tx_power_dbm(
                                SMALL_TX_POWER,
                                SMALL_NUM_OF_SC,
                                use_total_power,
                            ),
                        )
                        alloc_sc = max(int(action_list[i][0]), 0)
                        if use_sc_budget:
                            alloc_sc = min(alloc_sc, remaining_sc)
                            remaining_sc -= alloc_sc
                        rate = calculate_rate_mbps(
                            rsrp,
                            alloc_sc,
                            SMALL_NUM_OF_SC,
                            SMALL_TOTAL_BW_HZ,
                            action_list[i][1],
                        )
                        slot_total_throughput += rate
                        slot_user_rate_list[user.global_index] = rate
                        req = (
                            EDGE_THR_REQ_SMALL_Mbps
                            if use_required_rate
                            else user.qos["rate"]
                        )
                        ratio = min(rate / req, 1.0)
                        slot_user_satisfaction_list[user.global_index] = ratio
                        if rate >= req:
                            slot_satisfied_users_count += 1

                small_next_state = []
                for user in roster:
                    rsrp = user.calculate_rsrp(
                        "next",
                        bs,
                        SMALL_FREQUENCY,
                        get_tx_power_dbm(
                            SMALL_TX_POWER,
                            SMALL_NUM_OF_SC,
                            use_total_power,
                        ),
                    )
                    small_next_state.append(
                        [normalize_rsrp(rsrp)]
                        + [
                            1 if user.user_type == t else 0
                            for t in ["eMBB", "mMTC", "uRLLC"]
                        ]
                    )
                if len(small_next_state) < SMALL_BS_CAPACITY:
                    small_next_state.extend(
                        [[0.0, 0, 0, 0]] * (SMALL_BS_CAPACITY - len(small_next_state))
                    )

                if training_enabled:
                    with torch.no_grad():
                        q_max = torch.max(
                            small_target(
                                torch.tensor(small_next_state, dtype=torch.float32)
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .to(device)
                            )
                        )
                    q_target = torch.tensor(
                        [small_reward + SMALL_GAMMA * q_max.item()],
                        dtype=torch.float32,
                    ).to(device)

                    if isinstance(q_est, torch.Tensor):
                        if q_est.numel() > 1:
                            q_est = q_est.mean()
                        q_est = q_est.reshape(1)
                    else:
                        q_est = torch.tensor(
                            [q_est], dtype=torch.float32, device=device
                        )

                    small_loss = torch.nn.functional.mse_loss(q_est, q_target)
                    small_bs_optimizer.zero_grad()
                    small_loss.backward()
                    torch.nn.utils.clip_grad_norm_(small_bs_agent.parameters(), 1.0)
                    small_bs_optimizer.step()
                    slot_small_loss_total += small_loss.item()
                    slot_small_loss_count += 1

            slot_uav_reward_total = 0.0
            slot_uav_loss_total = 0.0
            slot_uav_loss_count = 0
            for bs in uav_bs_list:
                if not bs.serv_user_list:
                    continue
                uav_state = [
                    bs.curr_x / AREA_SIZE_X,
                    bs.curr_y / AREA_SIZE_Y,
                    bs.curr_z / UAV_MAX_HEIGHT,
                ]
                for user in bs.serv_user_list:
                    uav_state.extend([user.x / AREA_SIZE_X, user.y / AREA_SIZE_Y])

                expected_len = UAV_BS_CAPACITY * 2 + 3
                if len(uav_state) < expected_len:
                    uav_state.extend([-1.0] * (expected_len - len(uav_state)))
                else:
                    uav_state = uav_state[:expected_len]

                action, q_est = uav_choose_action(
                    uav_bs_agent, uav_state, device, epsilon=current_uav_epsilon
                )
                if not freeze_uav:
                    bs.move(action)
                uav_reward = uav_reward_calculation(
                    bs.serv_user_list,
                    bs,
                    UAV_FREQUENCY,
                    UAV_TX_POWER,
                    edge_weights=edge_weights,
                    norm_mode=reward_norm_mode,
                    reward_gamma=uav_reward_gamma,
                    reward_scale=uav_reward_scale,
                    use_required_rate=use_required_rate,
                    use_sc_budget=use_sc_budget,
                    use_total_power=use_total_power,
                )
                slot_uav_reward_total += uav_reward

                connected = bs.serv_user_list
                if connected:
                    sc_per_user = UAV_NUM_OF_SC / len(connected)
                    remaining_sc = UAV_NUM_OF_SC
                    for user in connected:
                        rsrp = user.calculate_rsrp(
                            "curr",
                            bs,
                            UAV_FREQUENCY,
                            get_tx_power_dbm(
                                UAV_TX_POWER, UAV_NUM_OF_SC, use_total_power
                            ),
                        )
                        alloc_sc = max(int(sc_per_user), 0)
                        if use_sc_budget:
                            alloc_sc = min(alloc_sc, remaining_sc)
                            remaining_sc -= alloc_sc
                        rate = calculate_rate_mbps(
                            rsrp, alloc_sc, UAV_NUM_OF_SC, UAV_TOTAL_BW_HZ, 1.0
                        )
                        slot_total_throughput += rate
                        slot_user_rate_list[user.global_index] = rate
                        req = (
                            EDGE_THR_REQ_UAV_Mbps
                            if use_required_rate
                            else user.qos["rate"]
                        )
                        ratio = min(rate / req, 1.0)
                        slot_user_satisfaction_list[user.global_index] = ratio
                        if rate >= req:
                            slot_satisfied_users_count += 1

                uav_next_state = [
                    bs.curr_x / AREA_SIZE_X,
                    bs.curr_y / AREA_SIZE_Y,
                    bs.curr_z / UAV_MAX_HEIGHT,
                ]
                for user in bs.serv_user_list:
                    uav_next_state.extend(
                        [user.next_x / AREA_SIZE_X, user.next_y / AREA_SIZE_Y]
                    )
                if len(uav_next_state) < expected_len:
                    uav_next_state.extend([-1.0] * (expected_len - len(uav_next_state)))
                else:
                    uav_next_state = uav_next_state[:expected_len]

                if training_enabled:
                    with torch.no_grad():
                        q_max = torch.max(
                            uav_target(
                                torch.tensor(uav_next_state, dtype=torch.float32).to(
                                    device
                                )
                            )
                        )
                    q_target = torch.tensor(
                        [uav_reward + UAV_GAMMA * q_max.item()],
                        dtype=torch.float32,
                        device=device,
                    )
                    q_est = q_est.reshape(1)

                    uav_loss = torch.nn.functional.mse_loss(q_est, q_target)
                    uav_bs_optimizer.zero_grad()
                    uav_loss.backward()
                    torch.nn.utils.clip_grad_norm_(uav_bs_agent.parameters(), 2.0)
                    uav_bs_optimizer.step()
                    slot_uav_loss_total += uav_loss.item()
                    slot_uav_loss_count += 1

            for bs in sat_bs_list:
                if bs.serv_user_list:
                    sc_per_user = SAT_NUM_OF_SC / len(bs.serv_user_list)
                    remaining_sc = SAT_NUM_OF_SC
                    for user in bs.serv_user_list:
                        rsrp = user.calculate_rsrp(
                            "curr",
                            bs,
                            SAT_FREQUENCY,
                            get_tx_power_dbm(
                                SAT_TX_POWER, SAT_NUM_OF_SC, use_total_power
                            ),
                        )
                        alloc_sc = max(int(sc_per_user), 0)
                        if use_sc_budget:
                            alloc_sc = min(alloc_sc, remaining_sc)
                            remaining_sc -= alloc_sc
                        rate = calculate_rate_mbps(
                            rsrp, alloc_sc, SAT_NUM_OF_SC, SAT_TOTAL_BW_HZ, 1.0
                        )
                        slot_total_throughput += rate
                        slot_user_rate_list[user.global_index] = rate
                        req = (
                            SAT_THR_REQ_Mbps if use_required_rate else user.qos["rate"]
                        )
                        ratio = min(rate / req, 1.0)
                        slot_user_satisfaction_list[user.global_index] = ratio
                        if rate >= req:
                            slot_satisfied_users_count += 1

            slot_macro_reward_avg = slot_macro_reward_total / max(len(macro_bs_list), 1)
            slot_small_reward_avg = slot_small_reward_total / max(len(small_bs_list), 1)
            slot_uav_reward_avg = slot_uav_reward_total / max(len(uav_bs_list), 1)

            epoch_macro_reward_sum += slot_macro_reward_avg
            epoch_small_reward_sum += slot_small_reward_avg
            epoch_uav_reward_sum += slot_uav_reward_avg
            epoch_macro_loss_sum += slot_macro_loss_total
            epoch_small_loss_sum += slot_small_loss_total
            epoch_uav_loss_sum += slot_uav_loss_total
            epoch_macro_loss_count += slot_macro_loss_count
            epoch_small_loss_count += slot_small_loss_count
            epoch_uav_loss_count += slot_uav_loss_count

            injected_cloud_reward = cloud_reward_avg if is_cloud_action_slot else 0.0

            slot_qos_rate = (
                slot_satisfied_users_count / slot_connected_users_count
                if slot_connected_users_count > 0
                else 0.0
            )
            slot_jain = calculate_jain_fairness(slot_user_satisfaction_list)

            epoch_total_throughput_mbps += slot_total_throughput
            epoch_qos_sum += slot_qos_rate
            epoch_jain_sum += slot_jain
            for idx in range(len(user_list)):
                epoch_user_rate_sum[idx] += slot_user_rate_list[idx]
                epoch_user_satisfaction_sum[idx] += slot_user_satisfaction_list[idx]

            if len(uav_bs_list) > 0:
                uav_trajectories[0].append(
                    (
                        uav_bs_list[0].curr_x,
                        uav_bs_list[0].curr_y,
                        uav_bs_list[0].curr_z,
                    )
                )
            if len(uav_bs_list) > 1:
                uav_trajectories[1].append(
                    (
                        uav_bs_list[1].curr_x,
                        uav_bs_list[1].curr_y,
                        uav_bs_list[1].curr_z,
                    )
                )

            for user in user_list:
                user.move()

        macro_reward_avg = epoch_macro_reward_sum / max(num_slots, 1)
        small_reward_avg = epoch_small_reward_sum / max(num_slots, 1)
        uav_reward_avg = epoch_uav_reward_sum / max(num_slots, 1)

        injected_cloud_reward = cloud_reward_avg
        reward_sum_history.append(
            macro_reward_avg + small_reward_avg + uav_reward_avg + injected_cloud_reward
        )

        global_reward = calculate_global_reward(
            injected_cloud_reward, macro_reward_avg, small_reward_avg, uav_reward_avg
        )

        qos_rate = epoch_qos_sum / max(num_slots, 1)
        jain_fairness = epoch_jain_sum / max(num_slots, 1)

        epoch_user_rate_list = [v / max(num_slots, 1) for v in epoch_user_rate_sum]
        epoch_user_satisfaction_list = [
            v / max(num_slots, 1) for v in epoch_user_satisfaction_sum
        ]

        macro_loss_avg = (
            epoch_macro_loss_sum / epoch_macro_loss_count
            if epoch_macro_loss_count > 0
            else 0.0
        )
        small_loss_avg = (
            epoch_small_loss_sum / epoch_small_loss_count
            if epoch_small_loss_count > 0
            else 0.0
        )
        uav_loss_avg = (
            epoch_uav_loss_sum / epoch_uav_loss_count
            if epoch_uav_loss_count > 0
            else 0.0
        )

        cloud_losses.append(cloud_loss)
        macro_losses.append(macro_loss_avg)
        small_losses.append(small_loss_avg)
        uav_losses.append(uav_loss_avg)
        cloud_rewards.append(cloud_reward_avg)
        macro_rewards.append(macro_reward_avg)
        small_rewards.append(small_reward_avg)
        uav_rewards.append(uav_reward_avg)
        throughput_history.append(epoch_total_throughput_mbps / max(num_slots, 1))
        qos_history.append(qos_rate)
        jain_history.append(jain_fairness)

        bs_loads = [0.0] * 9
        if sat_bs_list:
            bs_loads[0] = len(sat_bs_list[0].serv_user_list) / SAT_BS_CAPACITY
        for i, bs in enumerate(macro_bs_list):
            bs_loads[i + 1] = len(bs.serv_user_list) / MACRO_BS_CAPACITY
        for i, bs in enumerate(small_bs_list):
            bs_loads[i + 3] = len(bs.serv_user_list) / SMALL_BS_CAPACITY
        for i, bs in enumerate(uav_bs_list):
            bs_loads[i + 7] = len(bs.serv_user_list) / UAV_BS_CAPACITY

        user_rate_history.append(epoch_user_rate_list)
        user_bs_history.append([user.serv_bs_id_by_cloud_agent for user in user_list])
        bs_load_history.append(bs_loads)
        switch_count_history.append(epoch_switch_count)

        if epoch % 50 == 0:
            debug_print(
                f"Epoch {epoch} | Global reward {global_reward:.4f} | Throughput {epoch_total_throughput_mbps / max(num_slots, 1):.1f} "
                f"| QoS {qos_rate:.2f} | Jain {jain_fairness:.4f}"
            )

        if convergence_check and (epoch + 1) >= min_convergence_epochs:
            if len(reward_sum_history) >= convergence_window:
                recent_slice = slice(-convergence_window, None)
                reward_std = float(np.std(reward_sum_history[recent_slice]))
                thr_std = float(np.std(throughput_history[recent_slice]))
                qos_std = float(np.std(qos_history[recent_slice]))
                jain_std = float(np.std(jain_history[recent_slice]))

                reward_peak = max(reward_sum_history) if reward_sum_history else 0.0
                thr_peak = max(throughput_history) if throughput_history else 0.0
                qos_peak = max(qos_history) if qos_history else 0.0
                jain_peak = max(jain_history) if jain_history else 0.0

                reward_ok = (
                    reward_peak > 0
                    and reward_std <= reward_peak * convergence_std_ratio
                )
                thr_ok = thr_peak > 0 and thr_std <= thr_peak * convergence_std_ratio
                qos_ok = qos_peak > 0 and qos_std <= qos_peak * convergence_std_ratio
                jain_ok = (
                    jain_peak > 0 and jain_std <= jain_peak * convergence_std_ratio
                )

                if reward_ok and thr_ok and qos_ok and jain_ok:
                    debug_print(
                        "Converged: stds are below threshold for reward/throughput/QoS/Jain."
                    )
                    break

        if training_enabled:
            for t, p in zip(cloud_target.parameters(), cloud_agent.parameters()):
                t.data.copy_(
                    t.data * (1.0 - TAU_SOFT_UPDATE) + p.data * TAU_SOFT_UPDATE
                )
            for t, p in zip(macro_target.parameters(), macro_bs_agent.parameters()):
                t.data.copy_(
                    t.data * (1.0 - TAU_SOFT_UPDATE) + p.data * TAU_SOFT_UPDATE
                )
            for t, p in zip(small_target.parameters(), small_bs_agent.parameters()):
                t.data.copy_(
                    t.data * (1.0 - TAU_SOFT_UPDATE) + p.data * TAU_SOFT_UPDATE
                )
            for t, p in zip(uav_target.parameters(), uav_bs_agent.parameters()):
                t.data.copy_(
                    t.data * (1.0 - TAU_SOFT_UPDATE) + p.data * TAU_SOFT_UPDATE
                )

    losses = {
        "cloud": cloud_losses,
        "macro": macro_losses,
        "small": small_losses,
        "uav": uav_losses,
    }
    rewards = {
        "cloud": cloud_rewards,
        "macro": macro_rewards,
        "small": small_rewards,
        "uav": uav_rewards,
    }
    eps_histories = {
        "cloud": cloud_eps_history,
        "macro": macro_eps_history,
        "small": small_eps_history,
        "uav": uav_eps_history,
    }
    return (
        losses,
        rewards,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        eps_histories,
        user_rate_history,
        bs_load_history,
        switch_count_history,
    )


def load_full_exploration_metrics(logs_dir):
    full_path = os.path.join(logs_dir, "full_exploration_metrics.csv")
    data = load_csv_matrix(full_path, expect_cols=5)
    if data is None:
        return None
    return {
        "reward_sum": data[:, 1].tolist(),
        "throughput": data[:, 2].tolist(),
        "qos": data[:, 3].tolist(),
        "jain": data[:, 4].tolist(),
    }


def run_full_exploration(device):
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()

    config = {
        "max_epochs": MAX_EPOCHS,
        "convergence_check": False,
    }
    (
        losses,
        rewards,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        eps_histories,
        _,
        _,
        _,
    ) = rl(
        sat_bs_list,
        macro_bs_list,
        small_bs_list,
        uav_bs_list,
        user_list,
        device,
        epsilon_mode="full_exploration",
        config=config,
    )

    save_system_metrics(
        logs_dir,
        reward_sum_history,
        throughput_history,
        qos_history,
        jain_history,
        "full_exploration_metrics.csv",
    )


def run_current_model(device):
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()

    config = {
        "epsilon_schedule": "exp2",
        "convergence_check": True,
        "convergence_window": 1000,
        "convergence_std_ratio": 0.05,
        "min_convergence_epochs": 2000,
    }

    (
        losses,
        rewards,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        eps_histories,
        _,
        _,
        _,
    ) = rl(
        sat_bs_list,
        macro_bs_list,
        small_bs_list,
        uav_bs_list,
        user_list,
        device,
        epsilon_mode="normal",
        config=config,
    )

    save_agent_losses(logs_dir, losses)
    save_agent_rewards(logs_dir, rewards)
    save_system_metrics(
        logs_dir,
        reward_sum_history,
        throughput_history,
        qos_history,
        jain_history,
        "system_metrics.csv",
    )

    full_metrics = load_full_exploration_metrics(logs_dir)
    if full_metrics is None:
        print("Missing full exploration metrics. Run option 1 first.")

    plot_training_summary(
        losses,
        rewards,
        reward_sum_history,
        throughput_history,
        qos_history,
        jain_history,
        eps_histories,
        full_exploration_metrics=full_metrics,
        show=True,
        filename=os.path.join(logs_dir, "training_summary.png"),
    )


def run_ablation_batch(device):
    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    ablation_dir = os.path.join(logs_root, "ablations")
    os.makedirs(ablation_dir, exist_ok=True)

    base_cloud = dict(CLOUD_REWARD_WEIGHTS)
    base_edge = dict(EDGE_REWARD_WEIGHTS)
    no_qos_cloud = renormalize_weights({**base_cloud, "qos": 0.0})
    no_qos_edge = renormalize_weights({**base_edge, "qos": 0.0})
    no_thr_cloud = renormalize_weights({**base_cloud, "thr": 0.0})
    no_thr_edge = renormalize_weights({**base_edge, "thr": 0.0})
    no_load_cap_cloud = renormalize_weights({**base_cloud, "load": 0.0, "cap": 0.0})
    no_fair_edge = renormalize_weights({**base_edge, "fair": 0.0})

    runs = [
        ("baseline", {}),
        (
            "A1_no_qos",
            {"cloud_weights": no_qos_cloud, "edge_weights": no_qos_edge},
        ),
        (
            "A2_no_thr",
            {"cloud_weights": no_thr_cloud, "edge_weights": no_thr_edge},
        ),
        ("A3_no_load_cap", {"cloud_weights": no_load_cap_cloud}),
        (
            "A4_no_stability",
            {"switch_penalty": 0.0, "stickiness_bonus": 0.0},
        ),
        ("A5_no_fair", {"edge_weights": no_fair_edge}),
    ]

    full_run_id = "full_exploration"
    run_results = {}

    for run_id, cfg in runs:
        print(f"[Ablation] {run_id} start")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
        user_list = define_users()

        run_cfg = {
            "epsilon_schedule": "exp2",
            "convergence_check": True,
            "convergence_window": 1000,
            "convergence_std_ratio": 0.05,
            "min_convergence_epochs": 2000,
        }
        run_cfg.update(cfg)

        (
            losses,
            rewards,
            throughput_history,
            reward_sum_history,
            qos_history,
            jain_history,
            eps_histories,
            user_rate_history,
            bs_load_history,
            switch_count_history,
        ) = rl(
            sat_bs_list,
            macro_bs_list,
            small_bs_list,
            uav_bs_list,
            user_list,
            device,
            epsilon_mode="normal",
            config=run_cfg,
        )

        run_results[run_id] = {
            "reward_sum": reward_sum_history,
            "throughput": throughput_history,
            "qos": qos_history,
            "jain": jain_history,
            "user_rate": user_rate_history,
            "bs_load": bs_load_history,
            "switch_count": switch_count_history,
            "user_count": len(user_list),
        }
        print(f"[Ablation] {run_id} done")

    print(f"[Ablation] {full_run_id} start")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()

    (
        _,
        _,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        _,
        user_rate_history,
        bs_load_history,
        switch_count_history,
    ) = rl(
        sat_bs_list,
        macro_bs_list,
        small_bs_list,
        uav_bs_list,
        user_list,
        device,
        epsilon_mode="full_exploration",
        config={"max_epochs": MAX_EPOCHS, "convergence_check": False},
    )

    run_results[full_run_id] = {
        "reward_sum": reward_sum_history,
        "throughput": throughput_history,
        "qos": qos_history,
        "jain": jain_history,
        "user_rate": user_rate_history,
        "bs_load": bs_load_history,
        "switch_count": switch_count_history,
        "user_count": len(user_list),
    }
    print(f"[Ablation] {full_run_id} done")

    fig1_rows = []
    for run_id in ["baseline", full_run_id]:
        metrics = run_results.get(run_id)
        if not metrics:
            continue
        reward_ma = moving_average_full(metrics["reward_sum"], MOVING_AVG_WINDOW)
        thr_ma = moving_average_full(metrics["throughput"], MOVING_AVG_WINDOW)
        qos_ma = moving_average_full(metrics["qos"], MOVING_AVG_WINDOW)
        jain_ma = moving_average_full(metrics["jain"], MOVING_AVG_WINDOW)
        for i in range(len(metrics["reward_sum"])):
            fig1_rows.append(
                [
                    run_id,
                    i,
                    metrics["reward_sum"][i],
                    metrics["throughput"][i],
                    metrics["qos"][i],
                    metrics["jain"][i],
                    reward_ma[i],
                    thr_ma[i],
                    qos_ma[i],
                    jain_ma[i],
                ]
            )
    write_csv(
        os.path.join(ablation_dir, "fig1_kpi.csv"),
        [
            "run_id",
            "epoch",
            "reward_sum",
            "throughput",
            "qos",
            "jain",
            "reward_sum_ma",
            "throughput_ma",
            "qos_ma",
            "jain_ma",
        ],
        fig1_rows,
    )

    fig2_rows = []
    fig2_kpis = ["reward_sum", "throughput", "qos", "jain"]
    for run_id in ["baseline", "A1_no_qos", "A2_no_thr", "A3_no_load_cap"]:
        metrics = run_results.get(run_id)
        if not metrics:
            continue
        for kpi in fig2_kpis:
            fig2_rows.append([run_id, kpi, last_fraction_mean(metrics[kpi], 0.1)])
    write_csv(
        os.path.join(ablation_dir, "fig2_kpi_summary.csv"),
        ["run_id", "kpi", "mean_last_10pct"],
        fig2_rows,
    )

    fig3_rows = []
    for run_id in ["baseline", "A4_no_stability", "A5_no_fair"]:
        metrics = run_results.get(run_id)
        if not metrics:
            continue
        user_count = metrics["user_count"]
        switch_series = []
        for i in range(len(metrics["reward_sum"])):
            switch_series.append(
                metrics["switch_count"][i] / user_count if user_count > 0 else 0.0
            )
        switch_ma = moving_average_full(switch_series, MOVING_AVG_WINDOW)
        qos_ma = moving_average_full(metrics["qos"], MOVING_AVG_WINDOW)
        jain_ma = moving_average_full(metrics["jain"], MOVING_AVG_WINDOW)
        for i in range(len(metrics["reward_sum"])):
            switch_ratio = switch_series[i]
            fig3_rows.append(
                [
                    run_id,
                    i,
                    switch_ratio,
                    metrics["qos"][i],
                    metrics["jain"][i],
                    switch_ma[i],
                    qos_ma[i],
                    jain_ma[i],
                ]
            )
    write_csv(
        os.path.join(ablation_dir, "fig3_stability.csv"),
        [
            "run_id",
            "epoch",
            "switch_ratio",
            "qos",
            "jain",
            "switch_ratio_ma",
            "qos_ma",
            "jain_ma",
        ],
        fig3_rows,
    )

    fig4_rows = []
    corr_labels = [
        "reward_sum",
        "throughput",
        "qos",
        "jain",
        "rate_cv",
        "load_cv",
        "switch_ratio",
    ]
    for run_id in [
        "baseline",
        "A1_no_qos",
        "A2_no_thr",
        "A3_no_load_cap",
        "A4_no_stability",
        "A5_no_fair",
    ]:
        metrics = run_results.get(run_id)
        if not metrics:
            continue
        user_rate = metrics["user_rate"]
        bs_load = metrics["bs_load"]
        switch_count = metrics["switch_count"]
        epoch_len = min(
            len(metrics["reward_sum"]),
            len(user_rate),
            len(bs_load),
            len(switch_count),
        )
        rate_cv = []
        load_cv = []
        switch_ratio = []
        for i in range(epoch_len):
            rate_cv.append(compute_rate_cv(user_rate[i]))
            load_cv.append(compute_load_cv(bs_load[i]))
            if metrics["user_count"] > 0:
                switch_ratio.append(switch_count[i] / metrics["user_count"])
            else:
                switch_ratio.append(0.0)
        X = np.column_stack(
            [
                metrics["reward_sum"][:epoch_len],
                metrics["throughput"][:epoch_len],
                metrics["qos"][:epoch_len],
                metrics["jain"][:epoch_len],
                rate_cv,
                load_cv,
                switch_ratio,
            ]
        )
        corr = np.corrcoef(X, rowvar=False)
        for i, row_label in enumerate(corr_labels):
            for j, col_label in enumerate(corr_labels):
                fig4_rows.append([run_id, row_label, col_label, corr[i, j]])
    write_csv(
        os.path.join(ablation_dir, "fig4_correlation.csv"),
        ["run_id", "row", "col", "value"],
        fig4_rows,
    )

    fig5_rows = []
    for run_id, metrics in run_results.items():
        for i in range(len(metrics["reward_sum"])):
            fig5_rows.append(
                [
                    run_id,
                    i,
                    metrics["reward_sum"][i],
                    metrics["throughput"][i],
                    metrics["qos"][i],
                    metrics["jain"][i],
                ]
            )
    write_csv(
        os.path.join(ablation_dir, "fig5_kpi_box.csv"),
        ["run_id", "epoch", "reward_sum", "throughput", "qos", "jain"],
        fig5_rows,
    )


def run_plotting(device):
    ablation_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "ablations"
    )
    os.makedirs(ablation_dir, exist_ok=True)

    fig1 = load_csv_rows(os.path.join(ablation_dir, "fig1_kpi.csv"))
    if fig1 is not None:
        _, rows = fig1
        run_ids = sorted({row[0] for row in rows})
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        titles = ["reward_sum", "throughput", "qos", "jain"]
        for idx, title in enumerate(titles):
            ax = axes[idx // 2, idx % 2]
            for run_id in run_ids:
                series = [
                    (int(row[1]), float(row[idx + 6]))
                    for row in rows
                    if row[0] == run_id
                ]
                if not series:
                    continue
                xs, ys = zip(*series)
                ax.plot(xs, ys, label=run_id)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_dir, "fig1_kpi.png"), dpi=300)

    fig2 = load_csv_rows(os.path.join(ablation_dir, "fig2_kpi_summary.csv"))
    if fig2 is not None:
        _, rows = fig2
        run_ids = ["baseline", "A1_no_qos", "A2_no_thr", "A3_no_load_cap"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        titles = ["reward_sum", "throughput", "qos", "jain"]
        for idx, title in enumerate(titles):
            ax = axes[idx // 2, idx % 2]
            values = []
            for run_id in run_ids:
                value = next(
                    (
                        float(row[2])
                        for row in rows
                        if row[0] == run_id and row[1] == title
                    ),
                    0.0,
                )
                values.append(value)
            ax.bar(run_ids, values)
            ax.set_title(f"{title} (last 10% mean)")
            ax.set_xlabel("run_id")
            ax.grid(True, axis="y")
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_dir, "fig2_kpi.png"), dpi=300)

    fig3 = load_csv_rows(os.path.join(ablation_dir, "fig3_stability.csv"))
    if fig3 is not None:
        _, rows = fig3
        run_ids = sorted({row[0] for row in rows})
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["switch_ratio", "qos", "jain"]
        for idx, title in enumerate(titles):
            ax = axes[idx]
            for run_id in run_ids:
                series = [
                    (int(row[1]), float(row[idx + 5]))
                    for row in rows
                    if row[0] == run_id
                ]
                if not series:
                    continue
                xs, ys = zip(*series)
                ax.plot(xs, ys, label=run_id)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_dir, "fig3_stability.png"), dpi=300)

    fig4 = load_csv_rows(os.path.join(ablation_dir, "fig4_correlation.csv"))
    if fig4 is not None:
        _, data_rows = fig4
        run_ids = sorted({row[0] for row in data_rows})
        labels = [
            "reward_sum",
            "throughput",
            "qos",
            "jain",
            "rate_cv",
            "load_cv",
            "switch_ratio",
        ]
        n = len(run_ids)
        cols = int(np.ceil(np.sqrt(n)))
        grid_rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            grid_rows,
            cols,
            figsize=(4 * cols, 4 * grid_rows),
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for idx, run_id in enumerate(run_ids):
            ax = axes[idx // cols, idx % cols]
            mat = np.full((len(labels), len(labels)), np.nan)
            for row in data_rows:
                if row[0] != run_id:
                    continue
                r = labels.index(row[1])
                c = labels.index(row[2])
                mat[r, c] = float(row[3])
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_title(str(run_id))
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        fig.savefig(os.path.join(ablation_dir, "fig4_correlation.png"), dpi=300)

    fig5 = load_csv_rows(os.path.join(ablation_dir, "fig5_kpi_box.csv"))
    if fig5 is not None:
        _, rows = fig5
        run_ids = sorted({row[0] for row in rows})
        kpi_titles = ["reward_sum", "throughput", "qos", "jain"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, title in enumerate(kpi_titles):
            ax = axes[idx // 2, idx % 2]
            data_by_run = []
            for run_id in run_ids:
                values = [float(row[idx + 2]) for row in rows if row[0] == run_id]
                data_by_run.append(values)
            ax.boxplot(data_by_run, tick_labels=run_ids, showfliers=False)
            ax.set_title(title)
            ax.set_xlabel("run_id")
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_dir, "fig5_kpi_box.png"), dpi=300)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_parameters()

    mode_choice = input(
        "Select mode: 1=full exploration, 2=normal (default 2), 3=run ablations, 4=generate figures: "
    ).strip()

    if mode_choice == "1":
        run_full_exploration(device)
        sys.exit(0)

    if mode_choice == "3":
        run_ablation_batch(device)
        sys.exit(0)

    if mode_choice == "4":
        run_plotting(device)
        sys.exit(0)

    run_current_model(device)
