# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Parameters import (
    CLOUD_ASSOCIATION_PERIOD,
    CLOUD_REWARD_WEIGHTS,
    CLOUD_SWITCH_PENALTY,
    CLOUD_STICKINESS_BONUS,
    EDGE_REWARD_WEIGHTS,
    EDGE_THR_REQ_MACRO_Mbps,
    EDGE_THR_REQ_SMALL_Mbps,
    EDGE_THR_REQ_UAV_Mbps,
    MACRO_BS_CAPACITY,
    MACRO_FREQUENCY,
    MACRO_NUM_OF_SC,
    MACRO_TOTAL_BW_HZ,
    MACRO_TX_POWER,
    MAX_EPOCHS,
    SAT_BS_CAPACITY,
    SAT_FREQUENCY,
    SAT_NUM_OF_SC,
    SAT_THR_REQ_Mbps,
    SAT_TOTAL_BW_HZ,
    SAT_TX_POWER,
    SMALL_BS_CAPACITY,
    SMALL_FREQUENCY,
    SMALL_NUM_OF_SC,
    SMALL_TOTAL_BW_HZ,
    SMALL_TX_POWER,
    UAV_BS_CAPACITY,
    UAV_FREQUENCY,
    UAV_NUM_OF_SC,
    UAV_TOTAL_BW_HZ,
    UAV_TX_POWER,
)
from RewardCalculator import (
    calculate_jain_index_helper as calculate_jain_fairness,
    calculate_rate_mbps_helper as calculate_rate_mbps,
    cloud_reward_calculation,
    macro_reward_calculation,
    small_reward_calculation,
    uav_reward_calculation,
)
from SC_RL_main import get_tx_power_dbm, normalize_rsrp, rl
from Topology import define_topology, define_users


REPO_DIR = Path(__file__).resolve().parent
LOG_ROOT = REPO_DIR / "logs" / "revision_round1" / "group1_baselines"
PLOT_ROOT = REPO_DIR / "plots" / "revision_round1" / "group1_baselines"
METHOD_CONFIG_DIR = LOG_ROOT / "method_configs"
RUN_DIR = LOG_ROOT / "runs"

REQUIRED_KPIS = [
    "reward_sum",
    "throughput_mbps",
    "qos_satisfaction",
    "jain_fairness",
    "switch_ratio",
]
METRIC_HEADER = REQUIRED_KPIS + [
    "cloud_loss",
    "macro_loss",
    "small_loss",
    "uav_loss",
]


def rel(path):
    if not path:
        return ""
    return str(Path(path).resolve().relative_to(REPO_DIR))


def ensure_dirs(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "method_configs").mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def node_specs(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list):
    specs = []
    specs.append(
        {
            "bid": 0,
            "bs": sat_bs_list[0],
            "capacity": SAT_BS_CAPACITY,
            "frequency": SAT_FREQUENCY,
            "tx_power": SAT_TX_POWER,
            "total_sc": SAT_NUM_OF_SC,
            "total_bw": SAT_TOTAL_BW_HZ,
            "req": SAT_THR_REQ_Mbps,
            "tier": "sat",
        }
    )
    for idx, bs in enumerate(macro_bs_list, start=1):
        specs.append(
            {
                "bid": idx,
                "bs": bs,
                "capacity": MACRO_BS_CAPACITY,
                "frequency": MACRO_FREQUENCY,
                "tx_power": MACRO_TX_POWER,
                "total_sc": MACRO_NUM_OF_SC,
                "total_bw": MACRO_TOTAL_BW_HZ,
                "req": EDGE_THR_REQ_MACRO_Mbps,
                "tier": "macro",
            }
        )
    for idx, bs in enumerate(small_bs_list, start=3):
        specs.append(
            {
                "bid": idx,
                "bs": bs,
                "capacity": SMALL_BS_CAPACITY,
                "frequency": SMALL_FREQUENCY,
                "tx_power": SMALL_TX_POWER,
                "total_sc": SMALL_NUM_OF_SC,
                "total_bw": SMALL_TOTAL_BW_HZ,
                "req": EDGE_THR_REQ_SMALL_Mbps,
                "tier": "small",
            }
        )
    for idx, bs in enumerate(uav_bs_list, start=7):
        specs.append(
            {
                "bid": idx,
                "bs": bs,
                "capacity": UAV_BS_CAPACITY,
                "frequency": UAV_FREQUENCY,
                "tx_power": UAV_TX_POWER,
                "total_sc": UAV_NUM_OF_SC,
                "total_bw": UAV_TOTAL_BW_HZ,
                "req": EDGE_THR_REQ_UAV_Mbps,
                "tier": "uav",
            }
        )
    return specs


def equal_action_list(active_count, total_sc, capacity, split_power=True):
    actions = []
    if active_count <= 0:
        return [(0, 0.0) for _ in range(capacity)]
    base_sc = total_sc // active_count
    remainder = total_sc - base_sc * active_count
    power = (1.0 / active_count) if split_power else 1.0
    for idx in range(capacity):
        if idx < active_count:
            alloc_sc = base_sc + (1 if idx < remainder else 0)
            actions.append((alloc_sc, power))
        else:
            actions.append((0, 0.0))
    return actions


def clear_serving_lists(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list):
    for bs in sat_bs_list + macro_bs_list + small_bs_list + uav_bs_list:
        bs.serv_user_list.clear()


def populate_serving_lists(
    user_list, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
):
    clear_serving_lists(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list)
    specs_by_bid = {
        spec["bid"]: spec
        for spec in node_specs(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list)
    }
    connected = 0
    for user in sorted(user_list, key=lambda u: u.global_index):
        bid = user.serv_bs_id_by_cloud_agent
        spec = specs_by_bid.get(bid)
        if spec is None:
            continue
        bs = spec["bs"]
        if len(bs.serv_user_list) >= spec["capacity"]:
            continue
        user.id_in_serv_bs = len(bs.serv_user_list)
        bs.serv_user_list.append(user)
        connected += 1
    return connected


def current_rsrp_list(user, specs):
    values = []
    for spec in specs:
        tx_dbm = get_tx_power_dbm(spec["tx_power"], spec["total_sc"], False)
        values.append(user.calculate_rsrp("curr", spec["bs"], spec["frequency"], tx_dbm))
    return values


def choose_greedy_association(user_list, specs, lambda_load):
    assigned_counts = {spec["bid"]: 0 for spec in specs}
    switch_count = 0
    cloud_reward_values = []
    for user in sorted(user_list, key=lambda u: u.global_index):
        rsrp_values = current_rsrp_list(user, specs)
        best_bid = None
        best_score = None
        for spec, rsrp in zip(specs, rsrp_values):
            bid = spec["bid"]
            capacity = max(spec["capacity"], 1)
            if assigned_counts[bid] >= capacity:
                continue
            normalized_load = assigned_counts[bid] / capacity
            score = float(normalize_rsrp(rsrp) - lambda_load * normalized_load)
            if best_score is None or score > best_score or (
                score == best_score and bid < best_bid
            ):
                best_score = score
                best_bid = bid
        if best_bid is None:
            best_bid = 0
        if user.previous_bs is not None and best_bid != user.previous_bs:
            switch_count += 1
        user.serv_bs_id_by_cloud_agent = best_bid
        user.previous_bs = best_bid
        assigned_counts[best_bid] += 1
        predicted_loads = [
            assigned_counts[spec["bid"]] / max(spec["capacity"], 1) for spec in specs
        ]
        cloud_reward = cloud_reward_calculation(
            best_bid,
            user,
            predicted_loads,
            rsrp_list=rsrp_values,
            cloud_weights=CLOUD_REWARD_WEIGHTS,
            use_required_rate=True,
        )
        cloud_reward_values.append(cloud_reward["total"])
    cloud_reward_avg = (
        float(np.mean(cloud_reward_values)) if cloud_reward_values else 0.0
    )
    historical_actions = {
        user.global_index: user.serv_bs_id_by_cloud_agent for user in user_list
    }
    return historical_actions, switch_count, cloud_reward_avg


def apply_historical_association(user_list, historical_actions):
    for user in user_list:
        user.serv_bs_id_by_cloud_agent = historical_actions.get(user.global_index, 0)


def rate_for_user(user, spec, alloc_sc, power_split):
    tx_dbm = get_tx_power_dbm(spec["tx_power"], spec["total_sc"], False)
    rsrp = user.calculate_rsrp("curr", spec["bs"], spec["frequency"], tx_dbm)
    return calculate_rate_mbps(
        rsrp, alloc_sc, spec["total_sc"], spec["total_bw"], power_split
    )


def evaluate_greedy_slot(user_list, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list):
    slot_connected = populate_serving_lists(
        user_list, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
    )
    slot_total_throughput = 0.0
    slot_satisfaction = [0.0] * len(user_list)
    slot_rates = [0.0] * len(user_list)
    satisfied = 0

    macro_reward_total = 0.0
    for bs in macro_bs_list:
        active = len(bs.serv_user_list)
        if active <= 0:
            continue
        actions = equal_action_list(active, MACRO_NUM_OF_SC, MACRO_BS_CAPACITY)
        macro_reward_total += macro_reward_calculation(
            actions,
            bs.serv_user_list,
            MACRO_FREQUENCY,
            MACRO_TX_POWER,
            bs_object=bs,
            edge_weights=EDGE_REWARD_WEIGHTS,
            use_required_rate=True,
            use_sc_budget=True,
            use_total_power=False,
        )
        spec = {
            "bs": bs,
            "frequency": MACRO_FREQUENCY,
            "tx_power": MACRO_TX_POWER,
            "total_sc": MACRO_NUM_OF_SC,
            "total_bw": MACRO_TOTAL_BW_HZ,
            "req": EDGE_THR_REQ_MACRO_Mbps,
        }
        for idx, user in enumerate(bs.serv_user_list):
            alloc_sc, power_split = actions[idx]
            rate = rate_for_user(user, spec, alloc_sc, power_split)
            slot_total_throughput += rate
            slot_rates[user.global_index] = rate
            ratio = min(rate / spec["req"], 1.0)
            slot_satisfaction[user.global_index] = ratio
            if rate >= spec["req"]:
                satisfied += 1

    small_reward_total = 0.0
    for bs in small_bs_list:
        active = len(bs.serv_user_list)
        if active <= 0:
            continue
        actions = equal_action_list(active, SMALL_NUM_OF_SC, SMALL_BS_CAPACITY)
        small_reward_total += small_reward_calculation(
            actions,
            bs.serv_user_list,
            SMALL_FREQUENCY,
            SMALL_TX_POWER,
            bs_object=bs,
            edge_weights=EDGE_REWARD_WEIGHTS,
            use_required_rate=True,
            use_sc_budget=True,
            use_total_power=False,
        )
        spec = {
            "bs": bs,
            "frequency": SMALL_FREQUENCY,
            "tx_power": SMALL_TX_POWER,
            "total_sc": SMALL_NUM_OF_SC,
            "total_bw": SMALL_TOTAL_BW_HZ,
            "req": EDGE_THR_REQ_SMALL_Mbps,
        }
        for idx, user in enumerate(bs.serv_user_list):
            alloc_sc, power_split = actions[idx]
            rate = rate_for_user(user, spec, alloc_sc, power_split)
            slot_total_throughput += rate
            slot_rates[user.global_index] = rate
            ratio = min(rate / spec["req"], 1.0)
            slot_satisfaction[user.global_index] = ratio
            if rate >= spec["req"]:
                satisfied += 1

    uav_reward_total = 0.0
    for bs in uav_bs_list:
        active = len(bs.serv_user_list)
        if active <= 0:
            continue
        uav_reward_total += uav_reward_calculation(
            bs.serv_user_list,
            bs,
            UAV_FREQUENCY,
            UAV_TX_POWER,
            edge_weights=EDGE_REWARD_WEIGHTS,
            use_required_rate=True,
            use_sc_budget=True,
            use_total_power=False,
        )
        actions = equal_action_list(active, UAV_NUM_OF_SC, UAV_BS_CAPACITY, False)
        spec = {
            "bs": bs,
            "frequency": UAV_FREQUENCY,
            "tx_power": UAV_TX_POWER,
            "total_sc": UAV_NUM_OF_SC,
            "total_bw": UAV_TOTAL_BW_HZ,
            "req": EDGE_THR_REQ_UAV_Mbps,
        }
        for idx, user in enumerate(bs.serv_user_list):
            alloc_sc, power_split = actions[idx]
            rate = rate_for_user(user, spec, alloc_sc, power_split)
            slot_total_throughput += rate
            slot_rates[user.global_index] = rate
            ratio = min(rate / spec["req"], 1.0)
            slot_satisfaction[user.global_index] = ratio
            if rate >= spec["req"]:
                satisfied += 1

    for bs in sat_bs_list:
        active = len(bs.serv_user_list)
        if active <= 0:
            continue
        actions = equal_action_list(active, SAT_NUM_OF_SC, SAT_BS_CAPACITY, False)
        spec = {
            "bs": bs,
            "frequency": SAT_FREQUENCY,
            "tx_power": SAT_TX_POWER,
            "total_sc": SAT_NUM_OF_SC,
            "total_bw": SAT_TOTAL_BW_HZ,
            "req": SAT_THR_REQ_Mbps,
        }
        for idx, user in enumerate(bs.serv_user_list):
            alloc_sc, power_split = actions[idx]
            rate = rate_for_user(user, spec, alloc_sc, power_split)
            slot_total_throughput += rate
            slot_rates[user.global_index] = rate
            ratio = min(rate / spec["req"], 1.0)
            slot_satisfaction[user.global_index] = ratio
            if rate >= spec["req"]:
                satisfied += 1

    qos = satisfied / slot_connected if slot_connected > 0 else 0.0
    jain = calculate_jain_fairness(slot_satisfaction)
    edge_rewards = {
        "macro": macro_reward_total / max(len(macro_bs_list), 1),
        "small": small_reward_total / max(len(small_bs_list), 1),
        "uav": uav_reward_total / max(len(uav_bs_list), 1),
    }
    return slot_total_throughput, qos, jain, slot_rates, slot_satisfaction, edge_rewards


def run_greedy(seed, max_epochs, cloud_association_period, lambda_load):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()
    historical_actions = {user.global_index: 0 for user in user_list}
    metrics = {key: [] for key in METRIC_HEADER}

    for epoch in range(max_epochs):
        specs = node_specs(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list)
        epoch_throughput = 0.0
        epoch_qos = 0.0
        epoch_jain = 0.0
        epoch_macro_reward = 0.0
        epoch_small_reward = 0.0
        epoch_uav_reward = 0.0
        epoch_switch_count = 0
        cloud_reward_avg = 0.0

        for slot in range(cloud_association_period):
            for user in user_list:
                user.calculate_next_step()
            if slot == 0:
                historical_actions, epoch_switch_count, cloud_reward_avg = (
                    choose_greedy_association(user_list, specs, lambda_load)
                )
            else:
                apply_historical_association(user_list, historical_actions)

            throughput, qos, jain, _, _, edge_rewards = evaluate_greedy_slot(
                user_list, sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list
            )
            epoch_throughput += throughput
            epoch_qos += qos
            epoch_jain += jain
            epoch_macro_reward += edge_rewards["macro"]
            epoch_small_reward += edge_rewards["small"]
            epoch_uav_reward += edge_rewards["uav"]

            for user in user_list:
                user.move()

        denom = max(cloud_association_period, 1)
        macro_reward_avg = epoch_macro_reward / denom
        small_reward_avg = epoch_small_reward / denom
        uav_reward_avg = epoch_uav_reward / denom
        metrics["reward_sum"].append(
            cloud_reward_avg + macro_reward_avg + small_reward_avg + uav_reward_avg
        )
        metrics["throughput_mbps"].append(epoch_throughput / denom)
        metrics["qos_satisfaction"].append(epoch_qos / denom)
        metrics["jain_fairness"].append(epoch_jain / denom)
        metrics["switch_ratio"].append(epoch_switch_count / max(len(user_list), 1))
        metrics["cloud_loss"].append(None)
        metrics["macro_loss"].append(None)
        metrics["small_loss"].append(None)
        metrics["uav_loss"].append(None)

    return metrics


def run_rl_metrics(seed, max_epochs, device, config):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()
    (
        losses,
        _,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        _,
        _,
        _,
        switch_count_history,
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
    user_count = max(len(user_list), 1)
    metrics = {
        "reward_sum": reward_sum_history,
        "throughput_mbps": throughput_history,
        "qos_satisfaction": qos_history,
        "jain_fairness": jain_history,
        "switch_ratio": [v / user_count for v in switch_count_history],
        "cloud_loss": losses["cloud"],
        "macro_loss": losses["macro"],
        "small_loss": losses["small"],
        "uav_loss": losses["uav"],
    }
    return metrics


def run_single_timescale(seed, max_epochs, device):
    config = {
        "max_epochs": max_epochs,
        "cloud_association_period": 1,
        "epsilon_schedule": "exp2",
        "convergence_check": True,
        "convergence_window": 1000,
        "convergence_std_ratio": 0.05,
        "min_convergence_epochs": 2000,
        "switch_penalty": 0.0,
        "stickiness_bonus": 0.0,
    }
    return run_rl_metrics(seed, max_epochs, device, config)


def run_hmadqn_k3(seed, max_epochs, device):
    config = {
        "max_epochs": max_epochs,
        "cloud_association_period": 3,
        "epsilon_schedule": "exp2",
        "convergence_check": False,
        "switch_penalty": CLOUD_SWITCH_PENALTY,
        "stickiness_bonus": CLOUD_STICKINESS_BONUS,
    }
    return run_rl_metrics(seed, max_epochs, device, config)


def read_hmadqn_original():
    kpi_path = REPO_DIR / "logs" / "fig9_kpi_distribution.csv"
    switch_path = REPO_DIR / "logs" / "fig7_stability_fairness_tradeoff.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"missing original KPI log: {kpi_path}")
    if not switch_path.exists():
        raise FileNotFoundError(f"missing original switch log: {switch_path}")

    switch_by_epoch = {}
    with switch_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_id") != "baseline":
                continue
            epoch = row.get("epoch")
            switch_ratio = row.get("switch_ratio")
            if epoch == "" or switch_ratio == "":
                continue
            switch_by_epoch[int(epoch)] = float(switch_ratio)

    metrics = {key: [] for key in METRIC_HEADER}
    with kpi_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_id") != "baseline":
                continue
            epoch = int(row["epoch"])
            metrics["reward_sum"].append(float(row["reward_sum"]))
            metrics["throughput_mbps"].append(float(row["throughput"]))
            metrics["qos_satisfaction"].append(float(row["qos"]))
            metrics["jain_fairness"].append(float(row["jain"]))
            metrics["switch_ratio"].append(float(switch_by_epoch.get(epoch, 0.0)))
            metrics["cloud_loss"].append(None)
            metrics["macro_loss"].append(None)
            metrics["small_loss"].append(None)
            metrics["uav_loss"].append(None)
    if not metrics["reward_sum"]:
        raise ValueError("no baseline rows found in original KPI log")
    return metrics


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_epoch_metrics(output_root, run_id, metrics):
    run_path = output_root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "epoch_metrics.csv"
    row_count = len(metrics["reward_sum"])
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + METRIC_HEADER)
        for epoch in range(row_count):
            row = [epoch]
            for key in METRIC_HEADER:
                value = metrics[key][epoch] if epoch < len(metrics[key]) else None
                row.append("" if value is None else value)
            writer.writerow(row)
    return metrics_path


def load_epoch_metrics(metrics_path):
    metrics = {key: [] for key in METRIC_HEADER}
    with Path(metrics_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in METRIC_HEADER:
                value = row.get(key, "")
                metrics[key].append(None if value == "" else float(value))
    return metrics


def maybe_reuse_existing_run(
    output_root,
    run_id,
    method,
    seed,
    source,
    config_path,
    manifest_rows,
    metric_records,
    min_epochs,
):
    metrics_path = output_root / "runs" / run_id / "epoch_metrics.csv"
    if not metrics_path.exists():
        return False
    metrics = load_epoch_metrics(metrics_path)
    if len(metrics["reward_sum"]) < min_epochs:
        return False
    metric_records.append(
        {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "metrics": metrics,
        }
    )
    add_manifest_row(
        manifest_rows,
        run_id,
        method,
        seed,
        source,
        config_path,
        metrics_path,
        "success",
        "reused existing revision output",
    )
    return True


def tail_slice(values, tail_window):
    if not values:
        return []
    if tail_window == "last_10pct":
        count = max(int(len(values) * 0.1), 1)
    elif tail_window == "last_1000":
        count = min(1000, len(values))
    else:
        raise ValueError(f"unknown tail window: {tail_window}")
    return values[-count:]


def summarize_seed(method, seed, metrics, tail_window):
    row = {
        "method": method,
        "seed": seed,
        "tail_window": tail_window,
    }
    for key in REQUIRED_KPIS:
        values = [float(v) for v in tail_slice(metrics[key], tail_window)]
        row[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return row


def write_csv_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(output_root, rows):
    header = [
        "run_id",
        "method",
        "seed",
        "source",
        "config_path",
        "metrics_path",
        "status",
        "notes",
    ]
    write_csv_rows(output_root / "run_manifest.csv", header, rows)


def write_summaries(output_root, metric_records):
    seed_rows = []
    for record in metric_records:
        for tail_window in ["last_10pct", "last_1000"]:
            seed_rows.append(
                summarize_seed(
                    record["method"], record["seed"], record["metrics"], tail_window
                )
            )

    by_seed_header = [
        "method",
        "seed",
        "tail_window",
        "reward_sum_mean",
        "throughput_mbps_mean",
        "qos_satisfaction_mean",
        "jain_fairness_mean",
        "switch_ratio_mean",
    ]
    write_csv_rows(output_root / "summary_by_seed.csv", by_seed_header, seed_rows)

    mean_std_rows = []
    for method in sorted({row["method"] for row in seed_rows}):
        for tail_window in ["last_10pct", "last_1000"]:
            group = [
                row
                for row in seed_rows
                if row["method"] == method and row["tail_window"] == tail_window
            ]
            if not group:
                continue
            out = {
                "method": method,
                "n_seed": len(group),
                "tail_window": tail_window,
            }
            for key in REQUIRED_KPIS:
                values = [row[f"{key}_mean"] for row in group]
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            mean_std_rows.append(out)

    mean_std_header = [
        "method",
        "n_seed",
        "tail_window",
        "reward_sum_mean",
        "reward_sum_std",
        "throughput_mbps_mean",
        "throughput_mbps_std",
        "qos_satisfaction_mean",
        "qos_satisfaction_std",
        "jain_fairness_mean",
        "jain_fairness_std",
        "switch_ratio_mean",
        "switch_ratio_std",
    ]
    write_csv_rows(output_root / "summary_mean_std.csv", mean_std_header, mean_std_rows)

    fig_rows = []
    notes_by_method = {
        "hmadqn_k3_original": "H-MADQN K=3: seed 42 original log plus reproduced seeds 43/44",
        "greedy_rsrp_capacity": "3-seed deterministic heuristic with lambda_load=0.2",
        "single_timescale_madqn_k1": "3-seed K=1 MADQN-style run with stability terms disabled",
    }
    for row in mean_std_rows:
        for key in REQUIRED_KPIS:
            fig_rows.append(
                {
                    "method": row["method"],
                    "kpi": key,
                    "mean": row[f"{key}_mean"],
                    "std": row[f"{key}_std"],
                    "n_seed": row["n_seed"],
                    "tail_window": row["tail_window"],
                    "plot_group": "baseline_comparison",
                    "notes": notes_by_method.get(row["method"], ""),
                }
            )
    fig_header = [
        "method",
        "kpi",
        "mean",
        "std",
        "n_seed",
        "tail_window",
        "plot_group",
        "notes",
    ]
    write_csv_rows(output_root / "fig10_baseline_comparison.csv", fig_header, fig_rows)
    return seed_rows, mean_std_rows, fig_rows


def plot_fig10_preview(plot_root, fig_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in fig_rows if row["tail_window"] == "last_10pct"]
    if not rows:
        return
    methods = ["hmadqn_k3_original", "greedy_rsrp_capacity", "single_timescale_madqn_k1"]
    fig, axes = plt.subplots(1, len(REQUIRED_KPIS), figsize=(18, 4))
    for idx, kpi in enumerate(REQUIRED_KPIS):
        ax = axes[idx]
        values = []
        errors = []
        for method in methods:
            row = next((r for r in rows if r["method"] == method and r["kpi"] == kpi), None)
            values.append(float(row["mean"]) if row else 0.0)
            errors.append(float(row["std"]) if row else 0.0)
        ax.bar(methods, values, yerr=errors, capsize=3)
        ax.set_title(kpi)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "fig10_preview_baseline_comparison.png", dpi=300)
    plt.close(fig)


def plot_training_curves(plot_root, metric_records):
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(REQUIRED_KPIS), 1, figsize=(9, 12), sharex=False)
    for idx, kpi in enumerate(REQUIRED_KPIS):
        ax = axes[idx]
        for record in metric_records:
            y = record["metrics"][kpi]
            label = f"{record['method']}:{record['seed']}"
            ax.plot(range(len(y)), y, linewidth=0.8, alpha=0.7, label=label)
        ax.set_title(kpi)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(plot_root / "training_curves_by_method.png", dpi=300)
    plt.close(fig)


def validate_fig10(output_root):
    path = output_root / "fig10_baseline_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    present = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            present.add(row["kpi"])
    missing = set(REQUIRED_KPIS) - present
    if missing:
        raise ValueError(f"fig10 missing KPI rows: {sorted(missing)}")


def write_method_configs(output_root, max_epochs, smoke_mode, device_name):
    config_dir = output_root / "method_configs"
    hmadqn_config = {
        "method": "hmadqn_k3_original",
        "sources": {
            "seed42": "logs/fig9_kpi_distribution.csv + logs/fig7_stability_fairness_tradeoff.csv",
            "seeds43_44": "reproduced_hmadqn runs under logs/revision_round1/group1_baselines/runs",
        },
        "seeds": [42, 43, 44],
        "cloud_association_period": 3,
        "epsilon_schedule": "exp2",
        "switch_penalty": CLOUD_SWITCH_PENALTY,
        "stickiness_bonus": CLOUD_STICKINESS_BONUS,
        "convergence_check": False,
        "execution_device": device_name,
        "reused_original_seed42_logs": True,
        "smoke_mode": smoke_mode,
    }
    greedy_config = {
        "method": "greedy_rsrp_capacity",
        "seeds": [42, 43, 44],
        "max_epochs": max_epochs,
        "cloud_association_period": 3,
        "association_score": "normalize_rsrp(rsrp_dbm) - lambda_load * normalized_load",
        "lambda_load": 0.2,
        "capacity_constraint": "strict; full nodes are excluded",
        "resource_allocation": "equal subcarrier split among active users",
        "power_allocation": "equal terrestrial power split; UAV and satellite use existing power convention",
        "uav_policy": "frozen",
        "smoke_mode": smoke_mode,
    }
    single_config = {
        "method": "single_timescale_madqn_k1",
        "seeds": [42, 43, 44],
        "max_epochs": max_epochs,
        "cloud_association_period": 1,
        "epsilon_schedule": "exp2",
        "switch_penalty": 0.0,
        "stickiness_bonus": 0.0,
        "network_structure": "existing CloudAgent, BsAgent, and UavAgent classes",
        "execution_device": device_name,
        "smoke_mode": smoke_mode,
    }
    paths = {
        "hmadqn_k3_original": config_dir / "hmadqn_k3_original.json",
        "greedy_rsrp_capacity": config_dir / "greedy_rsrp_capacity.json",
        "single_timescale_madqn_k1": config_dir / "single_timescale_madqn_k1.json",
    }
    write_json(paths["hmadqn_k3_original"], hmadqn_config)
    write_json(paths["greedy_rsrp_capacity"], greedy_config)
    write_json(paths["single_timescale_madqn_k1"], single_config)
    return paths


def add_manifest_row(rows, run_id, method, seed, source, config_path, metrics_path, status, notes):
    rows.append(
        {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "source": source,
            "config_path": rel(config_path) if config_path else "",
            "metrics_path": rel(metrics_path) if metrics_path else "",
            "status": status,
            "notes": notes,
        }
    )


def run_all(
    output_root,
    plot_root,
    max_epochs,
    smoke_mode,
    skip_plots,
    resume,
    device_name,
    only_hmadqn_reproduce,
):
    ensure_dirs(output_root)
    plot_root.mkdir(parents=True, exist_ok=True)
    config_paths = write_method_configs(output_root, max_epochs, smoke_mode, device_name)
    manifest_rows = []
    metric_records = []

    try:
        run_id = "hmadqn_k3_original_seed42"
        if not (
            resume
            and maybe_reuse_existing_run(
                output_root,
                run_id,
                "hmadqn_k3_original",
                42,
                "original_logs",
                config_paths["hmadqn_k3_original"],
                manifest_rows,
                metric_records,
                min_epochs=1,
            )
        ):
            metrics = read_hmadqn_original()
            metrics_path = write_epoch_metrics(output_root, run_id, metrics)
            metric_records.append(
                {
                    "run_id": run_id,
                    "method": "hmadqn_k3_original",
                    "seed": 42,
                    "metrics": metrics,
                }
            )
            add_manifest_row(
                manifest_rows,
                run_id,
                "hmadqn_k3_original",
                42,
                "original_logs",
                config_paths["hmadqn_k3_original"],
                metrics_path,
                "success",
                "reused original H-MADQN K=3 baseline rows",
            )
        write_manifest(output_root, manifest_rows)
    except Exception as exc:
        add_manifest_row(
            manifest_rows,
            "hmadqn_k3_original_seed42",
            "hmadqn_k3_original",
            42,
            "original_logs",
            config_paths.get("hmadqn_k3_original"),
            "",
            "failed",
            str(exc),
        )
        write_manifest(output_root, manifest_rows)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    for seed in [43, 44]:
        run_id = f"hmadqn_k3_reproduced_seed{seed}"
        try:
            if not (
                resume
                and maybe_reuse_existing_run(
                    output_root,
                    run_id,
                    "hmadqn_k3_original",
                    seed,
                    "reproduced_hmadqn",
                    config_paths["hmadqn_k3_original"],
                    manifest_rows,
                    metric_records,
                    min_epochs=max_epochs,
                )
            ):
                metrics = run_hmadqn_k3(seed, max_epochs, device)
                metrics_path = write_epoch_metrics(output_root, run_id, metrics)
                metric_records.append(
                    {
                        "run_id": run_id,
                        "method": "hmadqn_k3_original",
                        "seed": seed,
                        "metrics": metrics,
                    }
                )
                add_manifest_row(
                    manifest_rows,
                    run_id,
                    "hmadqn_k3_original",
                    seed,
                    "reproduced_hmadqn",
                    config_paths["hmadqn_k3_original"],
                    metrics_path,
                    "success",
                    f"completed {len(metrics['reward_sum'])} epochs",
                )
            write_manifest(output_root, manifest_rows)
        except Exception as exc:
            add_manifest_row(
                manifest_rows,
                run_id,
                "hmadqn_k3_original",
                seed,
                "reproduced_hmadqn",
                config_paths.get("hmadqn_k3_original"),
                "",
                "failed",
                str(exc),
            )
            write_manifest(output_root, manifest_rows)

    for seed in [42, 43, 44]:
        run_id = f"greedy_rsrp_capacity_seed{seed}"
        try:
            if not (
                resume
                and maybe_reuse_existing_run(
                    output_root,
                    run_id,
                    "greedy_rsrp_capacity",
                    seed,
                    "simulated_greedy",
                    config_paths["greedy_rsrp_capacity"],
                    manifest_rows,
                    metric_records,
                    min_epochs=max_epochs,
                )
            ):
                if only_hmadqn_reproduce:
                    raise FileNotFoundError(
                        f"missing existing {run_id}; --only-hmadqn-reproduce does not rerun Greedy"
                    )
                metrics = run_greedy(seed, max_epochs, 3, lambda_load=0.2)
                metrics_path = write_epoch_metrics(output_root, run_id, metrics)
                metric_records.append(
                    {
                        "run_id": run_id,
                        "method": "greedy_rsrp_capacity",
                        "seed": seed,
                        "metrics": metrics,
                    }
                )
                add_manifest_row(
                    manifest_rows,
                    run_id,
                    "greedy_rsrp_capacity",
                    seed,
                    "simulated_greedy",
                    config_paths["greedy_rsrp_capacity"],
                    metrics_path,
                    "success",
                    "completed",
                )
            write_manifest(output_root, manifest_rows)
        except Exception as exc:
            add_manifest_row(
                manifest_rows,
                run_id,
                "greedy_rsrp_capacity",
                seed,
                "simulated_greedy",
                config_paths.get("greedy_rsrp_capacity"),
                "",
                "failed",
                str(exc),
            )
            write_manifest(output_root, manifest_rows)

    for seed in [42, 43, 44]:
        run_id = f"single_timescale_madqn_k1_seed{seed}"
        try:
            if not (
                resume
                and maybe_reuse_existing_run(
                    output_root,
                    run_id,
                    "single_timescale_madqn_k1",
                    seed,
                    "trained_rl",
                    config_paths["single_timescale_madqn_k1"],
                    manifest_rows,
                    metric_records,
                    min_epochs=max_epochs,
                )
            ):
                if only_hmadqn_reproduce:
                    raise FileNotFoundError(
                        f"missing existing {run_id}; --only-hmadqn-reproduce does not rerun single-timescale"
                    )
                metrics = run_single_timescale(seed, max_epochs, device)
                metrics_path = write_epoch_metrics(output_root, run_id, metrics)
                metric_records.append(
                    {
                        "run_id": run_id,
                        "method": "single_timescale_madqn_k1",
                        "seed": seed,
                        "metrics": metrics,
                    }
                )
                add_manifest_row(
                    manifest_rows,
                    run_id,
                    "single_timescale_madqn_k1",
                    seed,
                    "trained_rl",
                    config_paths["single_timescale_madqn_k1"],
                    metrics_path,
                    "success",
                    f"completed {len(metrics['reward_sum'])} epochs",
                )
            write_manifest(output_root, manifest_rows)
        except Exception as exc:
            add_manifest_row(
                manifest_rows,
                run_id,
                "single_timescale_madqn_k1",
                seed,
                "trained_rl",
                config_paths.get("single_timescale_madqn_k1"),
                "",
                "failed",
                str(exc),
            )
            write_manifest(output_root, manifest_rows)

    write_manifest(output_root, manifest_rows)
    _, _, fig_rows = write_summaries(output_root, metric_records)
    validate_fig10(output_root)
    if not skip_plots:
        plot_fig10_preview(plot_root, fig_rows)
        plot_training_curves(plot_root, metric_records)
    return manifest_rows, metric_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run revision round 1 Group 1 baseline experiments."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device for the single-timescale neural baseline. CPU is faster for this small-tensor loop.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed revision run CSVs instead of recomputing them.",
    )
    parser.add_argument(
        "--only-hmadqn-reproduce",
        action="store_true",
        help="Only run missing H-MADQN K=3 reproduced seeds; require existing Greedy and single-timescale run CSVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    smoke_mode = args.mode == "smoke"
    if smoke_mode:
        output_root = LOG_ROOT / "smoke"
        plot_root = PLOT_ROOT / "smoke"
        max_epochs = args.smoke_epochs
    else:
        output_root = LOG_ROOT
        plot_root = PLOT_ROOT
        max_epochs = args.max_epochs
    manifest_rows, metric_records = run_all(
        output_root,
        plot_root,
        max_epochs,
        smoke_mode,
        args.skip_plots,
        args.resume,
        args.device,
        args.only_hmadqn_reproduce,
    )
    status_counts = {}
    for row in manifest_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_root": rel(output_root),
                "runs": len(manifest_rows),
                "metric_records": len(metric_records),
                "status_counts": status_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
