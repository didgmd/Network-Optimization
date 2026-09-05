# -*- coding: utf-8 -*-
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Classes import User
from Parameters import (
    AREA_SIZE_X,
    AREA_SIZE_Y,
    CLOUD_ASSOCIATION_PERIOD,
    CLOUD_REWARD_WEIGHTS,
    CLOUD_STICKINESS_BONUS,
    CLOUD_SWITCH_PENALTY,
    EDGE_REWARD_WEIGHTS,
    MAX_EPOCHS,
)
from SC_RL_main import rl
from Topology import define_topology
from baselines import (
    apply_historical_association,
    choose_greedy_association,
    evaluate_greedy_slot,
    node_specs,
)


REPO_DIR = Path(__file__).resolve().parent
LOG_ROOT = REPO_DIR / "logs" / "revision_round1" / "group4_scalability_density"
PLOT_ROOT = REPO_DIR / "plots" / "revision_round1" / "group4_scalability_density"
SEED = 42
K_VALUE = 3
LAMBDA_LOAD = 0.2
METHODS = ["hmadqn_k3", "greedy_rsrp_capacity"]
REQUIRED_KPIS = [
    "reward_sum",
    "throughput_mbps",
    "qos_satisfaction",
    "jain_fairness",
    "switch_ratio",
]
LOSS_KEYS = ["cloud_loss", "macro_loss", "small_loss", "uav_loss"]
METRIC_HEADER = REQUIRED_KPIS + LOSS_KEYS
TAIL_WINDOWS = ["last_10pct", "last_1000"]
HOTSPOT_CENTERS = [(2500.0, 2500.0), (7500.0, 3500.0), (5000.0, 7500.0)]
HOTSPOT_FRACTION = 0.70
HOTSPOT_SIGMA_M = 600.0


SCENARIOS = [
    {
        "scenario_id": "scale_uniform_N50",
        "scenario_type": "scale_sweep",
        "N_UE": 50,
        "distribution": "uniform",
    },
    {
        "scenario_id": "scale_uniform_N100",
        "scenario_type": "scale_sweep",
        "N_UE": 100,
        "distribution": "uniform",
    },
    {
        "scenario_id": "scale_uniform_N150",
        "scenario_type": "scale_sweep",
        "N_UE": 150,
        "distribution": "uniform",
    },
    {
        "scenario_id": "density_hotspot_N100",
        "scenario_type": "density_stress",
        "N_UE": 100,
        "distribution": "hotspot",
    },
]

SMOKE_RECORDS = [
    ("hmadqn_k3", "scale_uniform_N50"),
    ("hmadqn_k3", "scale_uniform_N150"),
    ("hmadqn_k3", "density_hotspot_N100"),
    ("greedy_rsrp_capacity", "density_hotspot_N100"),
]


def rel(path):
    return str(Path(path).resolve().relative_to(REPO_DIR)) if path else ""


def scenario_by_id(scenario_id):
    for scenario in SCENARIOS:
        if scenario["scenario_id"] == scenario_id:
            return scenario
    raise KeyError(scenario_id)


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


def device_from_name(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_user_types(n_ue):
    num_embb = n_ue // 3
    num_mmtc = n_ue // 3
    num_urllc = n_ue - num_embb - num_mmtc
    user_types = ["eMBB"] * num_embb + ["mMTC"] * num_mmtc + ["uRLLC"] * num_urllc
    np.random.shuffle(user_types)
    return user_types


def unique_uniform_positions(n_ue):
    positions = []
    occupied = set()
    for _ in range(n_ue):
        for _ in range(1000):
            x = round(np.random.uniform(0, AREA_SIZE_X), 2)
            y = round(np.random.uniform(0, AREA_SIZE_Y), 2)
            if (x, y) not in occupied:
                occupied.add((x, y))
                positions.append((x, y))
                break
        else:
            grid_size = int(np.ceil(np.sqrt(n_ue)))
            grid_x = (len(occupied) % grid_size) * (AREA_SIZE_X / grid_size)
            grid_y = (len(occupied) // grid_size) * (AREA_SIZE_Y / grid_size)
            occupied.add((grid_x, grid_y))
            positions.append((grid_x, grid_y))
    return positions


def hotspot_positions(n_ue):
    hotspot_count = int(round(n_ue * HOTSPOT_FRACTION))
    background_count = n_ue - hotspot_count
    base = hotspot_count // len(HOTSPOT_CENTERS)
    remainder = hotspot_count - base * len(HOTSPOT_CENTERS)
    center_counts = [
        base + (1 if idx < remainder else 0)
        for idx in range(len(HOTSPOT_CENTERS))
    ]
    positions = []
    occupied = set()

    def add_unique(x, y):
        x = round(float(np.clip(x, 0, AREA_SIZE_X)), 2)
        y = round(float(np.clip(y, 0, AREA_SIZE_Y)), 2)
        if (x, y) not in occupied:
            occupied.add((x, y))
            positions.append((x, y))
            return
        for _ in range(1000):
            jitter_x = round(float(np.clip(x + np.random.normal(0, 1), 0, AREA_SIZE_X)), 2)
            jitter_y = round(float(np.clip(y + np.random.normal(0, 1), 0, AREA_SIZE_Y)), 2)
            if (jitter_x, jitter_y) not in occupied:
                occupied.add((jitter_x, jitter_y))
                positions.append((jitter_x, jitter_y))
                return
        positions.append((x, y))

    for (center_x, center_y), count in zip(HOTSPOT_CENTERS, center_counts):
        for _ in range(count):
            add_unique(
                center_x + np.random.normal(0, HOTSPOT_SIGMA_M),
                center_y + np.random.normal(0, HOTSPOT_SIGMA_M),
            )

    for _ in range(background_count):
        for _ in range(1000):
            x = round(np.random.uniform(0, AREA_SIZE_X), 2)
            y = round(np.random.uniform(0, AREA_SIZE_Y), 2)
            if (x, y) not in occupied:
                occupied.add((x, y))
                positions.append((x, y))
                break
        else:
            add_unique(np.random.uniform(0, AREA_SIZE_X), np.random.uniform(0, AREA_SIZE_Y))

    return positions


def make_users(n_ue, distribution):
    user_types = make_user_types(n_ue)
    if distribution == "uniform":
        positions = unique_uniform_positions(n_ue)
    elif distribution == "hotspot":
        positions = hotspot_positions(n_ue)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return [
        User(global_index=i, x=positions[i][0], y=positions[i][1], user_type=user_types[i])
        for i in range(n_ue)
    ]


def users_for_scenario(seed, scenario):
    setup_seed(seed)
    return make_users(scenario["N_UE"], scenario["distribution"])


def user_position_rows(seed):
    rows = []
    for scenario_id in ["scale_uniform_N100", "density_hotspot_N100"]:
        scenario = scenario_by_id(scenario_id)
        users = users_for_scenario(seed, scenario)
        for user in users:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "distribution": scenario["distribution"],
                    "seed": seed,
                    "global_index": user.global_index,
                    "x": user.x,
                    "y": user.y,
                    "user_type": user.user_type,
                }
            )
    return rows


def nearest_hotspot_fraction(users, radius_m=1200.0):
    count = 0
    for user in users:
        dist = min(
            ((user.x - cx) ** 2 + (user.y - cy) ** 2) ** 0.5
            for cx, cy in HOTSPOT_CENTERS
        )
        if dist <= radius_m:
            count += 1
    return count / max(len(users), 1)


def validate_user_generation(seed):
    hotspot_a = users_for_scenario(seed, scenario_by_id("density_hotspot_N100"))
    hotspot_b = users_for_scenario(seed, scenario_by_id("density_hotspot_N100"))
    coords_a = [(u.x, u.y, u.user_type) for u in hotspot_a]
    coords_b = [(u.x, u.y, u.user_type) for u in hotspot_b]
    if coords_a != coords_b:
        raise ValueError("hotspot user generation is not reproducible")

    uniform = users_for_scenario(seed, scenario_by_id("scale_uniform_N100"))
    hotspot_fraction = nearest_hotspot_fraction(hotspot_a)
    uniform_fraction = nearest_hotspot_fraction(uniform)
    if hotspot_fraction <= uniform_fraction:
        raise ValueError(
            f"hotspot density check failed: hotspot={hotspot_fraction}, uniform={uniform_fraction}"
        )
    return {
        "hotspot_fraction_within_1200m": hotspot_fraction,
        "uniform_fraction_within_1200m": uniform_fraction,
    }


def run_hmadqn(seed, scenario, max_epochs, device):
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = users_for_scenario(seed, scenario)
    config = {
        "max_epochs": max_epochs,
        "cloud_association_period": K_VALUE,
        "epsilon_schedule": "exp2",
        "convergence_check": False,
    }
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
    return {
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


def run_greedy(seed, scenario, max_epochs):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = make_users(scenario["N_UE"], scenario["distribution"])
    historical_actions = {user.global_index: 0 for user in user_list}
    metrics = {key: [] for key in METRIC_HEADER}

    for _ in range(max_epochs):
        specs = node_specs(sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list)
        epoch_throughput = 0.0
        epoch_qos = 0.0
        epoch_jain = 0.0
        epoch_macro_reward = 0.0
        epoch_small_reward = 0.0
        epoch_uav_reward = 0.0
        epoch_switch_count = 0
        cloud_reward_avg = 0.0

        for slot in range(K_VALUE):
            for user in user_list:
                user.calculate_next_step()
            if slot == 0:
                historical_actions, epoch_switch_count, cloud_reward_avg = (
                    choose_greedy_association(user_list, specs, LAMBDA_LOAD)
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

        macro_reward_avg = epoch_macro_reward / K_VALUE
        small_reward_avg = epoch_small_reward / K_VALUE
        uav_reward_avg = epoch_uav_reward / K_VALUE
        metrics["reward_sum"].append(
            cloud_reward_avg + macro_reward_avg + small_reward_avg + uav_reward_avg
        )
        metrics["throughput_mbps"].append(epoch_throughput / K_VALUE)
        metrics["qos_satisfaction"].append(epoch_qos / K_VALUE)
        metrics["jain_fairness"].append(epoch_jain / K_VALUE)
        metrics["switch_ratio"].append(epoch_switch_count / max(len(user_list), 1))
        for key in LOSS_KEYS:
            metrics[key].append(None)
    return metrics


def read_original_hmadqn_metrics():
    kpi_path = REPO_DIR / "logs" / "fig9_kpi_distribution.csv"
    switch_path = REPO_DIR / "logs" / "fig7_stability_fairness_tradeoff.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"missing original KPI log: {kpi_path}")
    if not switch_path.exists():
        raise FileNotFoundError(f"missing original switch log: {switch_path}")

    switch_by_epoch = {}
    with switch_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id") != "baseline":
                continue
            if row.get("epoch") == "" or row.get("switch_ratio") == "":
                continue
            switch_by_epoch[int(row["epoch"])] = float(row["switch_ratio"])

    metrics = {key: [] for key in METRIC_HEADER}
    with kpi_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id") != "baseline":
                continue
            epoch = int(row["epoch"])
            metrics["reward_sum"].append(float(row["reward_sum"]))
            metrics["throughput_mbps"].append(float(row["throughput"]))
            metrics["qos_satisfaction"].append(float(row["qos"]))
            metrics["jain_fairness"].append(float(row["jain"]))
            metrics["switch_ratio"].append(float(switch_by_epoch.get(epoch, 0.0)))
            for key in LOSS_KEYS:
                metrics[key].append(None)
    if len(metrics["reward_sum"]) != 10000:
        raise ValueError(f"original H-MADQN rows={len(metrics['reward_sum'])}, expected 10000")
    return metrics


def read_group1_greedy_metrics():
    manifest_path = REPO_DIR / "logs" / "revision_round1" / "group1_baselines" / "run_manifest.csv"
    config_path = REPO_DIR / "logs" / "revision_round1" / "group1_baselines" / "method_configs" / "greedy_rsrp_capacity.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "cloud_association_period": K_VALUE,
        "lambda_load": LAMBDA_LOAD,
        "max_epochs": MAX_EPOCHS,
        "uav_policy": "frozen",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Group1 Greedy config mismatch for {key}: {config.get(key)}")

    metrics_path = None
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("run_id") == "greedy_rsrp_capacity_seed42"
                and row.get("status") == "success"
            ):
                metrics_path = REPO_DIR / row["metrics_path"]
                break
    if metrics_path is None or not metrics_path.exists():
        raise FileNotFoundError("missing Group1 Greedy seed42 metrics")
    metrics = load_epoch_metrics(metrics_path)
    if len(metrics["reward_sum"]) != 10000:
        raise ValueError(f"Group1 Greedy rows={len(metrics['reward_sum'])}, expected 10000")
    return metrics


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_method_config(output_root, max_epochs, device_name, density_check):
    config_path = output_root / "method_configs" / "scalability_density_config.json"
    write_json(
        config_path,
        {
            "methods": METHODS,
            "scenarios": SCENARIOS,
            "seed": SEED,
            "cloud_association_period": K_VALUE,
            "max_epochs": max_epochs,
            "epsilon_schedule": "exp2",
            "convergence_check": False,
            "greedy_lambda_load": LAMBDA_LOAD,
            "greedy_uav_policy": "frozen",
            "greedy_resource_allocation": "equal subcarrier split among active users",
            "hotspot_centers_m": HOTSPOT_CENTERS,
            "hotspot_fraction": HOTSPOT_FRACTION,
            "hotspot_sigma_m": HOTSPOT_SIGMA_M,
            "density_check": density_check,
            "reuse_policy": {
                "hmadqn_k3_scale_uniform_N100": "original_logs if available",
                "greedy_scale_uniform_N100": "group1_logs if compatible",
            },
            "execution_device": device_name,
            "claim_boundary": "evaluated UE-scale and density-stress settings only; not a universal large-scale SAGIN scalability proof",
            "default_reward_weights": {
                "cloud": CLOUD_REWARD_WEIGHTS,
                "edge": EDGE_REWARD_WEIGHTS,
            },
            "default_stability_terms": {
                "switch_penalty": CLOUD_SWITCH_PENALTY,
                "stickiness_bonus": CLOUD_STICKINESS_BONUS,
            },
        },
    )
    return config_path


def run_id_for(method, scenario_id, seed):
    return f"{method}_{scenario_id}_seed{seed}"


def write_epoch_metrics(output_root, run_id, method, scenario, seed, metrics):
    run_path = output_root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "epoch_metrics.csv"
    header = [
        "epoch",
        "method",
        "scenario_id",
        "scenario_type",
        "N_UE",
        "distribution",
        "seed",
    ] + METRIC_HEADER
    row_count = len(metrics["reward_sum"])
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for epoch in range(row_count):
            row = [
                epoch,
                method,
                scenario["scenario_id"],
                scenario["scenario_type"],
                scenario["N_UE"],
                scenario["distribution"],
                seed,
            ]
            for key in METRIC_HEADER:
                value = metrics[key][epoch] if epoch < len(metrics[key]) else None
                row.append("" if value is None else value)
            writer.writerow(row)
    return metrics_path


def load_epoch_metrics(metrics_path):
    metrics = {key: [] for key in METRIC_HEADER}
    with Path(metrics_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in METRIC_HEADER:
                value = row.get(key, "")
                metrics[key].append(None if value == "" else float(value))
    return metrics


def add_manifest_row(
    rows,
    run_id,
    method,
    scenario,
    seed,
    source,
    config_path,
    metrics_path,
    status,
    notes,
):
    rows.append(
        {
            "run_id": run_id,
            "method": method,
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "N_UE": scenario["N_UE"],
            "distribution": scenario["distribution"],
            "seed": seed,
            "source": source,
            "config_path": rel(config_path) if config_path else "",
            "metrics_path": rel(metrics_path) if metrics_path else "",
            "status": status,
            "notes": notes,
        }
    )


def write_manifest(output_root, rows):
    header = [
        "run_id",
        "method",
        "scenario_id",
        "scenario_type",
        "N_UE",
        "distribution",
        "seed",
        "source",
        "config_path",
        "metrics_path",
        "status",
        "notes",
    ]
    write_csv_rows(output_root / "run_manifest.csv", header, rows)


def maybe_reuse_existing_run(
    output_root,
    run_id,
    method,
    scenario,
    seed,
    source,
    config_path,
    manifest_rows,
    metric_records,
):
    metrics_path = output_root / "runs" / run_id / "epoch_metrics.csv"
    if not metrics_path.exists():
        return False
    metrics = load_epoch_metrics(metrics_path)
    if len(metrics["reward_sum"]) < 1:
        return False
    metric_records.append(
        {
            "run_id": run_id,
            "method": method,
            "scenario": scenario,
            "seed": seed,
            "source": source,
            "metrics": metrics,
        }
    )
    add_manifest_row(
        manifest_rows,
        run_id,
        method,
        scenario,
        seed,
        source,
        config_path,
        metrics_path,
        "success",
        "reused existing group4 output",
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


def summarize_record(record, tail_window):
    scenario = record["scenario"]
    out = {
        "method": record["method"],
        "scenario_id": scenario["scenario_id"],
        "scenario_type": scenario["scenario_type"],
        "N_UE": scenario["N_UE"],
        "distribution": scenario["distribution"],
        "seed": record["seed"],
        "tail_window": tail_window,
    }
    for key in REQUIRED_KPIS:
        values = [float(v) for v in tail_slice(record["metrics"][key], tail_window)]
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return out


def write_summaries(output_root, metric_records):
    seed_rows = []
    for record in sorted(
        metric_records,
        key=lambda r: (r["scenario"]["scenario_id"], r["method"], r["seed"]),
    ):
        for tail_window in TAIL_WINDOWS:
            seed_rows.append(summarize_record(record, tail_window))

    by_seed_header = [
        "method",
        "scenario_id",
        "scenario_type",
        "N_UE",
        "distribution",
        "seed",
        "tail_window",
        "reward_sum_mean",
        "throughput_mbps_mean",
        "qos_satisfaction_mean",
        "jain_fairness_mean",
        "switch_ratio_mean",
    ]
    write_csv_rows(
        output_root / "summary_by_scenario_seed.csv", by_seed_header, seed_rows
    )

    summary_rows = []
    keys = sorted(
        {
            (
                row["method"],
                row["scenario_id"],
                row["scenario_type"],
                int(row["N_UE"]),
                row["distribution"],
                row["tail_window"],
            )
            for row in seed_rows
        }
    )
    for method, scenario_id, scenario_type, n_ue, distribution, tail_window in keys:
        group = [
            row
            for row in seed_rows
            if row["method"] == method
            and row["scenario_id"] == scenario_id
            and row["tail_window"] == tail_window
        ]
        out = {
            "method": method,
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "N_UE": n_ue,
            "distribution": distribution,
            "n_seed": len(group),
            "tail_window": tail_window,
        }
        for key in REQUIRED_KPIS:
            values = [row[f"{key}_mean"] for row in group]
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(out)

    summary_header = [
        "method",
        "scenario_id",
        "scenario_type",
        "N_UE",
        "distribution",
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
    write_csv_rows(output_root / "summary_by_scenario.csv", summary_header, summary_rows)
    return seed_rows, summary_rows


def unit_for_metric(metric):
    if metric.endswith("mbps"):
        return "Mbps"
    if metric in ["qos_satisfaction", "jain_fairness", "switch_ratio"]:
        return "ratio"
    return "unitless"


def panel_for_metric(metric):
    if metric == "throughput_mbps":
        return "A_throughput"
    if metric in ["qos_satisfaction", "jain_fairness"]:
        return "B_qos_fairness"
    if metric == "switch_ratio":
        return "C_switch_ratio"
    return "supp_reward"


def write_fig13(output_root, metric_records, seed_rows):
    source_by_key = {
        (record["method"], record["scenario"]["scenario_id"]): record["run_id"]
        for record in metric_records
    }
    rows = []
    for seed_row in seed_rows:
        for metric in REQUIRED_KPIS:
            rows.append(
                {
                    "scenario_type": seed_row["scenario_type"],
                    "N_UE": seed_row["N_UE"],
                    "distribution": seed_row["distribution"],
                    "method": seed_row["method"],
                    "seed": seed_row["seed"],
                    "tail_window": seed_row["tail_window"],
                    "metric": metric,
                    "value": seed_row[f"{metric}_mean"],
                    "unit": unit_for_metric(metric),
                    "plot_panel": panel_for_metric(metric),
                    "source_run": source_by_key.get(
                        (seed_row["method"], seed_row["scenario_id"]), ""
                    ),
                    "notes": "reward retained for internal/supplemental checks",
                }
            )
    header = [
        "scenario_type",
        "N_UE",
        "distribution",
        "method",
        "seed",
        "tail_window",
        "metric",
        "value",
        "unit",
        "plot_panel",
        "source_run",
        "notes",
    ]
    write_csv_rows(output_root / "fig13_scalability_density.csv", header, rows)
    return rows


def plot_fig13_preview(plot_root, summary_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary_rows if row["tail_window"] == "last_10pct"]
    if not rows:
        return
    scale_rows = [
        row
        for row in rows
        if row["scenario_type"] == "scale_sweep" and row["distribution"] == "uniform"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for method in METHODS:
        group = sorted(
            [row for row in scale_rows if row["method"] == method],
            key=lambda r: int(r["N_UE"]),
        )
        if group:
            axes[0].plot(
                [int(row["N_UE"]) for row in group],
                [row["throughput_mbps_mean"] for row in group],
                marker="o",
                label=method,
            )
    axes[0].set_title("Scale Sweep Throughput")
    axes[0].set_ylabel("Mbps")
    axes[0].legend(fontsize=7)

    for metric, label in [
        ("qos_satisfaction_mean", "QoS"),
        ("jain_fairness_mean", "Jain"),
    ]:
        for method in METHODS:
            group = sorted(
                [row for row in scale_rows if row["method"] == method],
                key=lambda r: int(r["N_UE"]),
            )
            if group:
                axes[1].plot(
                    [int(row["N_UE"]) for row in group],
                    [row[metric] for row in group],
                    marker="o",
                    label=f"{method}:{label}",
                )
    axes[1].set_title("Scale Sweep QoS/Fairness")
    axes[1].legend(fontsize=6)

    density = [
        row
        for row in rows
        if row["N_UE"] == 100 and row["method"] in METHODS
    ]
    labels = [f"{row['method']}\n{row['distribution']}" for row in density]
    axes[2].bar(range(len(density)), [row["throughput_mbps_mean"] for row in density])
    axes[2].set_title("N=100 Distribution Throughput")
    axes[2].set_xticks(range(len(density)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[3].bar(range(len(density)), [row["switch_ratio_mean"] for row in density])
    axes[3].set_title("N=100 Switch Ratio")
    axes[3].set_xticks(range(len(density)))
    axes[3].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Scenario")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "fig13_preview_scalability_density.png", dpi=300)
    plt.close(fig)


def plot_training_curves(plot_root, metric_records):
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(REQUIRED_KPIS), 1, figsize=(10, 13), sharex=False)
    for idx, metric in enumerate(REQUIRED_KPIS):
        ax = axes[idx]
        for record in metric_records:
            y = record["metrics"][metric]
            label = f"{record['method']}:{record['scenario']['scenario_id']}"
            ax.plot(range(len(y)), y, linewidth=0.7, alpha=0.75, label=label)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=5)
    fig.tight_layout()
    fig.savefig(plot_root / "scalability_training_curves.png", dpi=300)
    plt.close(fig)


def plot_density_summary(plot_root, summary_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in summary_rows
        if row["tail_window"] == "last_10pct"
        and row["N_UE"] == 100
        and row["scenario_type"] in ["scale_sweep", "density_stress"]
    ]
    if not rows:
        return
    metrics = [
        ("throughput_mbps_mean", "Throughput"),
        ("qos_satisfaction_mean", "QoS"),
        ("jain_fairness_mean", "Jain"),
        ("switch_ratio_mean", "Switch"),
    ]
    labels = [f"{row['method']}\n{row['distribution']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.5))
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(x, [row[metric] for row in rows])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "density_hotspot_kpi_summary.png", dpi=300)
    plt.close(fig)


def plot_user_distribution(plot_root, seed):
    plot_root.mkdir(parents=True, exist_ok=True)
    uniform = users_for_scenario(seed, scenario_by_id("scale_uniform_N100"))
    hotspot = users_for_scenario(seed, scenario_by_id("density_hotspot_N100"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, users, title in [
        (axes[0], uniform, "Uniform N=100"),
        (axes[1], hotspot, "Hotspot N=100"),
    ]:
        ax.scatter([u.x for u in users], [u.y for u in users], s=12, alpha=0.8)
        if title.startswith("Hotspot"):
            ax.scatter(
                [c[0] for c in HOTSPOT_CENTERS],
                [c[1] for c in HOTSPOT_CENTERS],
                marker="x",
                s=80,
                color="black",
            )
        ax.set_title(title)
        ax.set_xlim(0, AREA_SIZE_X)
        ax.set_ylim(0, AREA_SIZE_Y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_root / "user_distribution_uniform_vs_hotspot.png", dpi=300)
    plt.close(fig)


def validate_outputs(output_root, expected_records, require_10000):
    manifest_path = output_root / "run_manifest.csv"
    fig_path = output_root / "fig13_scalability_density.csv"
    summary_path = output_root / "summary_by_scenario.csv"
    for path in [manifest_path, fig_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    success_keys = {
        (row["method"], row["scenario_id"])
        for row in manifest
        if row["status"] == "success"
    }
    missing = set(expected_records) - success_keys
    if missing:
        raise ValueError(f"manifest missing successful records: {sorted(missing)}")

    if require_10000:
        for row in manifest:
            if row["status"] != "success":
                continue
            metrics_path = REPO_DIR / row["metrics_path"]
            with metrics_path.open("r", newline="", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            if row_count != 10000:
                raise ValueError(
                    f"{row['run_id']} has {row_count} epoch rows, expected 10000"
                )

    with fig_path.open("r", newline="", encoding="utf-8") as f:
        fig_rows = list(csv.DictReader(f))
    for method, scenario_id in expected_records:
        scenario = scenario_by_id(scenario_id)
        for tail_window in TAIL_WINDOWS:
            kpi_rows = [
                row
                for row in fig_rows
                if row["method"] == method
                and int(row["N_UE"]) == scenario["N_UE"]
                and row["distribution"] == scenario["distribution"]
                and row["tail_window"] == tail_window
            ]
            metrics = {row["metric"] for row in kpi_rows}
            missing_metrics = set(REQUIRED_KPIS) - metrics
            if missing_metrics:
                raise ValueError(
                    f"fig13 missing metrics for {method}, {scenario_id}, {tail_window}: {sorted(missing_metrics)}"
                )
            if len(kpi_rows) != len(REQUIRED_KPIS):
                raise ValueError(
                    f"fig13 has {len(kpi_rows)} rows for {method}, {scenario_id}, {tail_window}"
                )

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    for row in summary_rows:
        if int(row["n_seed"]) != 1:
            raise ValueError("expected n_seed=1 in Group 4 summary")


def get_metrics_for_record(method, scenario, seed, max_epochs, device):
    if method == "hmadqn_k3" and scenario["scenario_id"] == "scale_uniform_N100":
        try:
            return read_original_hmadqn_metrics(), "original_logs", "reused original H-MADQN K=3 seed42 logs"
        except Exception:
            metrics = run_hmadqn(seed, scenario, max_epochs, device)
            return metrics, "trained_hmadqn", "original logs unavailable; reran strict Group 4 H-MADQN"
    if method == "greedy_rsrp_capacity" and scenario["scenario_id"] == "scale_uniform_N100":
        try:
            return read_group1_greedy_metrics(), "group1_logs", "reused Group 1 Greedy seed42 logs; default N=100 uniform path"
        except Exception:
            metrics = run_greedy(seed, scenario, max_epochs)
            return metrics, "simulated_greedy", "Group 1 Greedy reuse rejected; reran strict Group 4 Greedy"
    if method == "hmadqn_k3":
        return run_hmadqn(seed, scenario, max_epochs, device), "trained_hmadqn", "completed Group 4 H-MADQN run"
    if method == "greedy_rsrp_capacity":
        return run_greedy(seed, scenario, max_epochs), "simulated_greedy", "completed Group 4 Greedy simulation"
    raise ValueError(f"unknown method: {method}")


def run_all(
    output_root,
    plot_root,
    records,
    max_epochs,
    seed,
    device_name,
    resume,
    skip_plots,
    require_10000,
):
    ensure_dirs(output_root)
    density_check = validate_user_generation(seed)
    config_path = write_method_config(output_root, max_epochs, device_name, density_check)
    write_csv_rows(
        output_root / "user_distribution_preview.csv",
        ["scenario_id", "distribution", "seed", "global_index", "x", "y", "user_type"],
        user_position_rows(seed),
    )
    manifest_rows = []
    metric_records = []
    device = device_from_name(device_name)

    for method, scenario_id in records:
        scenario = scenario_by_id(scenario_id)
        run_id = run_id_for(method, scenario_id, seed)
        try:
            default_source = (
                "original_logs"
                if method == "hmadqn_k3" and scenario_id == "scale_uniform_N100"
                else "group1_logs"
                if method == "greedy_rsrp_capacity" and scenario_id == "scale_uniform_N100"
                else "trained_hmadqn"
                if method == "hmadqn_k3"
                else "simulated_greedy"
            )
            if resume and maybe_reuse_existing_run(
                output_root,
                run_id,
                method,
                scenario,
                seed,
                default_source,
                config_path,
                manifest_rows,
                metric_records,
            ):
                write_manifest(output_root, manifest_rows)
                continue
            metrics, source, notes = get_metrics_for_record(
                method, scenario, seed, max_epochs, device
            )
            metrics_path = write_epoch_metrics(
                output_root, run_id, method, scenario, seed, metrics
            )
            metric_records.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "source": source,
                    "metrics": metrics,
                }
            )
            add_manifest_row(
                manifest_rows,
                run_id,
                method,
                scenario,
                seed,
                source,
                config_path,
                metrics_path,
                "success",
                f"{notes}; rows={len(metrics['reward_sum'])}",
            )
        except Exception as exc:
            add_manifest_row(
                manifest_rows,
                run_id,
                method,
                scenario,
                seed,
                default_source,
                config_path,
                "",
                "failed",
                str(exc),
            )
        write_manifest(output_root, manifest_rows)

    seed_rows, summary_rows = write_summaries(output_root, metric_records)
    fig_rows = write_fig13(output_root, metric_records, seed_rows)
    validate_outputs(output_root, records, require_10000)
    if not skip_plots:
        plot_fig13_preview(plot_root, summary_rows)
        plot_training_curves(plot_root, metric_records)
        plot_density_summary(plot_root, summary_rows)
        plot_user_distribution(plot_root, seed)
    return manifest_rows, metric_records, fig_rows


def all_records():
    return [
        (method, scenario["scenario_id"])
        for scenario in SCENARIOS
        for method in METHODS
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run revision round 1 Group 4 scalability and density outputs."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    smoke = args.mode == "smoke"
    output_root = LOG_ROOT / "smoke" if smoke else LOG_ROOT
    plot_root = PLOT_ROOT / "smoke" if smoke else PLOT_ROOT
    records = SMOKE_RECORDS if smoke else all_records()
    max_epochs = args.smoke_epochs if smoke else args.max_epochs
    manifest_rows, metric_records, _ = run_all(
        output_root,
        plot_root,
        records,
        max_epochs,
        args.seed,
        args.device,
        args.resume,
        args.skip_plots,
        require_10000=not smoke,
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
