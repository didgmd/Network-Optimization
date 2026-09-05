# -*- coding: utf-8 -*-
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Parameters import (
    CLOUD_ASSOCIATION_PERIOD,
    CLOUD_REWARD_WEIGHTS,
    CLOUD_STICKINESS_BONUS,
    CLOUD_SWITCH_PENALTY,
    EDGE_REWARD_WEIGHTS,
    MAX_EPOCHS,
)
from SC_RL_main import build_sdm_order_indices, rl
from Topology import define_topology, define_users


REPO_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_DIR / "logs" / "revision_round1" / "group5_sdm_reward"
PLOT_ROOT = REPO_DIR / "plots" / "revision_round1" / "group5_sdm_reward"
GROUP1_ROOT = REPO_DIR / "logs" / "revision_round1" / "group1_baselines"
METHOD = "hmadqn_k3_sdm_order"
SEEDS = [42, 43, 44]
ORDER_MODES = ["fixed_order", "random_per_cloud_epoch_order"]
TAIL_WINDOWS = ["last_10pct", "last_1000"]
REQUIRED_KPIS = [
    "reward_sum",
    "throughput_mbps",
    "qos_satisfaction",
    "jain_fairness",
    "switch_ratio",
]
LOSS_KEYS = ["cloud_loss", "macro_loss", "small_loss", "uav_loss"]
METRIC_HEADER = REQUIRED_KPIS + LOSS_KEYS
FIG_HASH_PATTERN = "fig*.csv"

ABLATION_COMPONENTS = {
    "baseline": "none",
    "A1_no_qos": "qos",
    "A2_no_thr": "throughput",
    "A3_no_load_cap": "load_and_capacity",
    "A4_no_stability": "switch_penalty_and_stickiness_bonus",
    "A5_no_fair": "fairness",
}


def rel(path):
    if not path:
        return ""
    return str(Path(path).resolve().relative_to(REPO_DIR))


def ensure_dirs(output_root, plot_root):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    (output_root / "method_configs").mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_name(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fig_hash_rows():
    rows = []
    for path in sorted((REPO_DIR / "logs").glob(FIG_HASH_PATTERN)):
        if path.is_file():
            rows.append(
                {
                    "file": rel(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
            )
    return rows


def write_csv_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_method_config(output_root, max_epochs, device_name, records):
    config_path = output_root / "method_configs" / "group5_sdm_reward_config.json"
    write_json(
        config_path,
        {
            "method": METHOD,
            "order_modes": ORDER_MODES,
            "seeds": SEEDS,
            "selected_records": records,
            "cloud_association_period": CLOUD_ASSOCIATION_PERIOD,
            "epsilon_schedule": "exp2",
            "convergence_check": False,
            "max_epochs": max_epochs,
            "execution_device": device_name,
            "sdm_order_random_policy": (
                "random.Random(seed + cloud_epoch * 1000003).shuffle(indices)"
            ),
            "identity_binding": (
                "shuffled processing order uses original user_idx for historical "
                "actions, previous_bs, reward, replay transition, switch count, "
                "and metrics"
            ),
            "reuse_policy": {
                "fixed_seed42": "Group 1 H-MADQN original_logs",
                "fixed_seed43_44": "Group 1 reproduced_hmadqn logs",
            },
            "cloud_reward_weights": CLOUD_REWARD_WEIGHTS,
            "edge_reward_weights": EDGE_REWARD_WEIGHTS,
            "stability_terms": {
                "switch_penalty": CLOUD_SWITCH_PENALTY,
                "stickiness_bonus": CLOUD_STICKINESS_BONUS,
            },
            "claim_boundary": (
                "fixed vs random per-cloud-epoch SDM order sensitivity and "
                "component-level reward sensitivity only; not unbiased SDM proof "
                "or exhaustive reward-weight grid search"
            ),
        },
    )
    return config_path


def validate_sdm_order_helper():
    fixed = build_sdm_order_indices(100)
    if fixed != list(range(100)):
        raise ValueError("default SDM order is not fixed order")
    a = build_sdm_order_indices(
        100, mode="random_per_cloud_epoch_order", seed=42, cloud_epoch=7
    )
    b = build_sdm_order_indices(
        100, mode="random_per_cloud_epoch_order", seed=42, cloud_epoch=7
    )
    if a != b:
        raise ValueError("random SDM order is not reproducible for same seed/epoch")
    if sorted(a) != list(range(100)):
        raise ValueError("random SDM order is not a permutation")
    changed = any(
        build_sdm_order_indices(
            100, mode="random_per_cloud_epoch_order", seed=42, cloud_epoch=epoch
        )
        != list(range(100))
        for epoch in range(5)
    )
    if not changed:
        raise ValueError("random SDM order did not differ from fixed order")


def run_hmadqn_order(seed, order_mode, max_epochs, device):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()
    config = {
        "seed": seed,
        "max_epochs": max_epochs,
        "cloud_association_period": CLOUD_ASSOCIATION_PERIOD,
        "epsilon_schedule": "exp2",
        "convergence_check": False,
        "switch_penalty": CLOUD_SWITCH_PENALTY,
        "stickiness_bonus": CLOUD_STICKINESS_BONUS,
        "sdm_order_mode": order_mode,
        "sdm_order_seed": seed,
    }
    (
        losses,
        _rewards,
        throughput_history,
        reward_sum_history,
        qos_history,
        jain_history,
        _eps_histories,
        _user_rate_history,
        _bs_load_history,
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


def load_epoch_metrics(metrics_path):
    metrics = {key: [] for key in METRIC_HEADER}
    with Path(metrics_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in METRIC_HEADER:
                value = row.get(key, "")
                metrics[key].append(None if value == "" else float(value))
    return metrics


def write_epoch_metrics(output_root, run_id, order_mode, seed, metrics):
    run_path = output_root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "epoch_metrics.csv"
    row_count = len(metrics["reward_sum"])
    header = ["epoch", "method", "order_mode", "seed"] + METRIC_HEADER
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for epoch in range(row_count):
            row = [epoch, METHOD, order_mode, seed]
            for key in METRIC_HEADER:
                value = metrics[key][epoch] if epoch < len(metrics[key]) else None
                row.append("" if value is None else value)
            writer.writerow(row)
    return metrics_path


def run_id_for(order_mode, seed):
    if order_mode == "fixed_order":
        return f"hmadqn_k3_sdm_fixed_order_seed{seed}"
    return f"hmadqn_k3_sdm_random_order_seed{seed}"


def group1_run_id_for_seed(seed):
    if seed == 42:
        return "hmadqn_k3_original_seed42"
    return f"hmadqn_k3_reproduced_seed{seed}"


def read_group1_hmadqn_metrics(seed, min_epochs):
    manifest_path = GROUP1_ROOT / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    group1_run_id = group1_run_id_for_seed(seed)
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row
        for row in rows
        if row.get("run_id") == group1_run_id and row.get("status") == "success"
    ]
    if not matches:
        raise ValueError(f"Group 1 run not available: {group1_run_id}")
    metrics_path = REPO_DIR / matches[0]["metrics_path"]
    metrics = load_epoch_metrics(metrics_path)
    if len(metrics["reward_sum"]) < min_epochs:
        raise ValueError(
            f"Group 1 run {group1_run_id} has {len(metrics['reward_sum'])} rows"
        )
    source = "original_logs" if seed == 42 else "reproduced_group1_logs"
    notes = (
        f"reused Group 1 {group1_run_id}; original source={matches[0].get('source')}"
    )
    return metrics, source, notes


def tail_slice(values, tail_window):
    clean = [v for v in values if v is not None]
    if not clean:
        return []
    if tail_window == "last_10pct":
        count = max(int(len(clean) * 0.1), 1)
    elif tail_window == "last_1000":
        count = min(1000, len(clean))
    else:
        raise ValueError(f"unknown tail window: {tail_window}")
    return clean[-count:]


def mean_tail(values, tail_window):
    segment = tail_slice(values, tail_window)
    return float(np.mean(segment)) if segment else 0.0


def summarize_record(record, tail_window):
    row = {
        "method": METHOD,
        "order_mode": record["order_mode"],
        "seed": record["seed"],
        "tail_window": tail_window,
    }
    for key in REQUIRED_KPIS:
        row[f"{key}_mean"] = mean_tail(record["metrics"][key], tail_window)
    return row


def write_summaries(output_root, metric_records):
    seed_rows = []
    for record in metric_records:
        for tail_window in TAIL_WINDOWS:
            seed_rows.append(summarize_record(record, tail_window))

    by_seed_header = [
        "method",
        "order_mode",
        "seed",
        "tail_window",
        "reward_sum_mean",
        "throughput_mbps_mean",
        "qos_satisfaction_mean",
        "jain_fairness_mean",
        "switch_ratio_mean",
    ]
    write_csv_rows(output_root / "summary_by_order_seed.csv", by_seed_header, seed_rows)

    order_rows = []
    for order_mode in ORDER_MODES:
        for tail_window in TAIL_WINDOWS:
            group = [
                row
                for row in seed_rows
                if row["order_mode"] == order_mode
                and row["tail_window"] == tail_window
            ]
            if not group:
                continue
            out = {"method": METHOD, "order_mode": order_mode, "n_seed": len(group)}
            out["tail_window"] = tail_window
            for key in REQUIRED_KPIS:
                values = [row[f"{key}_mean"] for row in group]
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            order_rows.append(out)

    by_order_header = [
        "method",
        "order_mode",
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
    write_csv_rows(output_root / "summary_by_order.csv", by_order_header, order_rows)
    return seed_rows, order_rows


def unit_for_metric(metric):
    if metric == "throughput_mbps":
        return "Mbps"
    if metric in {"qos_satisfaction", "jain_fairness", "switch_ratio"}:
        return "ratio"
    return "scalar"


def panel_for_metric(metric):
    return {
        "throughput_mbps": "A_throughput",
        "qos_satisfaction": "B_qos",
        "jain_fairness": "C_jain",
        "switch_ratio": "D_switch_ratio",
        "reward_sum": "supp_reward",
    }[metric]


def write_fig14(output_root, metric_records, seed_rows):
    source_by_key = {
        (record["order_mode"], int(record["seed"])): record["run_id"]
        for record in metric_records
    }
    rows = []
    for seed_row in seed_rows:
        for metric in REQUIRED_KPIS:
            rows.append(
                {
                    "order_mode": seed_row["order_mode"],
                    "seed": seed_row["seed"],
                    "tail_window": seed_row["tail_window"],
                    "metric": metric,
                    "value": seed_row[f"{metric}_mean"],
                    "unit": unit_for_metric(metric),
                    "plot_panel": panel_for_metric(metric),
                    "source_run": source_by_key[
                        (seed_row["order_mode"], int(seed_row["seed"]))
                    ],
                    "notes": (
                        "fixed-order rows reuse Group 1 H-MADQN where available; "
                        "random-order rows change only SDM processing order"
                    ),
                }
            )
    header = [
        "order_mode",
        "seed",
        "tail_window",
        "metric",
        "value",
        "unit",
        "plot_panel",
        "source_run",
        "notes",
    ]
    write_csv_rows(output_root / "fig14_sdm_order_sensitivity.csv", header, rows)
    return rows


def reward_weight_rows():
    return [
        {
            "layer": "cloud",
            "term": "qos",
            "weight_or_value": CLOUD_REWARD_WEIGHTS["qos"],
            "role": "service satisfaction priority",
            "notes": "exported from Parameters.CLOUD_REWARD_WEIGHTS",
        },
        {
            "layer": "cloud",
            "term": "load",
            "weight_or_value": CLOUD_REWARD_WEIGHTS["load"],
            "role": "avoid overloaded serving nodes",
            "notes": "exported from Parameters.CLOUD_REWARD_WEIGHTS",
        },
        {
            "layer": "cloud",
            "term": "thr",
            "weight_or_value": CLOUD_REWARD_WEIGHTS["thr"],
            "role": "throughput potential / efficiency",
            "notes": "exported from Parameters.CLOUD_REWARD_WEIGHTS",
        },
        {
            "layer": "cloud",
            "term": "cap",
            "weight_or_value": CLOUD_REWARD_WEIGHTS["cap"],
            "role": "capacity preference",
            "notes": "exported from Parameters.CLOUD_REWARD_WEIGHTS",
        },
        {
            "layer": "edge",
            "term": "qos",
            "weight_or_value": EDGE_REWARD_WEIGHTS["qos"],
            "role": "local service satisfaction",
            "notes": "exported from Parameters.EDGE_REWARD_WEIGHTS",
        },
        {
            "layer": "edge",
            "term": "fair",
            "weight_or_value": EDGE_REWARD_WEIGHTS["fair"],
            "role": "user-level fairness",
            "notes": "exported from Parameters.EDGE_REWARD_WEIGHTS",
        },
        {
            "layer": "edge",
            "term": "thr",
            "weight_or_value": EDGE_REWARD_WEIGHTS["thr"],
            "role": "local throughput efficiency",
            "notes": "exported from Parameters.EDGE_REWARD_WEIGHTS",
        },
        {
            "layer": "stability",
            "term": "switch_penalty",
            "weight_or_value": CLOUD_SWITCH_PENALTY,
            "role": "association stability regularizer",
            "notes": "exported from Parameters.CLOUD_SWITCH_PENALTY",
        },
        {
            "layer": "stability",
            "term": "stickiness_bonus",
            "weight_or_value": CLOUD_STICKINESS_BONUS,
            "role": "association persistence regularizer",
            "notes": "exported from Parameters.CLOUD_STICKINESS_BONUS",
        },
    ]


def write_reward_weights(output_root):
    header = ["layer", "term", "weight_or_value", "role", "notes"]
    rows = reward_weight_rows()
    write_csv_rows(output_root / "table_reward_weights.csv", header, rows)
    return rows


def load_fig9_metric_rows():
    path = REPO_DIR / "logs" / "fig9_kpi_distribution.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"]
            if run_id not in ABLATION_COMPONENTS:
                continue
            bucket = data.setdefault(
                run_id,
                {
                    "reward_sum": [],
                    "throughput_mbps": [],
                    "qos_satisfaction": [],
                    "jain_fairness": [],
                },
            )
            bucket["reward_sum"].append(float(row["reward_sum"]))
            bucket["throughput_mbps"].append(float(row["throughput"]))
            bucket["qos_satisfaction"].append(float(row["qos"]))
            bucket["jain_fairness"].append(float(row["jain"]))
    return data


def load_fig7_switch_rows():
    path = REPO_DIR / "logs" / "fig7_stability_fairness_tradeoff.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", "")
            if run_id not in ABLATION_COMPONENTS:
                continue
            value = row.get("switch_ratio", "")
            if value == "":
                continue
            data.setdefault(run_id, []).append(float(value))
    return data


def write_reward_component_sensitivity(output_root):
    kpi_data = load_fig9_metric_rows()
    switch_data = load_fig7_switch_rows()
    rows = []
    for variant, component in ABLATION_COMPONENTS.items():
        if variant not in kpi_data:
            continue
        for tail_window in TAIL_WINDOWS:
            for metric, values in kpi_data[variant].items():
                rows.append(
                    {
                        "variant": variant,
                        "removed_or_changed_component": component,
                        "tail_window": tail_window,
                        "metric": metric,
                        "value": mean_tail(values, tail_window),
                        "unit": unit_for_metric(metric),
                        "source_run": "logs/fig9_kpi_distribution.csv",
                        "notes": (
                            "component-level sensitivity evidence; not exhaustive "
                            "reward-weight optimization"
                        ),
                    }
                )
            if variant in switch_data:
                rows.append(
                    {
                        "variant": variant,
                        "removed_or_changed_component": component,
                        "tail_window": tail_window,
                        "metric": "switch_ratio",
                        "value": mean_tail(switch_data[variant], tail_window),
                        "unit": "ratio",
                        "source_run": "logs/fig7_stability_fairness_tradeoff.csv",
                        "notes": (
                            "switch ratio available only for baseline/A4/A5 in "
                            "existing Fig.7 logs"
                        ),
                    }
                )
    header = [
        "variant",
        "removed_or_changed_component",
        "tail_window",
        "metric",
        "value",
        "unit",
        "source_run",
        "notes",
    ]
    write_csv_rows(
        output_root / "reward_component_sensitivity_summary.csv", header, rows
    )
    return rows


def add_manifest_row(
    rows, run_id, order_mode, seed, source, config_path, metrics_path, status, notes
):
    rows.append(
        {
            "run_id": run_id,
            "method": METHOD,
            "order_mode": order_mode,
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
        "order_mode",
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
    order_mode,
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
            "order_mode": order_mode,
            "seed": seed,
            "source": source,
            "metrics": metrics,
        }
    )
    add_manifest_row(
        manifest_rows,
        run_id,
        order_mode,
        seed,
        source,
        config_path,
        metrics_path,
        "success",
        f"reused existing Group 5 output; rows={len(metrics['reward_sum'])}",
    )
    return True


def get_metrics_for_record(order_mode, seed, max_epochs, device):
    if order_mode == "fixed_order":
        try:
            return read_group1_hmadqn_metrics(seed, max_epochs)
        except Exception as exc:
            metrics = run_hmadqn_order(seed, order_mode, max_epochs, device)
            return (
                metrics,
                "reproduced_fixed_order",
                f"Group 1 reuse rejected ({exc}); reran fixed-order H-MADQN",
            )
    metrics = run_hmadqn_order(seed, order_mode, max_epochs, device)
    return metrics, "trained_random_order", "completed random-order H-MADQN run"


def parse_records(record_args):
    if not record_args:
        return [(order_mode, seed) for order_mode in ORDER_MODES for seed in SEEDS]
    records = []
    for token in record_args:
        if token == "all":
            return [(order_mode, seed) for order_mode in ORDER_MODES for seed in SEEDS]
        if ":" not in token:
            raise ValueError(f"record must be order_mode:seed, got {token}")
        order_mode, seed_text = token.split(":", 1)
        if order_mode not in ORDER_MODES:
            raise ValueError(f"unknown order mode: {order_mode}")
        seed = int(seed_text)
        records.append((order_mode, seed))
    return records


def plot_fig14_preview(plot_root, order_rows):
    rows = [row for row in order_rows if row["tail_window"] == "last_10pct"]
    metrics = [
        "throughput_mbps",
        "qos_satisfaction",
        "jain_fairness",
        "switch_ratio",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    x = np.arange(len(ORDER_MODES))
    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for mode in ORDER_MODES:
            match = [row for row in rows if row["order_mode"] == mode]
            if match:
                means.append(float(match[0][f"{metric}_mean"]))
                stds.append(float(match[0][f"{metric}_std"]))
            else:
                means.append(0.0)
                stds.append(0.0)
        ax.bar(x, means, yerr=stds, capsize=4, color=["#4c78a8", "#f58518"])
        ax.set_xticks(x)
        ax.set_xticklabels(["fixed", "random"], rotation=0)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_root / "fig14_preview_sdm_order_sensitivity.png", dpi=300)
    plt.close(fig)


def plot_training_curves(plot_root, metric_records):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    metrics = [
        "throughput_mbps",
        "qos_satisfaction",
        "jain_fairness",
        "switch_ratio",
    ]
    for ax, metric in zip(axes, metrics):
        for record in metric_records:
            values = record["metrics"][metric]
            if not values:
                continue
            step = max(len(values) // 1000, 1)
            xs = list(range(0, len(values), step))
            ys = [values[i] for i in xs]
            label = f"{record['order_mode'].replace('_order', '')}-s{record['seed']}"
            ax.plot(xs, ys, linewidth=0.8, alpha=0.75, label=label)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_root / "sdm_order_training_curves.png", dpi=300)
    plt.close(fig)


def plot_reward_component_summary(plot_root, reward_rows):
    rows = [
        row
        for row in reward_rows
        if row["tail_window"] == "last_10pct"
        and row["metric"]
        in {"throughput_mbps", "qos_satisfaction", "jain_fairness", "switch_ratio"}
    ]
    variants = list(ABLATION_COMPONENTS.keys())
    metrics = ["throughput_mbps", "qos_satisfaction", "jain_fairness", "switch_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        values = []
        labels = []
        for variant in variants:
            matches = [
                row for row in rows if row["variant"] == variant and row["metric"] == metric
            ]
            if matches:
                labels.append(variant.replace("_", "\n"))
                values.append(float(matches[0]["value"]))
        ax.bar(range(len(values)), values, color="#54a24b")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_root / "reward_component_sensitivity_summary.png", dpi=300)
    plt.close(fig)


def plot_reward_weights_table(plot_root, weight_rows):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis("off")
    cell_text = [
        [
            row["layer"],
            row["term"],
            f"{float(row['weight_or_value']):.3f}",
            row["role"],
        ]
        for row in weight_rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["layer", "term", "value", "role"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(plot_root / "reward_weights_table_preview.png", dpi=300)
    plt.close(fig)


def validate_weight_table(output_root):
    path = output_root / "table_reward_weights.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    actual = {
        (row["layer"], row["term"]): float(row["weight_or_value"]) for row in rows
    }
    expected = {
        ("cloud", "qos"): CLOUD_REWARD_WEIGHTS["qos"],
        ("cloud", "load"): CLOUD_REWARD_WEIGHTS["load"],
        ("cloud", "thr"): CLOUD_REWARD_WEIGHTS["thr"],
        ("cloud", "cap"): CLOUD_REWARD_WEIGHTS["cap"],
        ("edge", "qos"): EDGE_REWARD_WEIGHTS["qos"],
        ("edge", "fair"): EDGE_REWARD_WEIGHTS["fair"],
        ("edge", "thr"): EDGE_REWARD_WEIGHTS["thr"],
        ("stability", "switch_penalty"): CLOUD_SWITCH_PENALTY,
        ("stability", "stickiness_bonus"): CLOUD_STICKINESS_BONUS,
    }
    if actual != expected:
        raise ValueError(f"reward weight table mismatch: {actual} != {expected}")


def validate_hashes(output_root):
    before_path = output_root / "source_fig_hashes_before.csv"
    after_path = output_root / "source_fig_hashes_after.csv"
    with before_path.open("r", newline="", encoding="utf-8") as f:
        before = {row["file"]: row["sha256"] for row in csv.DictReader(f)}
    with after_path.open("r", newline="", encoding="utf-8") as f:
        after = {row["file"]: row["sha256"] for row in csv.DictReader(f)}
    if before != after:
        raise ValueError("source fig hashes changed")


def validate_outputs(output_root, expected_records, require_10000):
    manifest_path = output_root / "run_manifest.csv"
    fig14_path = output_root / "fig14_sdm_order_sensitivity.csv"
    summary_path = output_root / "summary_by_order.csv"
    reward_sensitivity_path = output_root / "reward_component_sensitivity_summary.csv"
    for path in [manifest_path, fig14_path, summary_path, reward_sensitivity_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    present = {
        (row["order_mode"], int(row["seed"]))
        for row in manifest
        if row["status"] == "success"
    }
    missing = set(expected_records) - present
    if missing:
        raise ValueError(f"manifest missing success records: {sorted(missing)}")
    if require_10000:
        for row in manifest:
            if row["status"] != "success":
                continue
            metrics_path = REPO_DIR / row["metrics_path"]
            with metrics_path.open("r", newline="", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            if row_count != 10000:
                raise ValueError(
                    f"{row['run_id']} has {row_count} rows, expected 10000"
                )

    with fig14_path.open("r", newline="", encoding="utf-8") as f:
        fig_rows = list(csv.DictReader(f))
    for order_mode, seed in expected_records:
        for tail_window in TAIL_WINDOWS:
            rows = [
                row
                for row in fig_rows
                if row["order_mode"] == order_mode
                and int(row["seed"]) == int(seed)
                and row["tail_window"] == tail_window
            ]
            metrics = {row["metric"] for row in rows}
            missing_metrics = set(REQUIRED_KPIS) - metrics
            if len(rows) != len(REQUIRED_KPIS) or missing_metrics:
                raise ValueError(
                    f"fig14 coverage mismatch for {order_mode} seed {seed} "
                    f"{tail_window}: rows={len(rows)} missing={sorted(missing_metrics)}"
                )

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    expected_order_modes = sorted({order_mode for order_mode, _seed in expected_records})
    for order_mode in expected_order_modes:
        for tail_window in TAIL_WINDOWS:
            matches = [
                row
                for row in summary_rows
                if row["order_mode"] == order_mode
                and row["tail_window"] == tail_window
            ]
            if not matches:
                raise ValueError(f"summary missing {order_mode} {tail_window}")
            if int(matches[0]["n_seed"]) != 3 and len(expected_records) == 6:
                raise ValueError(f"expected n_seed=3 for {order_mode}")

    with reward_sensitivity_path.open("r", newline="", encoding="utf-8") as f:
        reward_rows = list(csv.DictReader(f))
    variants = {row["variant"] for row in reward_rows}
    missing_variants = set(ABLATION_COMPONENTS) - variants
    if missing_variants:
        raise ValueError(
            f"reward sensitivity missing variants: {sorted(missing_variants)}"
        )
    validate_weight_table(output_root)
    validate_hashes(output_root)


def run_all(
    output_root,
    plot_root,
    records,
    max_epochs,
    device_name,
    resume,
    skip_plots,
    require_10000,
):
    ensure_dirs(output_root, plot_root)
    validate_sdm_order_helper()
    hash_header = ["file", "sha256", "size_bytes", "mtime"]
    write_csv_rows(output_root / "source_fig_hashes_before.csv", hash_header, source_fig_hash_rows())
    config_path = write_method_config(output_root, max_epochs, device_name, records)
    weight_rows = write_reward_weights(output_root)
    reward_rows = write_reward_component_sensitivity(output_root)

    manifest_rows = []
    metric_records = []
    device = device_from_name(device_name)
    min_epochs = 10000 if require_10000 else max_epochs

    for order_mode, seed in records:
        run_id = run_id_for(order_mode, seed)
        default_source = (
            "original_logs"
            if order_mode == "fixed_order" and seed == 42
            else "reproduced_group1_logs"
            if order_mode == "fixed_order"
            else "trained_random_order"
        )
        try:
            if resume and maybe_reuse_existing_run(
                output_root,
                run_id,
                order_mode,
                seed,
                default_source,
                config_path,
                manifest_rows,
                metric_records,
                min_epochs,
            ):
                write_manifest(output_root, manifest_rows)
                continue
            metrics, source, notes = get_metrics_for_record(
                order_mode, seed, max_epochs, device
            )
            metrics_path = write_epoch_metrics(
                output_root, run_id, order_mode, seed, metrics
            )
            metric_records.append(
                {
                    "run_id": run_id,
                    "order_mode": order_mode,
                    "seed": seed,
                    "source": source,
                    "metrics": metrics,
                }
            )
            add_manifest_row(
                manifest_rows,
                run_id,
                order_mode,
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
                order_mode,
                seed,
                default_source,
                config_path,
                "",
                "failed",
                str(exc),
            )
        write_manifest(output_root, manifest_rows)

    seed_rows, order_rows = write_summaries(output_root, metric_records)
    fig_rows = write_fig14(output_root, metric_records, seed_rows)
    write_csv_rows(output_root / "source_fig_hashes_after.csv", hash_header, source_fig_hash_rows())
    validate_outputs(output_root, records, require_10000)
    if not skip_plots:
        plot_fig14_preview(plot_root, order_rows)
        plot_training_curves(plot_root, metric_records)
        plot_reward_component_summary(plot_root, reward_rows)
        plot_reward_weights_table(plot_root, weight_rows)
    return manifest_rows, metric_records, fig_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Group 5 SDM order and reward evidence")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records = parse_records(args.records)
    if args.mode == "smoke":
        output_root = OUTPUT_ROOT / "smoke"
        plot_root = PLOT_ROOT / "smoke"
        max_epochs = args.smoke_epochs
        require_10000 = False
    else:
        output_root = OUTPUT_ROOT
        plot_root = PLOT_ROOT
        max_epochs = args.max_epochs
        require_10000 = max_epochs == 10000

    manifest_rows, metric_records, fig_rows = run_all(
        output_root=output_root,
        plot_root=plot_root,
        records=records,
        max_epochs=max_epochs,
        device_name=args.device,
        resume=args.resume,
        skip_plots=args.skip_plots,
        require_10000=require_10000,
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
                "fig14_rows": len(fig_rows),
                "status_counts": status_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
