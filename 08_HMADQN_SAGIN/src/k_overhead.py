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

from Parameters import (
    CLOUD_ASSOCIATION_PERIOD,
    CLOUD_INPUT_DIM,
    CLOUD_REWARD_WEIGHTS,
    CLOUD_STICKINESS_BONUS,
    CLOUD_SWITCH_PENALTY,
    EDGE_REWARD_WEIGHTS,
    MAX_EPOCHS,
    NUM_OF_MACRO,
    NUM_OF_SAT,
    NUM_OF_SMALL,
    NUM_OF_UAV,
    NUM_OF_USER,
)
from SC_RL_main import rl
from Topology import define_topology, define_users


REPO_DIR = Path(__file__).resolve().parent
LOG_ROOT = REPO_DIR / "logs" / "revision_round1" / "group2_k_overhead"
PLOT_ROOT = REPO_DIR / "plots" / "revision_round1" / "group2_k_overhead"
K_VALUES = [1, 2, 3, 5, 7]
SEED = 42
METHOD = "hmadqn_k_sweep"
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


def rel(path):
    return str(Path(path).resolve().relative_to(REPO_DIR)) if path else ""


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


def run_hmadqn_k(seed, k_value, max_epochs, device):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()
    config = {
        "max_epochs": max_epochs,
        "cloud_association_period": k_value,
        "epsilon_schedule": "exp2",
        "convergence_check": True,
        "convergence_window": 1000,
        "convergence_std_ratio": 0.05,
        "min_convergence_epochs": 2000,
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


def read_original_k3_metrics():
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
            for key in LOSS_KEYS:
                metrics[key].append(None)
    if not metrics["reward_sum"]:
        raise ValueError("no baseline rows found in original KPI log")
    return metrics


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_method_config(output_root, max_epochs, device_name):
    config_path = output_root / "method_configs" / "hmadqn_k_sweep.json"
    feature_dim = compute_cloud_feature_dim()
    write_json(
        config_path,
        {
            "method": METHOD,
            "seed": SEED,
            "k_values": K_VALUES,
            "reused_k3_original_logs": True,
            "reproduced_k_values": [1, 2, 5, 7],
            "max_epochs": max_epochs,
            "epsilon_schedule": "exp2",
            "convergence_check": True,
            "convergence_window": 1000,
            "convergence_std_ratio": 0.05,
            "min_convergence_epochs": 2000,
            "default_cloud_association_period": CLOUD_ASSOCIATION_PERIOD,
            "cloud_switch_penalty": CLOUD_SWITCH_PENALTY,
            "cloud_stickiness_bonus": CLOUD_STICKINESS_BONUS,
            "cloud_reward_weights": CLOUD_REWARD_WEIGHTS,
            "edge_reward_weights": EDGE_REWARD_WEIGHTS,
            "execution_device": device_name,
            "cloud_feature_dim_formula": "(NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV) + (NUM_OF_SAT + NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV) + 3",
            "cloud_feature_dim": feature_dim,
            "cloud_input_dim_constant": CLOUD_INPUT_DIM,
            "slot_duration_seconds": None,
        },
    )
    return config_path


def write_epoch_metrics(output_root, run_id, k_value, metrics):
    run_path = output_root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "epoch_metrics.csv"
    row_count = len(metrics["reward_sum"])
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "K"] + METRIC_HEADER)
        for epoch in range(row_count):
            row = [epoch, k_value]
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


def add_manifest_row(
    rows, run_id, k_value, seed, source, config_path, metrics_path, status, notes
):
    rows.append(
        {
            "run_id": run_id,
            "method": METHOD,
            "K": k_value,
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
        "K",
        "seed",
        "source",
        "config_path",
        "metrics_path",
        "status",
        "notes",
    ]
    write_csv_rows(output_root / "run_manifest.csv", header, rows)


def write_csv_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_reuse_existing_run(
    output_root, run_id, k_value, seed, source, config_path, manifest_rows, metric_records
):
    metrics_path = output_root / "runs" / run_id / "epoch_metrics.csv"
    if not metrics_path.exists():
        return False
    metrics = load_epoch_metrics(metrics_path)
    if not metrics["reward_sum"]:
        return False
    metric_records.append(
        {
            "run_id": run_id,
            "K": k_value,
            "seed": seed,
            "source": source,
            "metrics": metrics,
        }
    )
    add_manifest_row(
        manifest_rows,
        run_id,
        k_value,
        seed,
        source,
        config_path,
        metrics_path,
        "success",
        "reused existing group2 output",
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
    out = {
        "method": METHOD,
        "K": record["K"],
        "seed": record["seed"],
        "tail_window": tail_window,
    }
    for key in REQUIRED_KPIS:
        values = [float(v) for v in tail_slice(record["metrics"][key], tail_window)]
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return out


def write_summaries(output_root, metric_records):
    seed_rows = []
    for record in sorted(metric_records, key=lambda r: (r["K"], r["seed"])):
        for tail_window in TAIL_WINDOWS:
            seed_rows.append(summarize_record(record, tail_window))

    by_seed_header = [
        "method",
        "K",
        "seed",
        "tail_window",
        "reward_sum_mean",
        "throughput_mbps_mean",
        "qos_satisfaction_mean",
        "jain_fairness_mean",
        "switch_ratio_mean",
    ]
    write_csv_rows(output_root / "summary_by_k_seed.csv", by_seed_header, seed_rows)

    summary_rows = []
    for k_value in K_VALUES:
        for tail_window in TAIL_WINDOWS:
            group = [
                row
                for row in seed_rows
                if int(row["K"]) == k_value and row["tail_window"] == tail_window
            ]
            if not group:
                continue
            out = {"method": METHOD, "K": k_value, "n_seed": len(group), "tail_window": tail_window}
            for key in REQUIRED_KPIS:
                values = [row[f"{key}_mean"] for row in group]
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(out)

    summary_header = [
        "method",
        "K",
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
    write_csv_rows(output_root / "summary_by_k.csv", summary_header, summary_rows)
    return seed_rows, summary_rows


def compute_cloud_feature_dim():
    non_sat_rsrp = NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV
    all_nodes = NUM_OF_SAT + NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV
    return non_sat_rsrp + all_nodes + 3


def overhead_for_k(k_value):
    cloud_feature_dim = compute_cloud_feature_dim()
    uplink = NUM_OF_USER * cloud_feature_dim * 4
    downlink = NUM_OF_USER * 4
    total = uplink + downlink
    return {
        "K": k_value,
        "num_ue": NUM_OF_USER,
        "num_sat": NUM_OF_SAT,
        "num_macro": NUM_OF_MACRO,
        "num_small": NUM_OF_SMALL,
        "num_uav": NUM_OF_UAV,
        "cloud_feature_dim": cloud_feature_dim,
        "uplink_bytes_per_cloud_epoch": uplink,
        "downlink_bytes_per_cloud_epoch": downlink,
        "total_bytes_per_cloud_epoch": total,
        "total_bytes_per_slot": total / k_value,
        "slot_duration_seconds": "",
        "total_bytes_per_second": "",
        "notes": "not reported without explicit slot-duration assumption",
    }


def write_overhead(output_root):
    rows = [overhead_for_k(k_value) for k_value in K_VALUES]
    header = [
        "K",
        "num_ue",
        "num_sat",
        "num_macro",
        "num_small",
        "num_uav",
        "cloud_feature_dim",
        "uplink_bytes_per_cloud_epoch",
        "downlink_bytes_per_cloud_epoch",
        "total_bytes_per_cloud_epoch",
        "total_bytes_per_slot",
        "slot_duration_seconds",
        "total_bytes_per_second",
        "notes",
    ]
    write_csv_rows(output_root / "signaling_overhead.csv", header, rows)
    return rows


def unit_for_metric(metric):
    if metric.endswith("mbps"):
        return "Mbps"
    if metric.endswith("bytes_per_cloud_epoch") or metric.endswith("bytes_per_slot"):
        return "bytes"
    return "ratio" if metric in ["qos_satisfaction", "jain_fairness", "switch_ratio"] else "unitless"


def panel_for_metric(metric):
    if metric == "throughput_mbps":
        return "A_throughput"
    if metric in ["qos_satisfaction", "jain_fairness", "switch_ratio"]:
        return "B_ratios"
    if "bytes" in metric:
        return "C_overhead"
    return "supp_reward"


def write_fig11(output_root, metric_records, seed_rows, overhead_rows):
    source_by_k = {record["K"]: record["run_id"] for record in metric_records}
    rows = []
    for seed_row in seed_rows:
        for metric in REQUIRED_KPIS:
            rows.append(
                {
                    "K": seed_row["K"],
                    "seed": seed_row["seed"],
                    "tail_window": seed_row["tail_window"],
                    "metric": metric,
                    "value": seed_row[f"{metric}_mean"],
                    "unit": unit_for_metric(metric),
                    "plot_panel": panel_for_metric(metric),
                    "source_run": source_by_k.get(int(seed_row["K"]), ""),
                    "notes": "K=3 uses original logs" if int(seed_row["K"]) == 3 else "K sweep run",
                }
            )
    overhead_metrics = [
        "uplink_bytes_per_cloud_epoch",
        "downlink_bytes_per_cloud_epoch",
        "total_bytes_per_cloud_epoch",
        "total_bytes_per_slot",
    ]
    for overhead in overhead_rows:
        for metric in overhead_metrics:
            rows.append(
                {
                    "K": overhead["K"],
                    "seed": "",
                    "tail_window": "not_applicable",
                    "metric": metric,
                    "value": overhead[metric],
                    "unit": "bytes",
                    "plot_panel": "C_overhead",
                    "source_run": "signaling_overhead",
                    "notes": overhead["notes"],
                }
            )
    header = [
        "K",
        "seed",
        "tail_window",
        "metric",
        "value",
        "unit",
        "plot_panel",
        "source_run",
        "notes",
    ]
    write_csv_rows(output_root / "fig11_k_sensitivity.csv", header, rows)
    return rows


def plot_preview(plot_root, summary_rows, overhead_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    last = [row for row in summary_rows if row["tail_window"] == "last_10pct"]
    if not last:
        return
    k_values = [int(row["K"]) for row in last]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(k_values, [row["throughput_mbps_mean"] for row in last], marker="o")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("Mbps")
    for metric, label in [
        ("qos_satisfaction_mean", "QoS"),
        ("jain_fairness_mean", "Jain"),
        ("switch_ratio_mean", "Switch"),
    ]:
        axes[1].plot(k_values, [row[metric] for row in last], marker="o", label=label)
    axes[1].set_title("Ratios")
    axes[1].legend()
    axes[2].bar(
        [row["K"] for row in overhead_rows],
        [row["total_bytes_per_slot"] for row in overhead_rows],
    )
    axes[2].set_title("Overhead Per Slot")
    axes[2].set_ylabel("bytes/slot")
    axes[3].plot(k_values, [row["reward_sum_mean"] for row in last], marker="o")
    axes[3].set_title("Reward")
    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "fig11_preview_k_sensitivity.png", dpi=300)
    plt.close(fig)


def plot_training_curves(plot_root, metric_records):
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(REQUIRED_KPIS), 1, figsize=(9, 12), sharex=False)
    for idx, metric in enumerate(REQUIRED_KPIS):
        ax = axes[idx]
        for record in sorted(metric_records, key=lambda r: r["K"]):
            y = record["metrics"][metric]
            ax.plot(range(len(y)), y, linewidth=0.8, alpha=0.8, label=f"K={record['K']}")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_root / "k_training_curves.png", dpi=300)
    plt.close(fig)


def plot_overhead(plot_root, overhead_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [row["K"] for row in overhead_rows],
        [row["total_bytes_per_slot"] for row in overhead_rows],
        marker="o",
    )
    ax.set_title("Cloud Signaling Overhead")
    ax.set_xlabel("K")
    ax.set_ylabel("bytes/slot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "overhead_by_k.png", dpi=300)
    plt.close(fig)


def validate_outputs(output_root, expected_k_values):
    manifest_path = output_root / "run_manifest.csv"
    fig_path = output_root / "fig11_k_sensitivity.csv"
    overhead_path = output_root / "signaling_overhead.csv"
    for path in [manifest_path, fig_path, overhead_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    present_k = {int(row["K"]) for row in manifest if row["status"] == "success"}
    missing_k = set(expected_k_values) - present_k
    if missing_k:
        raise ValueError(f"manifest missing K values: {sorted(missing_k)}")

    with fig_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = set(REQUIRED_KPIS) | {
        "uplink_bytes_per_cloud_epoch",
        "downlink_bytes_per_cloud_epoch",
        "total_bytes_per_cloud_epoch",
        "total_bytes_per_slot",
    }
    for k_value in expected_k_values:
        metrics_for_k = {row["metric"] for row in rows if int(row["K"]) == k_value}
        missing = required - metrics_for_k
        if missing:
            raise ValueError(f"fig11 missing metrics for K={k_value}: {sorted(missing)}")

    overhead_rows = list(csv.DictReader(overhead_path.open("r", newline="", encoding="utf-8")))
    if int(overhead_rows[0]["cloud_feature_dim"]) != 20:
        raise ValueError("default cloud_feature_dim validation failed")
    for row in overhead_rows:
        k_value = int(row["K"])
        total = float(row["total_bytes_per_cloud_epoch"])
        per_slot = float(row["total_bytes_per_slot"])
        if total != 8400.0:
            raise ValueError(f"unexpected total bytes/cloud-epoch for K={k_value}: {total}")
        if not np.isclose(per_slot, 8400.0 / k_value):
            raise ValueError(f"unexpected bytes/slot for K={k_value}: {per_slot}")


def run_all(output_root, plot_root, k_values, max_epochs, seed, device_name, resume, skip_plots):
    ensure_dirs(output_root)
    config_path = write_method_config(output_root, max_epochs, device_name)
    manifest_rows = []
    metric_records = []
    device = device_from_name(device_name)

    for k_value in k_values:
        run_id = f"hmadqn_k_sweep_k{k_value}_seed{seed}"
        if k_value == 3:
            run_id = f"hmadqn_k3_original_seed{seed}"
        try:
            source = "original_logs" if k_value == 3 else "k_sweep_run"
            if resume and maybe_reuse_existing_run(
                output_root,
                run_id,
                k_value,
                seed,
                source,
                config_path,
                manifest_rows,
                metric_records,
            ):
                write_manifest(output_root, manifest_rows)
                continue
            if k_value == 3:
                metrics = read_original_k3_metrics()
                notes = "reused original H-MADQN K=3 logs"
            else:
                metrics = run_hmadqn_k(seed, k_value, max_epochs, device)
                notes = f"completed {len(metrics['reward_sum'])} epochs"
            metrics_path = write_epoch_metrics(output_root, run_id, k_value, metrics)
            metric_records.append(
                {
                    "run_id": run_id,
                    "K": k_value,
                    "seed": seed,
                    "source": source,
                    "metrics": metrics,
                }
            )
            add_manifest_row(
                manifest_rows,
                run_id,
                k_value,
                seed,
                source,
                config_path,
                metrics_path,
                "success",
                notes,
            )
        except Exception as exc:
            add_manifest_row(
                manifest_rows,
                run_id,
                k_value,
                seed,
                "original_logs" if k_value == 3 else "k_sweep_run",
                config_path,
                "",
                "failed",
                str(exc),
            )
        write_manifest(output_root, manifest_rows)

    _, summary_rows = write_summaries(output_root, metric_records)
    overhead_rows = write_overhead(output_root)
    seed_rows = []
    for record in metric_records:
        for tail_window in TAIL_WINDOWS:
            seed_rows.append(summarize_record(record, tail_window))
    fig_rows = write_fig11(output_root, metric_records, seed_rows, overhead_rows)
    validate_outputs(output_root, k_values)
    if not skip_plots:
        plot_preview(plot_root, summary_rows, overhead_rows)
        plot_training_curves(plot_root, metric_records)
        plot_overhead(plot_root, overhead_rows)
    return manifest_rows, metric_records, fig_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run revision round 1 Group 2 K sensitivity and overhead outputs."
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
    k_values = [1, 7] if smoke else K_VALUES
    max_epochs = args.smoke_epochs if smoke else args.max_epochs
    manifest_rows, metric_records, _ = run_all(
        output_root,
        plot_root,
        k_values,
        max_epochs,
        args.seed,
        args.device,
        args.resume,
        args.skip_plots,
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
