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
    CLOUD_REWARD_WEIGHTS,
    CLOUD_STICKINESS_BONUS,
    CLOUD_SWITCH_PENALTY,
    EDGE_REWARD_WEIGHTS,
    MAX_EPOCHS,
    MACRO_FREQUENCY,
    MACRO_TX_POWER,
)
from SC_RL_main import rl
from Topology import define_topology, define_users


REPO_DIR = Path(__file__).resolve().parent
LOG_ROOT = REPO_DIR / "logs" / "revision_round1" / "group3_channel_shadowing"
PLOT_ROOT = REPO_DIR / "plots" / "revision_round1" / "group3_channel_shadowing"
SIGMA_VALUES = [0.0, 4.0, 8.0]
SMOKE_SIGMA_VALUES = [4.0, 8.0]
SEED = 42
K_VALUE = 3
METHOD = "hmadqn_shadowing_robustness"
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


def sigma_label(sigma_db):
    if float(sigma_db).is_integer():
        return str(int(sigma_db))
    return str(sigma_db).replace(".", "p")


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


def shadowing_determinism_check():
    _, macro_bs_list, _, _ = define_topology()
    user = define_users()[0]
    bs = macro_bs_list[0]
    original = user.calculate_rsrp("curr", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    user.configure_shadowing(0.0, SEED)
    sigma_zero = user.calculate_rsrp("curr", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    if not np.isclose(original, sigma_zero):
        raise ValueError("sigma=0 shadowing does not match original RSRP")

    user.configure_shadowing(4.0, SEED)
    first = user.calculate_rsrp("curr", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    second = user.calculate_rsrp("curr", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    if not np.isclose(first, second):
        raise ValueError("sigma=4 shadowing is not deterministic for a fixed link")

    user.configure_shadowing(8.0, SEED)
    first = user.calculate_rsrp("next", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    second = user.calculate_rsrp("next", bs, MACRO_FREQUENCY, MACRO_TX_POWER)
    if not np.isclose(first, second):
        raise ValueError("sigma=8 shadowing is not deterministic for a fixed link")


def run_hmadqn_shadowing(seed, sigma_db, max_epochs, device):
    setup_seed(seed)
    sat_bs_list, macro_bs_list, small_bs_list, uav_bs_list = define_topology()
    user_list = define_users()
    config = {
        "max_epochs": max_epochs,
        "cloud_association_period": K_VALUE,
        "epsilon_schedule": "exp2",
        "convergence_check": False,
        "shadowing_sigma_db": sigma_db,
        "shadowing_seed": seed,
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


def read_original_sigma0_metrics():
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
    config_path = output_root / "method_configs" / f"{METHOD}.json"
    write_json(
        config_path,
        {
            "method": METHOD,
            "seed": SEED,
            "k": K_VALUE,
            "sigma_values_db": SIGMA_VALUES,
            "reused_sigma0_original_logs": True,
            "new_shadowing_sigma_values_db": [4.0, 8.0],
            "max_epochs": max_epochs,
            "epsilon_schedule": "exp2",
            "convergence_check": False,
            "default_cloud_association_period": CLOUD_ASSOCIATION_PERIOD,
            "cloud_switch_penalty": CLOUD_SWITCH_PENALTY,
            "cloud_stickiness_bonus": CLOUD_STICKINESS_BONUS,
            "cloud_reward_weights": CLOUD_REWARD_WEIGHTS,
            "edge_reward_weights": EDGE_REWARD_WEIGHTS,
            "execution_device": device_name,
            "shadowing_model": "rsrp_shadowed_dbm = rsrp_dbm + Normal(0, sigma_db)",
            "shadowing_sampling": "deterministic link-position sample from SHA256(seed, sigma, user, node, curr_next, positions)",
            "claim_boundary": "large-scale shadowing stress test only; no inter-cell interference, small-scale fading, or full SINR model",
        },
    )
    return config_path


def write_csv_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_epoch_metrics(output_root, run_id, sigma_db, metrics):
    run_path = output_root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "epoch_metrics.csv"
    row_count = len(metrics["reward_sum"])
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "sigma_db"] + METRIC_HEADER)
        for epoch in range(row_count):
            row = [epoch, sigma_db]
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
    rows, run_id, sigma_db, seed, source, config_path, metrics_path, status, notes
):
    rows.append(
        {
            "run_id": run_id,
            "method": METHOD,
            "sigma_db": sigma_db,
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
        "sigma_db",
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
    sigma_db,
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
    if not metrics["reward_sum"]:
        return False
    metric_records.append(
        {
            "run_id": run_id,
            "sigma_db": sigma_db,
            "seed": seed,
            "source": source,
            "metrics": metrics,
        }
    )
    add_manifest_row(
        manifest_rows,
        run_id,
        sigma_db,
        seed,
        source,
        config_path,
        metrics_path,
        "success",
        "reused existing group3 output",
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
        "sigma_db": record["sigma_db"],
        "seed": record["seed"],
        "tail_window": tail_window,
    }
    for key in REQUIRED_KPIS:
        values = [float(v) for v in tail_slice(record["metrics"][key], tail_window)]
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return out


def write_summaries(output_root, metric_records, sigma_values):
    seed_rows = []
    for record in sorted(metric_records, key=lambda r: (r["sigma_db"], r["seed"])):
        for tail_window in TAIL_WINDOWS:
            seed_rows.append(summarize_record(record, tail_window))

    by_seed_header = [
        "method",
        "sigma_db",
        "seed",
        "tail_window",
        "reward_sum_mean",
        "throughput_mbps_mean",
        "qos_satisfaction_mean",
        "jain_fairness_mean",
        "switch_ratio_mean",
    ]
    write_csv_rows(output_root / "summary_by_sigma_seed.csv", by_seed_header, seed_rows)

    summary_rows = []
    for sigma_db in sigma_values:
        for tail_window in TAIL_WINDOWS:
            group = [
                row
                for row in seed_rows
                if float(row["sigma_db"]) == float(sigma_db)
                and row["tail_window"] == tail_window
            ]
            if not group:
                continue
            out = {
                "method": METHOD,
                "sigma_db": sigma_db,
                "n_seed": len(group),
                "tail_window": tail_window,
            }
            for key in REQUIRED_KPIS:
                values = [row[f"{key}_mean"] for row in group]
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            summary_rows.append(out)

    summary_header = [
        "method",
        "sigma_db",
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
    write_csv_rows(output_root / "summary_by_sigma.csv", summary_header, summary_rows)
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


def write_fig12(output_root, metric_records, seed_rows):
    source_by_sigma = {
        float(record["sigma_db"]): record["run_id"] for record in metric_records
    }
    rows = []
    for seed_row in seed_rows:
        sigma_db = float(seed_row["sigma_db"])
        for metric in REQUIRED_KPIS:
            rows.append(
                {
                    "sigma_db": seed_row["sigma_db"],
                    "seed": seed_row["seed"],
                    "tail_window": seed_row["tail_window"],
                    "metric": metric,
                    "value": seed_row[f"{metric}_mean"],
                    "unit": unit_for_metric(metric),
                    "plot_panel": panel_for_metric(metric),
                    "source_run": source_by_sigma.get(sigma_db, ""),
                    "notes": (
                        "sigma=0 uses original logs"
                        if sigma_db == 0.0
                        else "large-scale shadowing stress test"
                    ),
                }
            )
    header = [
        "sigma_db",
        "seed",
        "tail_window",
        "metric",
        "value",
        "unit",
        "plot_panel",
        "source_run",
        "notes",
    ]
    write_csv_rows(output_root / "fig12_shadowing_robustness.csv", header, rows)
    return rows


def plot_preview(plot_root, summary_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    last = [row for row in summary_rows if row["tail_window"] == "last_10pct"]
    if not last:
        return
    x_values = [float(row["sigma_db"]) for row in last]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].plot(x_values, [row["throughput_mbps_mean"] for row in last], marker="o")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("Mbps")
    for metric, label in [
        ("qos_satisfaction_mean", "QoS"),
        ("jain_fairness_mean", "Jain"),
    ]:
        axes[1].plot(x_values, [row[metric] for row in last], marker="o", label=label)
    axes[1].set_title("QoS/Fairness")
    axes[1].legend()
    axes[2].plot(x_values, [row["switch_ratio_mean"] for row in last], marker="o")
    axes[2].set_title("Switch Ratio")
    axes[3].plot(x_values, [row["reward_sum_mean"] for row in last], marker="o")
    axes[3].set_title("Reward")
    for ax in axes:
        ax.set_xlabel("Shadowing sigma (dB)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "fig12_preview_shadowing_robustness.png", dpi=300)
    plt.close(fig)


def plot_training_curves(plot_root, metric_records):
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(REQUIRED_KPIS), 1, figsize=(9, 12), sharex=False)
    for idx, metric in enumerate(REQUIRED_KPIS):
        ax = axes[idx]
        for record in sorted(metric_records, key=lambda r: r["sigma_db"]):
            y = record["metrics"][metric]
            ax.plot(
                range(len(y)),
                y,
                linewidth=0.8,
                alpha=0.8,
                label=f"sigma={record['sigma_db']} dB",
            )
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_root / "shadowing_training_curves.png", dpi=300)
    plt.close(fig)


def plot_kpi_summary(plot_root, summary_rows):
    plot_root.mkdir(parents=True, exist_ok=True)
    last = [row for row in summary_rows if row["tail_window"] == "last_10pct"]
    if not last:
        return
    metrics = [
        ("throughput_mbps_mean", "Throughput"),
        ("qos_satisfaction_mean", "QoS"),
        ("jain_fairness_mean", "Jain"),
        ("switch_ratio_mean", "Switch"),
        ("reward_sum_mean", "Reward"),
    ]
    x = np.arange(len(last))
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
    labels = [str(row["sigma_db"]) for row in last]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(x, [row[metric] for row in last])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("sigma dB")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "shadowing_kpi_summary.png", dpi=300)
    plt.close(fig)


def validate_outputs(output_root, expected_sigmas, require_10000):
    manifest_path = output_root / "run_manifest.csv"
    fig_path = output_root / "fig12_shadowing_robustness.csv"
    summary_path = output_root / "summary_by_sigma.csv"
    for path in [manifest_path, fig_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    present = {
        float(row["sigma_db"]) for row in manifest if row["status"] == "success"
    }
    missing = {float(sigma) for sigma in expected_sigmas} - present
    if missing:
        raise ValueError(f"manifest missing successful sigma values: {sorted(missing)}")

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
        rows = list(csv.DictReader(f))
    for sigma_db in expected_sigmas:
        metrics = {
            row["metric"] for row in rows if float(row["sigma_db"]) == float(sigma_db)
        }
        missing_metrics = set(REQUIRED_KPIS) - metrics
        if missing_metrics:
            raise ValueError(
                f"fig12 missing metrics for sigma={sigma_db}: {sorted(missing_metrics)}"
            )
        for tail_window in TAIL_WINDOWS:
            kpi_rows = [
                row
                for row in rows
                if float(row["sigma_db"]) == float(sigma_db)
                and row["tail_window"] == tail_window
            ]
            if len(kpi_rows) != len(REQUIRED_KPIS):
                raise ValueError(
                    f"fig12 has {len(kpi_rows)} KPI rows for sigma={sigma_db}, {tail_window}"
                )

    summary_rows = list(
        csv.DictReader(summary_path.open("r", newline="", encoding="utf-8"))
    )
    for sigma_db in expected_sigmas:
        for tail_window in TAIL_WINDOWS:
            matches = [
                row
                for row in summary_rows
                if float(row["sigma_db"]) == float(sigma_db)
                and row["tail_window"] == tail_window
            ]
            if not matches:
                raise ValueError(
                    f"summary missing sigma={sigma_db}, tail_window={tail_window}"
                )
            if int(matches[0]["n_seed"]) != 1:
                raise ValueError("expected n_seed=1 for Group 3 summary")


def run_all(
    output_root,
    plot_root,
    sigma_values,
    max_epochs,
    seed,
    device_name,
    resume,
    skip_plots,
    require_10000,
):
    ensure_dirs(output_root)
    shadowing_determinism_check()
    config_path = write_method_config(output_root, max_epochs, device_name)
    manifest_rows = []
    metric_records = []
    device = device_from_name(device_name)

    for sigma_db in sigma_values:
        run_id = f"hmadqn_shadowing_sigma{sigma_label(sigma_db)}_seed{seed}"
        try:
            source = "original_logs" if float(sigma_db) == 0.0 else "shadowing_run"
            if resume and maybe_reuse_existing_run(
                output_root,
                run_id,
                sigma_db,
                seed,
                source,
                config_path,
                manifest_rows,
                metric_records,
            ):
                write_manifest(output_root, manifest_rows)
                continue
            if float(sigma_db) == 0.0:
                try:
                    metrics = read_original_sigma0_metrics()
                    notes = "reused original H-MADQN K=3 seed42 logs"
                except Exception:
                    source = "reproduced_sigma0"
                    metrics = run_hmadqn_shadowing(seed, sigma_db, max_epochs, device)
                    notes = "original logs unavailable; reproduced sigma=0 under group3"
            else:
                metrics = run_hmadqn_shadowing(seed, sigma_db, max_epochs, device)
                notes = f"completed {len(metrics['reward_sum'])} epochs"
            metrics_path = write_epoch_metrics(output_root, run_id, sigma_db, metrics)
            metric_records.append(
                {
                    "run_id": run_id,
                    "sigma_db": sigma_db,
                    "seed": seed,
                    "source": source,
                    "metrics": metrics,
                }
            )
            add_manifest_row(
                manifest_rows,
                run_id,
                sigma_db,
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
                sigma_db,
                seed,
                "original_logs" if float(sigma_db) == 0.0 else "shadowing_run",
                config_path,
                "",
                "failed",
                str(exc),
            )
        write_manifest(output_root, manifest_rows)

    seed_rows, summary_rows = write_summaries(output_root, metric_records, sigma_values)
    fig_rows = write_fig12(output_root, metric_records, seed_rows)
    validate_outputs(output_root, sigma_values, require_10000)
    if not skip_plots:
        plot_preview(plot_root, summary_rows)
        plot_training_curves(plot_root, metric_records)
        plot_kpi_summary(plot_root, summary_rows)
    return manifest_rows, metric_records, fig_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run revision round 1 Group 3 shadowing robustness outputs."
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
    sigma_values = SMOKE_SIGMA_VALUES if smoke else SIGMA_VALUES
    max_epochs = args.smoke_epochs if smoke else args.max_epochs
    manifest_rows, metric_records, _ = run_all(
        output_root,
        plot_root,
        sigma_values,
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
