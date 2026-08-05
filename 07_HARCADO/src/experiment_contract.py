from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(os.environ.get("CGDQN_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC_ROOT = REPO_ROOT / "src"
NS3_ROOT = Path(os.environ.get("CGDQN_NS3_ROOT", REPO_ROOT / "NS3")).resolve()

TRAJECTORY_DIR = Path(
    os.environ.get("CGDQN_TRAJECTORY_DIR", NS3_ROOT / "20250424_trajectory_all")
).resolve()
INTERPOLATED_TRAJECTORY_DIR = Path(
    os.environ.get("CGDQN_INTERPOLATED_TRAJECTORY_DIR", NS3_ROOT / "20250511_trajectory_all_interpolation")
).resolve()
CATBOOST_DATA_DIR = Path(
    os.environ.get("CGDQN_CATBOOST_DATA_DIR", NS3_ROOT / "derived" / "catboost_events")
).resolve()

MODEL_PATH = Path(os.environ.get("CGDQN_CATBOOST_MODEL", SRC_ROOT / "catboost_model.cbm")).resolve()
LOG_ROOT = Path(os.environ.get("CGDQN_LOG_ROOT", REPO_ROOT / "logs")).resolve()
FIGURE_DATA_DIR = Path(os.environ.get("CGDQN_FIGURE_DATA_DIR", REPO_ROOT / "figs")).resolve()
PLOT_OUTPUT_DIR = Path(os.environ.get("CGDQN_PLOT_OUTPUT_DIR", REPO_ROOT / "plots")).resolve()

SPEED_LEVELS = (1.0, 3.0, 6.0)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_dir(prefer_interpolated: bool = False) -> Path:
    if prefer_interpolated and INTERPOLATED_TRAJECTORY_DIR.exists():
        return INTERPOLATED_TRAJECTORY_DIR
    if TRAJECTORY_DIR.exists():
        return TRAJECTORY_DIR
    if INTERPOLATED_TRAJECTORY_DIR.exists():
        return INTERPOLATED_TRAJECTORY_DIR
    return TRAJECTORY_DIR


def trajectory_file(index: int, prefer_interpolated: bool = False) -> Path:
    base = trajectory_dir(prefer_interpolated=prefer_interpolated)
    candidates = [
        base / f"yzc_v8_ue_{index}_interpolation.csv",
        base / f"yzc_v8_ue_{index}.csv",
        base / f"yzc_v7_ue_{index}_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if prefer_interpolated else candidates[1]


def available_trajectory_indices(base: Path | None = None) -> list[int]:
    base = base or trajectory_dir()
    indices: list[int] = []
    for path in base.glob("yzc_v8_ue_*.csv"):
        stem = path.stem.replace("_interpolation", "")
        try:
            indices.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(set(indices))


def nearest_speed(speed: float) -> float:
    return min(SPEED_LEVELS, key=lambda level: abs(level - speed))


def infer_speed_from_csv(path: Path, sample_rows: int = 25) -> float | None:
    speeds: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "VelX" not in reader.fieldnames or "VelY" not in reader.fieldnames:
            return None
        for row_idx, row in enumerate(reader):
            if row_idx >= sample_rows:
                break
            try:
                vx = float(row["VelX"])
                vy = float(row["VelY"])
            except (TypeError, ValueError):
                continue
            speeds.append((vx * vx + vy * vy) ** 0.5)
    if not speeds:
        return None
    return nearest_speed(sum(speeds) / len(speeds))


def timestamped_run_dir(prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(LOG_ROOT / f"{timestamp}_{prefix}")


def write_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def contract_snapshot() -> dict:
    tracked_paths = {
        "repo_root": str(REPO_ROOT),
        "src_root": str(SRC_ROOT),
        "ns3_root": str(NS3_ROOT),
        "trajectory_dir": str(trajectory_dir()),
        "interpolated_trajectory_dir": str(INTERPOLATED_TRAJECTORY_DIR),
        "catboost_data_dir": str(CATBOOST_DATA_DIR),
        "catboost_model": str(MODEL_PATH),
        "log_root": str(LOG_ROOT),
        "figure_data_dir": str(FIGURE_DATA_DIR),
        "plot_output_dir": str(PLOT_OUTPUT_DIR),
    }
    existing_hashes = {
        name: file_sha256(Path(value))
        for name, value in tracked_paths.items()
        if Path(value).is_file()
    }
    return {
        "paths": tracked_paths,
        "hashes": existing_hashes,
        "speed_levels_mps": list(SPEED_LEVELS),
        "runtime_env_contract": {
            "variant": os.environ.get("CGDQN_VARIANT", ""),
            "random_seed": os.environ.get("CGDQN_RANDOM_SEED", ""),
            "wcfh_dwell_ticks": os.environ.get("CGDQN_WCFH_DWELL_TICKS", "3"),
            "wcfh_safe_margin_db": os.environ.get("CGDQN_WCFH_SAFE_MARGIN_DB", "0.5"),
            "wcfh_target_rsrp_min": os.environ.get("CGDQN_WCFH_TARGET_RSRP_MIN", "-96"),
            "rlf_sinr_threshold_db": os.environ.get("CGDQN_RLF_SINR_THRESHOLD_DB", "-5.0"),
            "enable_radio_link_trace": os.environ.get("CGDQN_ENABLE_RADIO_LINK_TRACE", "0"),
            "radio_link_trace_final_epoch_only": os.environ.get(
                "CGDQN_RADIO_LINK_TRACE_FINAL_EPOCH_ONLY", "1"
            ),
            "rl_alpha": os.environ.get("CGDQN_RL_ALPHA", "0.001"),
            "rl_gamma": os.environ.get("CGDQN_RL_GAMMA", "0.9"),
            "rl_epsilon": os.environ.get("CGDQN_RL_EPSILON", "0.1"),
            "outcome_reward_profile": os.environ.get("CGDQN_OUTCOME_REWARD_PROFILE", "outcome_v1"),
            "disable_loss_early_stop": os.environ.get("CGDQN_DISABLE_LOSS_EARLY_STOP", "0"),
            "ddqn_target_update_steps": os.environ.get("CGDQN_DDQN_TARGET_UPDATE_STEPS", "100"),
            "optimizer": "Adam",
            "value_td_loss": os.environ.get("CGDQN_VALUE_TD_LOSS", "smooth_l1"),
            "smooth_l1_beta": os.environ.get("CGDQN_SMOOTH_L1_BETA", "1.0"),
            "target_update_mode": os.environ.get("CGDQN_TARGET_UPDATE_MODE", "hard"),
            "replay_enabled": os.environ.get("CGDQN_REPLAY_ENABLED", "0"),
            "replay_capacity": os.environ.get("CGDQN_REPLAY_CAPACITY", "20000"),
            "replay_batch_size": os.environ.get("CGDQN_REPLAY_BATCH_SIZE", "64"),
            "replay_warmup_transitions": os.environ.get(
                "CGDQN_REPLAY_WARMUP_TRANSITIONS", "512"
            ),
            "replay_warmup_epochs": os.environ.get("CGDQN_REPLAY_WARMUP_EPOCHS", "3"),
            "replay_updates_per_step": os.environ.get("CGDQN_REPLAY_UPDATES_PER_STEP", "1"),
            "replay_sample_policy": os.environ.get("CGDQN_REPLAY_SAMPLE_POLICY", "uniform"),
            "ppo_clip_eps": os.environ.get("CGDQN_PPO_CLIP_EPS", "0.2"),
            "ppo_value_coef": os.environ.get("CGDQN_PPO_VALUE_COEF", "0.5"),
            "ppo_entropy_coef": os.environ.get("CGDQN_PPO_ENTROPY_COEF", "0.01"),
            "action_prior_beta": os.environ.get("CGDQN_ACTION_PRIOR_BETA", "0.05"),
            "stress_config": os.environ.get("CGDQN_STRESS_CONFIG", "nominal"),
            "bs_tx_power_dbm": os.environ.get("CGDQN_BS_TX_POWER", "46"),
            "shadow_sigma_db": os.environ.get("CGDQN_SHADOW_SIGMA_DB", "1.4142135623730951"),
        },
    }
