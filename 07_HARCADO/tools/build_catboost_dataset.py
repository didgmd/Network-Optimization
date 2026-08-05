from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from catboost_feature_schema import (
    CATBOOST_FEATURE_COLUMNS,
    CATBOOST_LABEL_COLUMN,
    write_feature_schema,
)
from experiment_contract import CATBOOST_DATA_DIR, TRAJECTORY_DIR, ensure_dir, infer_speed_from_csv, nearest_speed
from Formular import path_loss_calculation
from stress_config import (
    DEFAULT_BS_TX_POWER_DBM,
    DEFAULT_SHADOW_SIGMA_DB,
    resolve_stress_config,
)


RSRP_TH = -90.0
BEFORE_HO = 4
AFTER_HO = 10
BS_FREQUENCY = 3.5
FEATURE_COLUMNS = CATBOOST_FEATURE_COLUMNS + ["speed_mps", "source_file", "ue_index", CATBOOST_LABEL_COLUMN]


def infer_ue_index(path: Path) -> int:
    match = re.search(r"ue_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer UE index from {path.name}")
    return int(match.group(1))


def label_event(frame: pd.DataFrame, idx: int, start: int) -> int:
    rsrp_now = float(frame.iloc[idx]["SourceRsrp"])
    rsrp_avg = float(frame.iloc[start:idx]["SourceRsrp"].mean())
    if rsrp_avg < RSRP_TH:
        return 1
    if rsrp_now < rsrp_avg and rsrp_avg >= RSRP_TH:
        return 0
    return 2


def stress_adjust_rsrp(
    original_rsrp: float,
    distance_m: float,
    *,
    bs_tx_power_dbm: float,
    shadow_sigma_db: float,
) -> float:
    distance = max(float(distance_m), 1.0)
    path_loss = path_loss_calculation(distance, BS_FREQUENCY)
    baseline_noise = DEFAULT_BS_TX_POWER_DBM - path_loss - float(original_rsrp)
    scaled_noise = baseline_noise * (shadow_sigma_db / DEFAULT_SHADOW_SIGMA_DB)
    return bs_tx_power_dbm - path_loss - scaled_noise


def apply_stress_to_frame(frame: pd.DataFrame, stress_config: dict[str, float | str]) -> pd.DataFrame:
    if stress_config["stress_config"] == "nominal":
        return frame
    adjusted = frame.copy()
    bs_tx_power_dbm = float(stress_config["bs_tx_power_dbm"])
    shadow_sigma_db = float(stress_config["shadow_sigma_db"])
    old_source = adjusted["SourceRsrp"].astype(float)
    old_target = adjusted["TargetRsrp"].astype(float)
    new_source = [
        stress_adjust_rsrp(value, distance, bs_tx_power_dbm=bs_tx_power_dbm, shadow_sigma_db=shadow_sigma_db)
        for value, distance in zip(old_source, adjusted["SourceDis"].astype(float))
    ]
    new_target = [
        stress_adjust_rsrp(value, distance, bs_tx_power_dbm=bs_tx_power_dbm, shadow_sigma_db=shadow_sigma_db)
        for value, distance in zip(old_target, adjusted["TargetDis"].astype(float))
    ]
    adjusted["SourceRsrp"] = new_source
    adjusted["TargetRsrp"] = new_target
    adjusted["SourceSinr"] = adjusted["SourceSinr"].astype(float) + (adjusted["SourceRsrp"] - old_source)
    adjusted["TargetSinr"] = adjusted["TargetSinr"].astype(float) + (adjusted["TargetRsrp"] - old_target)
    return adjusted


def event_row(
    frame: pd.DataFrame,
    idx: int,
    start: int,
    label: int,
    speed_mps: float | None,
    source_file: str,
    ue_index: int,
) -> dict:
    row = frame.iloc[idx]
    pre_mean_source_rsrp = float(frame.iloc[start:idx]["SourceRsrp"].mean())
    source_rsrp = float(row["SourceRsrp"])
    target_rsrp = float(row["TargetRsrp"])
    return {
        "SourceBsId": int(row["SourceBsId"]),
        "TargetBsId": int(row["TargetBsId"]),
        "SourceRsrp": source_rsrp,
        "TargetRsrp": target_rsrp,
        "SourceSinr": float(row["SourceSinr"]),
        "TargetSinr": float(row["TargetSinr"]),
        "TargetDis": float(row["TargetDis"]),
        "VelX": float(row["VelX"]),
        "VelY": float(row["VelY"]),
        "Direction": float(row["Direction"]),
        "TargetSourceRsrpGap": target_rsrp - source_rsrp,
        "PreMeanSourceRsrp": pre_mean_source_rsrp,
        "SourceRsrpDrop": pre_mean_source_rsrp - source_rsrp,
        "speed_mps": speed_mps if speed_mps is not None else nearest_speed((float(row["VelX"]) ** 2 + float(row["VelY"]) ** 2) ** 0.5),
        "source_file": source_file,
        "ue_index": ue_index,
        "label": label,
    }


def process_file(path: Path, stress_config: dict[str, float | str]) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"TTT", "SourceRsrp", "SourceBsId", "TargetBsId", "TargetRsrp", "SourceSinr", "TargetSinr", "TargetDis", "VelX", "VelY", "Direction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = apply_stress_to_frame(frame, stress_config)

    speed_mps = infer_speed_from_csv(path)
    ue_index = infer_ue_index(path)
    rows = []
    for idx in frame.index[frame["TTT"] == 0].tolist():
        start = idx - BEFORE_HO
        end = idx + AFTER_HO + 1
        if start < 0 or end > len(frame):
            continue
        rows.append(
            event_row(
                frame,
                idx,
                start,
                label_event(frame, idx, start),
                speed_mps,
                str(path),
                ue_index,
            )
        )
    return rows


def build_dataset(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    stress_config = resolve_stress_config(args.stress_config)
    if args.indices:
        files = []
        for raw_index in args.indices.split(","):
            index = int(raw_index.strip())
            candidates = [
                input_dir / f"yzc_v8_ue_{index}.csv",
                input_dir / f"yzc_v8_ue_{index}_interpolation.csv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    files.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"No trajectory CSV found for UE index {index}")
    else:
        files = sorted(input_dir.glob("yzc_v8_ue_*.csv"))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError(f"No yzc_v8_ue_*.csv files found under {input_dir}")

    rows = []
    manifest_rows = []
    for path in files:
        file_rows = process_file(path, stress_config)
        rows.extend(file_rows)
        manifest_rows.append(
            {
                "source_file": str(path),
                "event_rows": len(file_rows),
                "speed_mps": infer_speed_from_csv(path),
                "stress_config": stress_config["stress_config"],
                "bs_tx_power_dbm": stress_config["bs_tx_power_dbm"],
                "shadow_sigma_db": stress_config["shadow_sigma_db"],
            }
        )

    dataset_path = output_dir / "catboost_events.csv"
    manifest_path = output_dir / "catboost_events_manifest.csv"
    pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_csv(dataset_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    write_feature_schema(output_dir / "catboost_feature_schema.json")
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-temporal CatBoost event data from yzc_v8 trajectory logs.")
    parser.add_argument("--input-dir", default=str(TRAJECTORY_DIR))
    parser.add_argument("--output-dir", default=str(CATBOOST_DATA_DIR))
    parser.add_argument("--indices", default="")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--stress-config", default="nominal")
    return parser.parse_args()


if __name__ == "__main__":
    output = build_dataset(parse_args())
    print(f"CatBoost event dataset written to {output}")
