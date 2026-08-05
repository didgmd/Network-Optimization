from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CATBOOST_SCHEMA_VERSION = "causal_v2"
CATBOOST_LABEL_COLUMN = "label"
CATBOOST_METADATA_COLUMNS = ["speed_mps", "__source_file", "source_file", "ue_index"]
CATBOOST_CATEGORICAL_FEATURES = ["SourceBsId", "TargetBsId"]
CATBOOST_FEATURE_COLUMNS = [
    "SourceBsId",
    "TargetBsId",
    "SourceRsrp",
    "TargetRsrp",
    "SourceSinr",
    "TargetSinr",
    "TargetDis",
    "VelX",
    "VelY",
    "Direction",
    "TargetSourceRsrpGap",
    "PreMeanSourceRsrp",
    "SourceRsrpDrop",
]
PROHIBITED_FEATURE_TOKENS = [
    "Post",
    "After",
    "DeltaRsrp",
    "RsrpGain",
    "t_plus",
    "t^+",
]


def feature_schema_payload() -> dict:
    return {
        "schema_version": CATBOOST_SCHEMA_VERSION,
        "feature_columns": CATBOOST_FEATURE_COLUMNS,
        "categorical_features": CATBOOST_CATEGORICAL_FEATURES,
        "label_column": CATBOOST_LABEL_COLUMN,
        "metadata_columns": CATBOOST_METADATA_COLUMNS,
        "prohibited_feature_tokens": PROHIBITED_FEATURE_TOKENS,
        "causality_contract": (
            "Predictor inputs are available at or before the control epoch. "
            "Post-HO variables and completed-event RSRP gain are outcome/reward fields only."
        ),
    }


def validate_feature_columns(columns: list[str], *, allow_label: bool, allow_metadata: bool) -> None:
    allowed = set(CATBOOST_FEATURE_COLUMNS)
    if allow_label:
        allowed.add(CATBOOST_LABEL_COLUMN)
    if allow_metadata:
        allowed.update(CATBOOST_METADATA_COLUMNS)

    missing = set(CATBOOST_FEATURE_COLUMNS).difference(columns)
    unexpected = set(columns).difference(allowed)
    if missing:
        raise ValueError(f"Missing CatBoost causal_v2 feature columns: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected CatBoost columns under causal_v2 schema: {sorted(unexpected)}")

    prohibited = [
        column
        for column in columns
        if column in CATBOOST_FEATURE_COLUMNS
        for token in PROHIBITED_FEATURE_TOKENS
        if token.lower() in column.lower()
    ]
    if prohibited:
        raise ValueError(f"Prohibited post-HO/ambiguous CatBoost feature columns: {sorted(set(prohibited))}")


def validate_feature_frame(
    frame: pd.DataFrame, *, allow_label: bool = True, allow_metadata: bool = True
) -> pd.DataFrame:
    validate_feature_columns(
        list(frame.columns),
        allow_label=allow_label,
        allow_metadata=allow_metadata,
    )
    ordered = frame[CATBOOST_FEATURE_COLUMNS].copy()
    for column in CATBOOST_CATEGORICAL_FEATURES:
        ordered[column] = ordered[column].astype(int)
    return ordered


def write_feature_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_schema_payload(), indent=2), encoding="utf-8")
    return path
