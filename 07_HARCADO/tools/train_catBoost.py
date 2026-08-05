from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from catboost_feature_schema import (
    CATBOOST_CATEGORICAL_FEATURES,
    CATBOOST_LABEL_COLUMN,
    CATBOOST_METADATA_COLUMNS,
    validate_feature_frame,
    write_feature_schema,
)
from experiment_contract import CATBOOST_DATA_DIR, LOG_ROOT, MODEL_PATH, ensure_dir, nearest_speed


CAT_FEATURES = CATBOOST_CATEGORICAL_FEATURES
LABEL_COLUMN = CATBOOST_LABEL_COLUMN

TRAIN_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.01,
    "depth": 10,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "verbose": 100,
    "border_count": 254,
    "auto_class_weights": "Balanced",
}


def load_all_csvs(folder_path: Path, recursive: bool = False, smoke_rows: int | None = None) -> pd.DataFrame:
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_paths = sorted(folder_path.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {folder_path}")

    frames = []
    remaining = smoke_rows
    for path in csv_paths:
        read_kwargs = {}
        if remaining is not None:
            if remaining <= 0:
                break
            read_kwargs["nrows"] = remaining
        frame = pd.read_csv(path, **read_kwargs)
        if LABEL_COLUMN not in frame.columns:
            continue
        frame["__source_file"] = str(path)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)

    if not frames:
        raise ValueError(f"No rows were loaded from {folder_path}")
    return pd.concat(frames, ignore_index=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a '{LABEL_COLUMN}' column in CatBoost data. "
            "Use CGDQN_CATBOOST_DATA_DIR to point at the non-temporal labeled dataset."
        )

    metadata_columns = [col for col in CATBOOST_METADATA_COLUMNS if col in df.columns]
    metadata = df[metadata_columns].copy()
    y = df[LABEL_COLUMN].astype(int)
    x = validate_feature_frame(df, allow_label=True, allow_metadata=True)

    for col in CAT_FEATURES:
        x[col] = x[col].astype(int)
    return x, y, metadata


def add_speed_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "speed_mps" in df.columns:
        speed = df["speed_mps"].astype(float)
    elif {"VelX", "VelY"}.issubset(df.columns):
        speed = (df["VelX"].astype(float) ** 2 + df["VelY"].astype(float) ** 2) ** 0.5
    else:
        df["speed_mps"] = pd.NA
        return df
    df["speed_mps"] = speed.map(nearest_speed)
    return df


def maybe_resample(
    x_train: pd.DataFrame, y_train: pd.Series, method: str
) -> tuple[pd.DataFrame, pd.Series]:
    if method == "none":
        return x_train, y_train
    if method != "smotetomek":
        raise ValueError(f"Unsupported resampling method: {method}")

    try:
        from imblearn.combine import SMOTETomek
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SMOTETomek resampling requires imbalanced-learn. "
            "Install imbalanced-learn or pass --resample none."
        ) from exc

    x_resampled, y_resampled = SMOTETomek(random_state=42).fit_resample(x_train, y_train)
    x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    for col in CAT_FEATURES:
        x_resampled[col] = x_resampled[col].round().astype(int)
    return x_resampled, y_resampled.astype(int)


def aggregate_metrics(y_true: pd.Series, y_pred, split: str, speed_mps=None) -> dict:
    return {
        "split": split,
        "speed_mps": speed_mps,
        "support": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "late_recall_class_1": recall_score(
            y_true == 1,
            pd.Series(y_pred).astype(int).to_numpy() == 1,
            zero_division=0,
        ),
    }


def classwise_metrics(y_true: pd.Series, y_pred, split: str) -> list[dict]:
    labels = sorted(set(y_true.astype(int)).union(set(pd.Series(y_pred).astype(int))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "split": split,
            "class": int(label),
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "support": int(support[idx]),
        }
        for idx, label in enumerate(labels)
    ]


def train_catboost_model(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).resolve()
    metrics_dir = ensure_dir(Path(args.metrics_dir).resolve())
    model_out = Path(args.model_out).resolve()

    df = load_all_csvs(data_dir, recursive=args.recursive, smoke_rows=args.smoke_rows)
    df = add_speed_groups(df)
    x, y, metadata = prepare_features(df)

    stratify = y if y.nunique() > 1 else None
    x_train, x_val, y_train, y_val, _, meta_val = train_test_split(
        x,
        y,
        metadata,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    x_fit, y_fit = maybe_resample(x_train, y_train, args.resample)
    params = dict(TRAIN_PARAMS)
    params["task_type"] = args.task_type
    params["verbose"] = args.verbose
    params["iterations"] = args.iterations

    train_pool = Pool(x_fit, y_fit, cat_features=CAT_FEATURES)
    val_pool = Pool(x_val, y_val, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    y_pred = model.predict(x_val).reshape(-1).astype(int)

    overall_rows = [aggregate_metrics(y_val, y_pred, "validation")]
    class_rows = classwise_metrics(y_val, y_pred, "validation")

    speed_rows = []
    if "speed_mps" in meta_val.columns:
        pred_series = pd.Series(y_pred, index=y_val.index)
        for speed in sorted(meta_val["speed_mps"].dropna().unique()):
            mask = meta_val["speed_mps"] == speed
            if mask.any():
                speed_rows.append(
                    aggregate_metrics(
                        y_val.loc[mask],
                        pred_series.loc[mask].to_numpy(),
                        "validation",
                        speed_mps=float(speed),
                    )
                )

    ensure_dir(model_out.parent)
    model.save_model(str(model_out))
    pd.DataFrame(overall_rows).to_csv(metrics_dir / "catboost_overall_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(metrics_dir / "catboost_classwise_metrics.csv", index=False)
    pd.DataFrame(speed_rows).to_csv(metrics_dir / "catboost_speedwise_metrics.csv", index=False)
    write_feature_schema(metrics_dir / "catboost_feature_schema.json")
    write_feature_schema(model_out.parent / "catboost_feature_schema.json")
    return metrics_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost and emit traceable metrics CSVs.")
    parser.add_argument("--data-dir", default=str(CATBOOST_DATA_DIR))
    parser.add_argument("--model-out", default=str(MODEL_PATH))
    parser.add_argument("--metrics-dir", default=str(LOG_ROOT / "catboost_metrics"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resample", choices=["none", "smotetomek"], default="smotetomek")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1500)
    return parser.parse_args()


if __name__ == "__main__":
    output_dir = train_catboost_model(parse_args())
    print(f"CatBoost metrics written to {output_dir}")
