from __future__ import annotations

import math

DEFAULT_BS_TX_POWER_DBM = 46.0
DEFAULT_SHADOW_SIGMA_DB = math.sqrt(2.0)

STRESS_CONFIGS = {
    "nominal": {
        "bs_tx_power_dbm": DEFAULT_BS_TX_POWER_DBM,
        "shadow_sigma_db": DEFAULT_SHADOW_SIGMA_DB,
    },
    "extreme_r1": {
        "bs_tx_power_dbm": 43.0,
        "shadow_sigma_db": 3.0,
    },
    "extreme_r2": {
        "bs_tx_power_dbm": 40.0,
        "shadow_sigma_db": 4.0,
    },
}


def resolve_stress_config(name: str) -> dict[str, float | str]:
    key = (name or "nominal").strip()
    if key not in STRESS_CONFIGS:
        raise ValueError(f"Unsupported stress config {key!r}")
    values = STRESS_CONFIGS[key]
    return {
        "stress_config": key,
        "bs_tx_power_dbm": float(values["bs_tx_power_dbm"]),
        "shadow_sigma_db": float(values["shadow_sigma_db"]),
    }
