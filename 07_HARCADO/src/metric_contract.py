from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


RLF_SINR_THRESHOLD_DB = env_float("CGDQN_RLF_SINR_THRESHOLD_DB", -5.0)
