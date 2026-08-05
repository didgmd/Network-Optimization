from __future__ import annotations

# Public release alias.
# The implementation remains identical to the original training utility.
# This wrapper preserves the lowercase filename convention used in HARCADO.

from train_catBoost import *


if __name__ == "__main__":
    output_dir = train_catboost_model(parse_args())
    print(f"CatBoost metrics written to {output_dir}")
