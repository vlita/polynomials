#!/home/vlita3/miniconda3/envs/p4_qcml/bin/python
"""
Filtering script for per-method timing dataframes.

Reads from unfiltered_method_dfs/{method}.pkl, applies a similarity-based
deduplication on the 'predicted log user time' column, and writes the result
to filtered_method_dfs/{method}.pkl.

Filtering algorithm
-------------------
1. Sort rows by 'predicted log user time'.
2. Assign each row to a bin of width THRESHOLD (in log10 units).
3. Randomly sample one row per non-empty bin (reproducible via RANDOM_SEED).

Methods processed: HF, MP2, B2PLYP-D3, B3LYP-D3, M05-2X, PBE-D3, wB97X-D, wB97X-V
FNO-CCSD, FNO-CCSD(T), CCSD, CCSD(T) are not processed by this script.
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Configuration — adjust these as needed
# ---------------------------------------------------------------------------
THRESHOLD   = 0.005   # bin width in log10(time) units
RANDOM_SEED = 42

UNFILTERED_DIR = "unfiltered_method_dfs"
FILTERED_DIR   = "filtered_method_dfs"

METHODS_TO_FILTER = [
    "HF",
    "MP2",
    "B2PLYP-D3",
    "B3LYP-D3",
    "M05-2X",
    "PBE-D3",
    "wB97X-D",
    "wB97X-V",
]

os.makedirs(FILTERED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def filter_method(method: str, threshold: float, rng: np.random.Generator) -> None:
    in_path = os.path.join(UNFILTERED_DIR, f"{method}.pkl")
    df = pd.read_pickle(in_path)

    pred_col = "predicted log user time"
    if pred_col not in df.columns:
        print(f"  WARNING: '{pred_col}' column missing — skipping {method}.")
        return

    n_before = len(df)

    # Assign each row to a bin index based on predicted log time
    pred = df[pred_col]
    pred_min = pred.min()
    df["_bin"] = ((pred - pred_min) / threshold).astype(int)

    # Sample one row per bin
    df_filtered = (
        df.groupby("_bin", group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=int(rng.integers(0, 2**31))),
               include_groups=False)
        .reset_index(drop=True)
    )
    df_filtered = df_filtered.drop(columns=["_bin"], errors="ignore")

    n_after = len(df_filtered)
    pct_kept = 100 * n_after / n_before

    out_path = os.path.join(FILTERED_DIR, f"{method}.pkl")
    df_filtered.to_pickle(out_path)

    print(f"  {n_before} → {n_after} rows kept ({pct_kept:.1f}%)")
    print(f"  Saved: {out_path}")

    # Sanity check: confirm prediction range is preserved
    print(f"  Prediction range: [{df_filtered[pred_col].min():.4f}, {df_filtered[pred_col].max():.4f}]"
          f"  (was [{pred.min():.4f}, {pred.max():.4f}])")


def main():
    print(f"Threshold: {THRESHOLD} log10 units  |  Random seed: {RANDOM_SEED}\n")

    # Single RNG instance so each method gets a different but reproducible seed
    rng = np.random.default_rng(RANDOM_SEED)

    for method in METHODS_TO_FILTER:
        print(f"{'='*60}")
        print(f"METHOD: {method}")
        filter_method(method, THRESHOLD, rng)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
