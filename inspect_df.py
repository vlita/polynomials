"""
inspect_df.py  --  Inspect method dataframes from filtered/unfiltered_method_dfs/.

Usage:
    python inspect_df.py <method> [--unfiltered]
    python inspect_df.py <method> [--unfiltered] --range <lo> <hi>
    python inspect_df.py <method> [--unfiltered] --psi4 <row_index>

Specifications:
    (default)        Print basic statistics on actual log10(user_time) for the df.
    --range lo hi    Display rows whose actual log10(user_time) falls in [lo, hi].
    --psi4 idx       Print the psi4 output file for the row at integer index idx.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FILTERED_DIR   = Path(__file__).parent / "filtered_method_dfs"
UNFILTERED_DIR = Path(__file__).parent / "unfiltered_method_dfs"

# Columns to show when displaying rows (omit bulky geometry/output columns)
DISPLAY_COLS = [
    "id", "basis_set", "n_atoms", "n_occupied", "n_virtual",
    "user_time (parse)", "predicted log user time", "job status",
]


def load_df(method: str, unfiltered: bool) -> pd.DataFrame:
    directory = UNFILTERED_DIR if unfiltered else FILTERED_DIR
    path = directory / f"{method}.pkl"
    if not path.exists():
        available = sorted(p.stem for p in directory.glob("*.pkl"))
        print(f"Error: '{path}' not found.")
        print(f"Available methods in {'unfiltered' if unfiltered else 'filtered'}: {available}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def add_log_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add actual log10(user_time) as a column (does not modify original)."""
    df = df.copy()
    df["log_user_time"] = np.log10(df["user_time (parse)"])
    return df


# ── Specification 1 ───────────────────────────────────────────────────────────

def print_info(df: pd.DataFrame, method: str, unfiltered: bool) -> None:
    source = "unfiltered" if unfiltered else "filtered"
    print(f"\n=== {method} ({source}) ===")
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")

    log_ut = df["log_user_time"]
    stats = log_ut.describe()

    print("\nActual log10(user_time) statistics:")
    print(f"  count : {stats['count']:.0f}")
    print(f"  mean  : {stats['mean']:.4f}")
    print(f"  std   : {stats['std']:.4f}")
    print(f"  min   : {stats['min']:.4f}")
    print(f"  25%   : {stats['25%']:.4f}")
    print(f"  50%   : {stats['50%']:.4f}")
    print(f"  75%   : {stats['75%']:.4f}")
    print(f"  max   : {stats['max']:.4f}")
    print()


# ── Specification 2 ───────────────────────────────────────────────────────────

def print_range(df: pd.DataFrame, lo: float, hi: float) -> None:
    mask = (df["log_user_time"] >= lo) & (df["log_user_time"] <= hi)
    subset = df.loc[mask]
    print(f"\nRows with log10(user_time) in [{lo}, {hi}]: {len(subset)} found\n")
    if subset.empty:
        return

    # Show whichever display columns are present
    cols = [c for c in DISPLAY_COLS if c in subset.columns] + ["log_user_time"]
    with pd.option_context("display.max_columns", None, "display.width", 0):
        print(subset[cols].to_string())
    print()


# ── Specification 3 ───────────────────────────────────────────────────────────

def print_psi4_output(df: pd.DataFrame, idx: int) -> None:
    if idx not in df.index:
        print(f"Error: index {idx} not in dataframe (valid range: {df.index.min()}–{df.index.max()}).")
        sys.exit(1)
    output = df.at[idx, "psi4 output"]
    if pd.isna(output) or output == "":
        print(f"No psi4 output stored for row {idx}.")
    else:
        print(f"\n=== psi4 output for row {idx} ===\n")
        print(output)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect method dataframes (filtered or unfiltered).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("method", help="Method name (e.g. HF, MP2, B3LYP-D3)")
    parser.add_argument(
        "--unfiltered", action="store_true",
        help="Load from unfiltered_method_dfs/ instead of filtered_method_dfs/",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--range", nargs=2, type=float, metavar=("LO", "HI"),
        help="Show rows with actual log10(user_time) in [LO, HI]",
    )
    group.add_argument(
        "--psi4", type=int, metavar="IDX",
        help="Print the psi4 output for the row at integer index IDX",
    )

    args = parser.parse_args()

    df = add_log_col(load_df(args.method, args.unfiltered))

    if args.range is not None:
        print_range(df, args.range[0], args.range[1])
    elif args.psi4 is not None:
        print_psi4_output(df, args.psi4)
    else:
        print_info(df, args.method, args.unfiltered)


if __name__ == "__main__":
    main()
