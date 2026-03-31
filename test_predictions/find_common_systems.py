#!/usr/bin/env python3
"""
Rank geometry+basis systems by cross-method coverage and frequency.

A "system" is defined as:
    (qcel_dimer geometry, basis_set)

To reduce expensive geometry comparisons, rows are bucketed by:
    (molecular_formula, basis_set)
first, then geometry signatures are compared within each bucket.

Outputs (written to --output-dir):
    - system_overlap_ranked.csv
    - system_overlap_in_all_methods.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORMULA_KEYS = (
    "molecular_formula",
    "formula",
    "molecule_formula",
)

FORMULA_LIKE_PATTERN = r"^[A-Z][A-Za-z0-9()]*$"


def extract_formula(dimer: Any) -> str:
    """Extract molecular formula from qcel_dimer with several safe fallbacks."""
    identifiers = getattr(dimer, "identifiers", None)

    if isinstance(identifiers, dict):
        for key in FORMULA_KEYS:
            value = identifiers.get(key)
            if value:
                return str(value)
    elif identifiers is not None:
        for key in FORMULA_KEYS:
            value = getattr(identifiers, key, None)
            if value:
                return str(value)

    for key in FORMULA_KEYS:
        value = getattr(dimer, key, None)
        if value:
            return str(value)

    # Common fallback in these datasets: formula-like name/comment fields.
    for key in ("name", "comment"):
        value = getattr(dimer, key, None)
        if isinstance(value, str) and value.strip():
            token = value.strip()
            if re.match(FORMULA_LIKE_PATTERN, token):
                return token

    if identifiers is not None:
        for key in ("name", "comment"):
            value = getattr(identifiers, key, None)
            if isinstance(value, str) and value.strip():
                token = value.strip()
                if re.match(FORMULA_LIKE_PATTERN, token):
                    return token

    getter = getattr(dimer, "get_molecular_formula", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:
            pass

    return "UNKNOWN"


def _as_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_serializable(v) for v in value]
    return str(value)


def geometry_signature(dimer: Any, decimals: int = 10) -> str:
    """Build a stable geometry signature from key qcel_dimer fields."""
    symbols = tuple(map(str, getattr(dimer, "symbols", ())))
    geometry = np.asarray(getattr(dimer, "geometry", ()), dtype=float)

    if geometry.size:
        rounded = np.round(geometry, decimals=decimals)
        geom_digest = hashlib.sha1(rounded.tobytes()).hexdigest()
    else:
        geom_digest = "no-geometry"

    fragments = _as_serializable(getattr(dimer, "fragments", None))
    mol_charge = _as_serializable(getattr(dimer, "molecular_charge", None))
    mol_mult = _as_serializable(getattr(dimer, "molecular_multiplicity", None))

    payload = {
        "symbols": symbols,
        "geom_digest": geom_digest,
        "fragments": fragments,
        "molecular_charge": mol_charge,
        "molecular_multiplicity": mol_mult,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(packed.encode("utf-8")).hexdigest()


def normalize_basis(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def iter_method_pickles(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")
    return files


def rank_systems(input_dir: Path, progress_every: int) -> tuple[pd.DataFrame, list[str]]:
    per_system = defaultdict(lambda: {
        "total_occurrences": 0,
        "methods": set(),
        "per_method_counts": defaultdict(int),
    })

    method_names: list[str] = []

    pickle_files = iter_method_pickles(input_dir)
    total_methods = len(pickle_files)

    for method_idx, pkl_path in enumerate(pickle_files, start=1):
        method = pkl_path.stem
        method_names.append(method)

        print(
            f"[{method_idx}/{total_methods}] Loading {pkl_path.name}...",
            flush=True,
        )
        t0 = time.perf_counter()

        df = pd.read_pickle(pkl_path)
        if "qcel_dimer" not in df.columns:
            raise KeyError(f"Missing 'qcel_dimer' column in {pkl_path.name}")
        if "basis_set" not in df.columns:
            raise KeyError(f"Missing 'basis_set' column in {pkl_path.name}")

        n_rows = len(df)
        print(f"[{method}] Rows to process: {n_rows}", flush=True)

        valid_rows = 0
        for row_idx, row in enumerate(df.itertuples(index=False), start=1):
            dimer = getattr(row, "qcel_dimer", None)
            if dimer is None:
                continue

            basis = normalize_basis(getattr(row, "basis_set", None))
            if not basis:
                continue

            valid_rows += 1

            # First-pass grouping by formula (cheap), then geometry signature.
            formula = extract_formula(dimer)
            geom_sig = geometry_signature(dimer)
            system_key = (formula, geom_sig, basis)

            stats = per_system[system_key]
            stats["total_occurrences"] += 1
            stats["methods"].add(method)
            stats["per_method_counts"][method] += 1

            if progress_every > 0 and row_idx % progress_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    (
                        f"[{method}] processed {row_idx}/{n_rows} rows "
                        f"({valid_rows} valid), unique systems so far: {len(per_system)} "
                        f"elapsed: {elapsed:.1f}s"
                    ),
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        print(
            (
                f"[{method}] done. processed {n_rows} rows, valid {valid_rows}, "
                f"global unique systems: {len(per_system)}, elapsed: {elapsed:.1f}s"
            ),
            flush=True,
        )

    all_methods = set(method_names)
    records = []
    for (formula, geom_sig, basis), stats in per_system.items():
        methods_sorted = sorted(stats["methods"])
        missing_methods = [m for m in method_names if m not in stats["methods"]]
        per_method = {m: stats["per_method_counts"].get(m, 0) for m in method_names}
        records.append(
            {
                "formula": formula,
                "basis_set": basis,
                "geometry_signature": geom_sig,
                "method_count": len(stats["methods"]),
                "total_occurrences": stats["total_occurrences"],
                "present_in_all_methods": stats["methods"] == all_methods,
                "methods": ";".join(methods_sorted),
                "missing_method_count": len(missing_methods),
                "missing_methods": ";".join(missing_methods),
                **{f"count_{m}": per_method[m] for m in method_names},
            }
        )

    ranked = pd.DataFrame.from_records(records)
    ranked.sort_values(
        by=["method_count", "total_occurrences", "formula", "basis_set"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    ranked.reset_index(drop=True, inplace=True)
    ranked.insert(0, "system_id", ranked.index + 1)
    return ranked, method_names


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    default_input = default_root / "unfiltered_method_dfs"
    default_output = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Find the most shared geometry+basis systems across method dataframes."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing method .pkl dataframes (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for output CSV files (default: {default_output})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Number of top systems to print (default: all; use 0 for all)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help=(
            "Print row-level progress every N rows within each dataframe "
            "(default: 100000, set 0 to disable)"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked, method_names = rank_systems(input_dir, progress_every=args.progress_every)

    ranked_path = output_dir / "system_overlap_ranked.csv"
    all_methods_path = output_dir / "system_overlap_in_all_methods.csv"

    ranked.to_csv(ranked_path, index=False)
    in_all = ranked[ranked["present_in_all_methods"]].copy()
    in_all.to_csv(all_methods_path, index=False)

    print("Methods processed:", len(method_names))
    print("Rows in ranked table:", len(ranked))
    print("Systems present in all methods:", len(in_all))
    print()

    cols = [
        "system_id",
        "formula",
        "basis_set",
        "method_count",
        "total_occurrences",
        "methods",
        "missing_methods",
    ]
    top_n = len(ranked) if args.top_n <= 0 else min(args.top_n, len(ranked))
    print(f"Top {top_n} systems by coverage/frequency:")
    if ranked.empty:
        print("  (no systems found)")
    else:
        top = ranked.loc[:, cols].head(top_n).copy()
        top["methods"] = top["methods"].replace("", "-")
        top["missing_methods"] = top["missing_methods"].replace("", "-")
        print(top.to_string(index=False))

    print()
    print(f"Wrote: {ranked_path}")
    print(f"Wrote: {all_methods_path}")


if __name__ == "__main__":
    main()
