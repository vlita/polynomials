#!/usr/bin/env python3
"""
Build a dataframe for one overlap system from system_overlap_ranked.csv.

Given a system_id (row id from find_common_systems.py output), this script
collects matching rows from all method dataframes and writes a consolidated
dataframe to overlap_dfs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from find_common_systems import extract_formula, geometry_signature, normalize_basis


def _get_qcfractal_id(row: dict[str, Any]) -> Any:
    for key in ("qcfractal id", "qcfractal_id", "qcfractal-id"):
        if key in row:
            return row[key]
    return None


def build_for_system(
    system_id: int,
    ranked_csv: Path,
    input_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranked = pd.read_csv(ranked_csv)
    if "system_id" not in ranked.columns:
        raise KeyError(
            f"'{ranked_csv}' does not contain 'system_id'. "
            "Regenerate ranking with updated find_common_systems.py."
        )

    match = ranked[ranked["system_id"] == system_id]
    if match.empty:
        raise ValueError(f"system_id {system_id} not found in {ranked_csv}")
    target = match.iloc[0].to_dict()

    target_formula = str(target["formula"])
    target_basis = str(target["basis_set"])
    target_geom_sig = str(target["geometry_signature"])

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    method_pickles = sorted(input_dir.glob("*.pkl"))
    if not method_pickles:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    for pkl_path in method_pickles:
        method = pkl_path.stem
        print(f"Scanning {pkl_path.name} ...", flush=True)
        df = pd.read_pickle(pkl_path)

        if "qcel_dimer" not in df.columns or "basis_set" not in df.columns:
            counts[method] = 0
            print(f"  skipped ({method} missing qcel_dimer or basis_set)", flush=True)
            continue

        basis_mask = df["basis_set"].map(normalize_basis) == target_basis
        subset = df.loc[basis_mask]

        n_match = 0
        for idx, row in subset.iterrows():
            dimer = row.get("qcel_dimer")
            if dimer is None:
                continue

            if extract_formula(dimer) != target_formula:
                continue
            if geometry_signature(dimer) != target_geom_sig:
                continue

            rec = row.to_dict()
            rec["source_method"] = method
            rec["source_row_index"] = idx
            rec["system_id"] = system_id
            rec["source_qcfractal_id"] = _get_qcfractal_id(rec)
            rows.append(rec)
            n_match += 1

        counts[method] = n_match
        print(f"  matches: {n_match}", flush=True)

    overlap_df = pd.DataFrame(rows)
    count_df = pd.DataFrame(
        [{"method": m, "matches": counts[m]} for m in sorted(counts.keys())]
    )
    return overlap_df, count_df, target


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build per-system overlap dataframe.")
    parser.add_argument("system_id", type=int, help="Row id from system_overlap_ranked.csv")
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=root / "system_overlap_ranked.csv",
        help="Path to ranked overlap CSV",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root.parent / "unfiltered_method_dfs",
        help="Directory with method pickle dataframes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "overlap_dfs",
        help="Directory for built overlap dataframes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ranked_csv = args.ranked_csv.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_df, count_df, target = build_for_system(
        system_id=args.system_id,
        ranked_csv=ranked_csv,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    out_prefix = f"system_{args.system_id}"
    out_pkl = output_dir / f"{out_prefix}.pkl"
    out_csv = output_dir / f"{out_prefix}.csv"
    out_counts = output_dir / f"{out_prefix}_method_counts.csv"

    overlap_df.to_pickle(out_pkl)
    overlap_df.to_csv(out_csv, index=False)
    count_df.to_csv(out_counts, index=False)

    print()
    print(f"system_id: {args.system_id}")
    print(f"formula: {target['formula']}")
    print(f"basis_set: {target['basis_set']}")
    print(f"geometry_signature: {target['geometry_signature']}")
    print(f"rows_written: {len(overlap_df)}")
    print(f"methods_with_matches: {(count_df['matches'] > 0).sum()}")
    print(f"Wrote: {out_pkl}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_counts}")


if __name__ == "__main__":
    main()
