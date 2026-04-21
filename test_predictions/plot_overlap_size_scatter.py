#!/usr/bin/env python3
"""
Create predicted-time vs system-size scatter plots from overlap systems.

Sampling strategy:
- Candidate systems come from overlap_dfs/system_*.pkl and system_overlap_ranked.csv
- Use a real per-system size value (mode of nocc+nvirt rows)
- Sample systems on a fixed size step (e.g., every 25 basis functions)
- No averaging in plotted points: each plotted point is a real row prediction

Outputs:
- timing_eval/plots/size_scatter/predicted_vs_size_non_true.png
- timing_eval/plots/size_scatter/predicted_vs_size_true.png
- timing_eval/plots/size_scatter/sampled_systems.csv
- timing_eval/plots/size_scatter/points_non_true.csv
- timing_eval/plots/size_scatter/points_true.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from find_common_systems import extract_formula, geometry_signature, normalize_basis
from rank_overlap_timings import (
    VAR_MAPPING,
    fit_all_methods,
    get_polynomial_variables,
    predict_log_time,
    prepare_df,
)


TARGET_METHODS = [
    "HF",
    "PBE-D3",
    "B3LYP-D3",
    "M05-2X",
    "wB97X-D",
    "wB97X-V",
    "MP2",
    "B2PLYP-D3",
    "FNO-CCSD",
    "FNO-CCSD(T)",
]


def safe_name(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    token = token.strip("._-")
    return token or "unknown"


def load_formula_map(ranked_csv: Path) -> dict[str, str]:
    if not ranked_csv.exists():
        return {}
    df = pd.read_csv(ranked_csv)
    if "system_id" not in df.columns or "formula" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        out[str(int(row["system_id"]))] = str(row["formula"])
    return out


def make_system_key(formula: str, geom_sig: str, basis: str) -> str:
    return f"{formula}|{geom_sig}|{basis}"


def load_method_frames_with_keys(input_dir: Path, methods: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for method in methods:
        pkl_path = input_dir / f"{method}.pkl"
        if not pkl_path.exists():
            continue

        print(f"Preparing keys for {method} ...", flush=True)
        df = pd.read_pickle(pkl_path)
        if "qcel_dimer" not in df.columns:
            continue
        if "basis_set" not in df.columns and "basis" not in df.columns:
            continue

        df = prepare_df(df)

        keys: list[str] = []
        n_rows = len(df)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            dimer = getattr(row, "qcel_dimer", None)
            if dimer is None:
                keys.append("")
                continue

            basis = normalize_basis(getattr(row, "basis", None))
            if not basis:
                keys.append("")
                continue

            formula = extract_formula(dimer)
            geom_sig = geometry_signature(dimer)
            keys.append(make_system_key(formula, geom_sig, basis))

            if i % 50000 == 0:
                print(f"  {method}: keyed {i}/{n_rows} rows", flush=True)

        df = df.copy()
        df["system_key"] = keys
        df = df[df["system_key"] != ""].copy()
        frames[method] = df
    return frames


def collect_candidates_from_ranked(
    ranked_csv: Path,
    methods: list[str],
    method_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    ranked = pd.read_csv(ranked_csv)
    if not {"system_id", "formula", "geometry_signature", "basis_set"}.issubset(ranked.columns):
        raise KeyError("Ranked CSV missing one of required columns: system_id, formula, geometry_signature, basis_set")

    ranked = ranked.copy()
    ranked["basis_set"] = ranked["basis_set"].map(normalize_basis)
    ranked["system_key"] = ranked.apply(
        lambda r: make_system_key(str(r["formula"]), str(r["geometry_signature"]), str(r["basis_set"])),
        axis=1,
    )

    key_to_sizes: dict[str, list[float]] = {}
    key_to_methods: dict[str, set[str]] = {}
    for method, df in method_frames.items():
        for key, grp in df.groupby("system_key"):
            if key not in key_to_sizes:
                key_to_sizes[key] = []
                key_to_methods[key] = set()
            key_to_sizes[key].extend(grp["nbf"].dropna().to_numpy(dtype=float).tolist())
            key_to_methods[key].add(method)

    rows: list[dict[str, object]] = []
    for _, r in ranked.iterrows():
        key = str(r["system_key"])
        if key not in key_to_sizes:
            continue
        sizes = np.asarray(key_to_sizes[key], dtype=float)
        if sizes.size == 0:
            continue
        size_mode = pd.Series(sizes).mode()
        if size_mode.empty:
            continue

        meths = sorted(key_to_methods[key])
        target_present = [m for m in methods if m in key_to_methods[key]]
        if not target_present:
            continue

        rows.append(
            {
                "system_id": int(r["system_id"]),
                "formula": str(r["formula"]),
                "basis_set": str(r["basis_set"]),
                "system_key": key,
                "size_ref": float(size_mode.iloc[0]),
                "n_methods_present": len(target_present),
                "methods_present": ";".join(meths),
            }
        )

    return pd.DataFrame(rows)


def sample_by_size_step(
    candidates: pd.DataFrame,
    size_step: int,
    systems_per_step: int,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    work = candidates.copy().sort_values(["size_ref", "n_methods_present"], ascending=[True, False]).reset_index(drop=True)
    min_size = int(np.floor(work["size_ref"].min()))
    max_size = int(np.ceil(work["size_ref"].max()))
    size_step = max(1, int(size_step))
    systems_per_step = max(1, int(systems_per_step))

    sampled_parts: list[pd.DataFrame] = []
    for step_idx, lo in enumerate(range(min_size, max_size + 1, size_step)):
        hi = lo + size_step
        bucket = work[(work["size_ref"] >= lo) & (work["size_ref"] < hi)].copy()
        if bucket.empty:
            continue

        # Prefer systems that contain more methods, then closest to bucket midpoint.
        mid = lo + (size_step / 2.0)
        bucket["dist_mid"] = np.abs(bucket["size_ref"] - mid)
        bucket.sort_values(["n_methods_present", "dist_mid"], ascending=[False, True], inplace=True)
        n_take = min(systems_per_step, len(bucket))
        sampled_parts.append(bucket.head(n_take).drop(columns=["dist_mid"]))

    if not sampled_parts:
        # fallback: sample a small set if step windows miss everything
        sampled = work.sample(n=min(20, len(work)), random_state=seed)
    else:
        sampled = pd.concat(sampled_parts, ignore_index=True)

    sampled = sampled.drop_duplicates(subset=["system_id"]) 
    sampled = sampled.sort_values("size_ref").reset_index(drop=True)
    return sampled


def build_points(
    sampled: pd.DataFrame,
    method_frames: dict[str, pd.DataFrame],
    system_feature_pool: dict[str, pd.DataFrame],
    fitted: dict[str, tuple[str, np.ndarray]],
    methods: list[str],
    fit_type: str,
) -> pd.DataFrame:
    pts: list[dict[str, object]] = []

    for _, sys_row in sampled.iterrows():
        system_key = str(sys_row["system_key"])

        for method in methods:
            if method not in fitted or method not in method_frames:
                continue
            poly_name, params = fitted[method]
            required_cols = [VAR_MAPPING.get(v, v) for v in get_polynomial_variables(poly_name)]

            grp = method_frames[method]
            grp = grp[grp["system_key"] == system_key].copy()
            grp = grp[grp["time(s)"] > 0]
            feature_source = "method_rows"
            if grp.empty:
                grp = system_feature_pool.get(system_key, pd.DataFrame()).copy()
                feature_source = "fallback_any_method"

            if grp.empty:
                continue

            missing = [c for c in required_cols if c not in grp.columns]
            if missing:
                continue

            grp = grp.dropna(subset=required_cols)
            if grp.empty:
                continue

            dedupe_cols = [c for c in ["nocc", "nvirt", "nbf_aux", "np_total", "nbf"] if c in grp.columns]
            if dedupe_cols:
                grp = grp.drop_duplicates(subset=dedupe_cols)

            try:
                pred = predict_log_time(params, grp, poly_name)
            except Exception:
                continue

            for nbf_val, pred_val in zip(grp["nbf"].to_numpy(dtype=float), pred):
                pts.append(
                    {
                        "system_id": int(sys_row["system_id"]),
                        "formula": str(sys_row["formula"]),
                        "method": method,
                        "fit_type": fit_type,
                        "poly_name": poly_name,
                        "feature_source": feature_source,
                        "nbf": float(nbf_val),
                        "predicted_log_time": float(pred_val),
                    }
                )

    out = pd.DataFrame(pts)
    if not out.empty:
        out.sort_values(["method", "nbf"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def plot_points(points: pd.DataFrame, methods: list[str], out_path: Path, title: str) -> None:
    if points.empty:
        return

    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in methods:
        sub = points[points["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["nbf"],
            sub["predicted_log_time"],
            s=46,
            alpha=0.88,
            color=color_map[method],
            label=method,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("System size (nocc + nvirt, log scale)")
    ax.set_ylabel("Predicted log10(time(s))")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_system_feature_pool(method_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    keep = ["system_key", "nocc", "nvirt", "nbf_aux", "np_total", "nbf"]

    for df in method_frames.values():
        cols = [c for c in keep if c in df.columns]
        if "system_key" not in cols:
            continue
        part = df[cols].copy()
        parts.append(part)

    if not parts:
        return {}

    all_features = pd.concat(parts, ignore_index=True)
    all_features = all_features.dropna(subset=["system_key", "nocc", "nvirt", "nbf"])
    dedupe_cols = [c for c in ["system_key", "nocc", "nvirt", "nbf_aux", "np_total", "nbf"] if c in all_features.columns]
    all_features = all_features.drop_duplicates(subset=dedupe_cols)

    out: dict[str, pd.DataFrame] = {}
    for key, grp in all_features.groupby("system_key"):
        out[str(key)] = grp.reset_index(drop=True)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Scatter predicted timing vs system size from overlap systems.")
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=here / "system_overlap_ranked.csv",
        help="CSV mapping system_id to formula",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=here.parent / "unfiltered_method_dfs",
        help="Directory with original method dataframes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "overlap_dfs" / "timing_eval" / "plots" / "size_scatter",
        help="Output directory for scatter plots and tables",
    )
    parser.add_argument("--size-step", type=int, default=25, help="Sample systems roughly every N basis functions")
    parser.add_argument("--systems-per-step", type=int, default=1, help="How many systems to keep per size step bucket")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for sampling")
    args = parser.parse_args()

    ranked_csv = args.ranked_csv.resolve()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_frames = load_method_frames_with_keys(input_dir, TARGET_METHODS)
    system_feature_pool = build_system_feature_pool(method_frames)
    candidates = collect_candidates_from_ranked(ranked_csv, TARGET_METHODS, method_frames)
    if candidates.empty:
        raise ValueError("No valid candidate systems found from ranked CSV + method data")

    sampled = sample_by_size_step(
        candidates,
        size_step=args.size_step,
        systems_per_step=args.systems_per_step,
        seed=args.seed,
    )
    sampled_csv = out_dir / "sampled_systems.csv"
    sampled.to_csv(sampled_csv, index=False)
    print(f"Sampled {len(sampled)} systems across size bins")
    print(f"Wrote: {sampled_csv}")

    methods = TARGET_METHODS
    fitted_non_true = fit_all_methods(input_dir, methods, fit_type="non_true")
    fitted_true = fit_all_methods(input_dir, methods, fit_type="true")

    pts_non_true = build_points(sampled, method_frames, system_feature_pool, fitted_non_true, methods, fit_type="non_true")
    pts_true = build_points(sampled, method_frames, system_feature_pool, fitted_true, methods, fit_type="true")

    pts_non_true_csv = out_dir / "points_non_true.csv"
    pts_true_csv = out_dir / "points_true.csv"
    pts_non_true.to_csv(pts_non_true_csv, index=False)
    pts_true.to_csv(pts_true_csv, index=False)

    non_true_plot = out_dir / "predicted_vs_size_non_true.png"
    true_plot = out_dir / "predicted_vs_size_true.png"

    plot_points(
        pts_non_true,
        methods,
        non_true_plot,
        title="Predicted log10(time) vs system size (non-true fits)",
    )
    plot_points(
        pts_true,
        methods,
        true_plot,
        title="Predicted log10(time) vs system size (true fits)",
    )

    print(f"Wrote: {pts_non_true_csv}")
    print(f"Wrote: {pts_true_csv}")
    print(f"Wrote: {non_true_plot}")
    print(f"Wrote: {true_plot}")


if __name__ == "__main__":
    main()
