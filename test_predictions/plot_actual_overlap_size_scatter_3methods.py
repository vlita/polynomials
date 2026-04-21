#!/usr/bin/env python3
"""
Create actual-time vs system-size scatter plot for systems shared by:
- B2PLYP-D3
- wB97X-D
- wB97X-V

Outputs:
- overlap_dfs/timing_eval/plots/size_scatter_3methods/actual_vs_size_common_three_methods.png
- overlap_dfs/timing_eval/plots/size_scatter_3methods/points_actual_common_three_methods.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from find_common_systems import extract_formula, geometry_signature, normalize_basis
from rank_overlap_timings import prepare_df


TARGET_METHODS = ["B2PLYP-D3", "wB97X-D", "wB97X-V"]


def make_system_key(formula: str, geom_sig: str, basis: str) -> str:
    return f"{formula}|{geom_sig}|{basis}"


def load_method_frames_with_keys(input_dir: Path, methods: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for method in methods:
        pkl_path = input_dir / f"{method}.pkl"
        if not pkl_path.exists():
            continue

        df = pd.read_pickle(pkl_path)
        if "qcel_dimer" not in df.columns:
            continue
        if "basis_set" not in df.columns and "basis" not in df.columns:
            continue

        df = prepare_df(df)

        keys: list[str] = []
        for row in df.itertuples(index=False):
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

        df = df.copy()
        df["system_key"] = keys
        df = df[df["system_key"] != ""].copy()
        frames[method] = df
    return frames


def build_common_points(method_frames: dict[str, pd.DataFrame], methods: list[str]) -> pd.DataFrame:
    available = [m for m in methods if m in method_frames]
    if len(available) != len(methods):
        missing = sorted(set(methods) - set(available))
        raise FileNotFoundError(f"Missing method dataframes for: {missing}")

    key_sets = [set(method_frames[m]["system_key"].unique()) for m in methods]
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for method in methods:
        df = method_frames[method]
        use = df[df["system_key"].isin(common_keys)].copy()
        use = use[use["time(s)"] > 0].copy()
        if use.empty:
            continue
        use["method"] = method
        rows.append(
            use[
                [
                    "method",
                    "system_key",
                    "nocc",
                    "nvirt",
                    "nbf",
                    "time(s)",
                    "log(time(s))",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"log(time(s))": "actual_log_time", "time(s)": "actual_time_s"})
    out = out[(out["nocc"] > 0) & (out["nvirt"] > 0)].copy()
    out["log_nocc"] = np.log10(out["nocc"].to_numpy(dtype=float))
    out["log_nvirt"] = np.log10(out["nvirt"].to_numpy(dtype=float))
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
            sub["actual_log_time"],
            s=46,
            alpha=0.88,
            color=color_map[method],
            label=method,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("System size (nocc + nvirt, log scale)")
    ax.set_ylabel("Actual log10(time(s))")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_points_3d(points: pd.DataFrame, methods: list[str], out_path: Path, title: str) -> None:
    if points.empty:
        return

    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig = plt.figure(figsize=(11.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for method in methods:
        sub = points[points["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["log_nocc"],
            sub["log_nvirt"],
            sub["actual_log_time"],
            s=18,
            alpha=0.62,
            color=color_map[method],
            depthshade=False,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color_map[m],
            markeredgecolor="none",
            label=m,
        )
        for m in methods
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    ax.set_xlabel("log10(nocc)")
    ax.set_ylabel("log10(nvirt)")
    ax.set_zlabel("Actual log10(time(s))")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Scatter actual timing vs system size for common B2PLYP-D3/wB97X-D/wB97X-V systems."
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
        default=here / "overlap_dfs" / "timing_eval" / "plots" / "size_scatter_3methods",
        help="Output directory for scatter plot and point table",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_frames = load_method_frames_with_keys(input_dir, TARGET_METHODS)
    points = build_common_points(method_frames, TARGET_METHODS)
    if points.empty:
        raise ValueError("No common systems with positive timings found for the three target methods")

    points_csv = out_dir / "points_actual_common_three_methods.csv"
    plot_path = out_dir / "actual_vs_size_common_three_methods.png"
    plot_3d_path = out_dir / "actual_logtime_vs_log_nocc_log_nvirt_common_three_methods.png"
    points.to_csv(points_csv, index=False)

    plot_points(
        points,
        TARGET_METHODS,
        plot_path,
        title="Actual log10(time) vs system size for common systems (B2PLYP-D3, wB97X-D, wB97X-V)",
    )
    plot_points_3d(
        points,
        TARGET_METHODS,
        plot_3d_path,
        title="Actual log10(time) vs log10(nocc), log10(nvirt) for common systems",
    )

    n_common = points["system_key"].nunique()
    print(f"Common systems: {n_common}")
    print(f"Total plotted rows: {len(points)}")
    print(f"Wrote: {points_csv}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {plot_3d_path}")


if __name__ == "__main__":
    main()
