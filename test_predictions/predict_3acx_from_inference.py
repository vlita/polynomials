#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from find_common_systems import normalize_basis


VAR_TO_COL = {
    "n_occ": "nocc",
    "n_virt": "nvirt",
    "nbf_aux": "nbf_aux",
    "np_total": "np_total",
    "nbf": "nbf",
}


def safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def make_system_key(formula: str, geom_sig: str, basis: str) -> str:
    return f"{formula}|{geom_sig}|{basis}"


def prepare_method_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "n_occupied": "nocc",
            "n_virtual": "nvirt",
            "basis_set": "basis",
            "user_time (parse)": "time(s)",
        }
    )

    mult_col = "multiplicity (parse)" if "multiplicity (parse)" in out.columns else None
    if mult_col is not None:
        mask = out[mult_col].notna() & (out[mult_col] != 1)
        if mask.any():
            out.loc[mask, ["nocc", "nvirt"]] = out.loc[mask, ["nocc", "nvirt"]] * 2

    out["basis"] = out["basis"].map(normalize_basis)
    out["nbf"] = out["nocc"] + out["nvirt"]
    return out


def eval_raw_expression(expr: str, values: dict[str, float]) -> float:
    safe_globals = {"__builtins__": {}}
    safe_locals = {k: float(v) for k, v in values.items()}
    return float(eval(expr, safe_globals, safe_locals))


def build_predictions(
    systems_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for infer_row in inference_df.itertuples(index=False):
        method = str(infer_row.method)
        variables = list(infer_row.variables)
        expr = str(infer_row.polynomial_expression_raw)

        req_cols = [VAR_TO_COL.get(v, v) for v in variables]

        for sys_row in systems_df.itertuples(index=False):
            match = features_df[
                (features_df["id"] == str(sys_row.id))
                & (features_df["basis"] == str(sys_row.basis))
            ]
            if match.empty:
                continue

            feat = match.iloc[0]
            if any(col not in feat.index for col in req_cols):
                continue

            values = {}
            for var in variables:
                col = VAR_TO_COL.get(var, var)
                val = feat[col]
                if pd.isna(val):
                    values = {}
                    break
                values[var] = float(val)

            if not values:
                continue

            raw = eval_raw_expression(expr, values)
            pred_log = float(np.log10(max(abs(raw), 1e-12)))
            pred_cpu_s = float(10.0**pred_log)

            rows.append(
                {
                    "id": str(sys_row.id),
                    "n_atoms": int(sys_row.n_atoms),
                    "basis": str(sys_row.basis),
                    "method": method,
                    "pred_log10_time_s": pred_log,
                    "pred_time_s": pred_cpu_s,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.sort_values(["method", "n_atoms", "id"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def plot_predictions(pred_df: pd.DataFrame, out_path: Path, basis_label: str) -> None:
    if pred_df.empty:
        raise ValueError("No predictions to plot.")

    systems = pred_df[["id", "n_atoms"]].drop_duplicates().sort_values(["n_atoms", "id"])
    x_labels = systems["id"].tolist()
    x_pos = np.arange(len(x_labels))
    id_to_x = {sid: i for i, sid in enumerate(x_labels)}

    methods = sorted(pred_df["method"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for i, method in enumerate(methods):
        sub = pred_df[pred_df["method"] == method].copy()
        sub["x"] = sub["id"].map(id_to_x)
        sub.sort_values("x", inplace=True)
        ax.plot(
            sub["x"],
            sub["pred_time_s"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=cmap(i % 10),
            label=method,
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_xlabel("System")
    ax.set_ylabel("Predicted CPU time (s)")
    ax.set_title(f"Predicted CPU time for 3ACX systems ({basis_label})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_all_bases(pred_by_basis: dict[str, pd.DataFrame], out_path: Path) -> None:
    if not pred_by_basis:
        raise ValueError("No predictions to plot.")

    basis_labels = sorted(pred_by_basis.keys())
    n_panels = len(basis_labels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    methods = sorted(pd.concat(pred_by_basis.values(), ignore_index=True)["method"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for idx, basis_label in enumerate(basis_labels):
        ax = axes_flat[idx]
        pred_df = pred_by_basis[basis_label]

        systems = pred_df[["id", "n_atoms"]].drop_duplicates().sort_values(["n_atoms", "id"])
        x_labels = systems["id"].tolist()
        x_pos = np.arange(len(x_labels))
        id_to_x = {sid: i for i, sid in enumerate(x_labels)}

        for method in methods:
            sub = pred_df[pred_df["method"] == method].copy()
            if sub.empty:
                continue
            sub["x"] = sub["id"].map(id_to_x)
            sub.sort_values("x", inplace=True)
            ax.plot(
                sub["x"],
                sub["pred_time_s"],
                marker="o",
                linewidth=1.4,
                markersize=3.8,
                color=color_map[method],
                alpha=0.95,
            )

        ax.set_yscale("log")
        ax.set_title(basis_label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Predicted CPU time (s)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("System")

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=color_map[m], marker="o", linewidth=1.6, markersize=4, label=m)
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Predicted CPU time for 3ACX systems by basis", y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run inference on 3acx input systems using saved time-fit polynomials."
    )
    parser.add_argument(
        "--input-pkl",
        type=Path,
        default=Path(__file__).resolve().parent / "3acx_103+_20_in.pkl",
        help="Input systems pickle",
    )
    parser.add_argument(
        "--inference-pkl",
        type=Path,
        default=root / "time_select" / "plots_latex" / "time_fit_inference_df.pkl",
        help="Saved inference dataframe from moo.py",
    )
    parser.add_argument(
        "--timing-vars-pkl",
        type=Path,
        default=root / "ice_timing_vars" / "output" / "3acx" / "3acx_HF.pkl",
        help="3ACX timing-variable dataframe with n_occupied/n_virtual/np_total/nbf_aux",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="all",
        help="Basis to use (or 'all' to run all available bases)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "inference_outputs",
        help="Output directory for predictions and plot",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    in_df_all = pd.read_pickle(args.input_pkl.resolve())
    in_df_all["basis"] = in_df_all["basis"].map(normalize_basis)

    feat_df = pd.read_pickle(args.timing_vars_pkl.resolve())
    feat_df = prepare_method_df(feat_df)
    feat_df = feat_df[["id", "basis", "nocc", "nvirt", "np_total", "nbf_aux", "nbf"]].copy()
    feat_df["id"] = feat_df["id"].astype(str)
    feat_df = feat_df.drop_duplicates(subset=["id", "basis"])

    infer_df = pd.read_pickle(args.inference_pkl.resolve())
    requested_basis = normalize_basis(args.basis)
    if requested_basis.lower() == "all":
        bases = sorted(in_df_all["basis"].dropna().unique())
    else:
        bases = [requested_basis]

    all_preds: list[pd.DataFrame] = []
    plot_paths: list[Path] = []
    pred_by_basis: dict[str, pd.DataFrame] = {}
    for basis_label in bases:
        in_df = in_df_all[in_df_all["basis"] == basis_label].copy()
        in_df = in_df.sort_values(["n_atoms", "id"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
        if len(in_df) != 20:
            raise ValueError(f"Expected 20 systems for basis {basis_label}, found {len(in_df)}.")

        pred_df = build_predictions(in_df, infer_df, feat_df)
        expected = len(in_df) * len(infer_df)
        if len(pred_df) != expected:
            print(f"Warning ({basis_label}): expected {expected} predictions, got {len(pred_df)}")

        all_preds.append(pred_df)
        pred_by_basis[basis_label] = pred_df.copy()

        basis_token = safe_token(basis_label)
        plot_path = out_dir / f"predicted_cpu_time_3acx_20systems_10methods_{basis_token}.png"
        plot_predictions(pred_df, plot_path, basis_label)
        plot_paths.append(plot_path)

    panel_plot_path = out_dir / "predicted_cpu_time_3acx_20systems_10methods_all_bases_panel.png"
    plot_predictions_all_bases(pred_by_basis, panel_plot_path)

    pred_csv = out_dir / "predicted_times_3acx_20systems_10methods.csv"
    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(pred_csv, index=False)

    print(f"Bases used: {', '.join(bases)}")
    print(f"Methods used: {infer_df['method'].nunique()}")
    print(f"Predictions rows: {len(pred_all)}")
    print(f"Wrote: {pred_csv}")
    for p in plot_paths:
        print(f"Wrote: {p}")
    print(f"Wrote: {panel_plot_path}")


if __name__ == "__main__":
    main()
