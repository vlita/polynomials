#!/usr/bin/env python3
"""
Evaluate overlap dataframes with non-true vs true timing polynomials.

For each overlap dataframe in overlap_dfs/system_*.pkl:
1) Fit per-method polynomial parameters on unfiltered_method_dfs/{method}.pkl
2) Predict log10(time) for rows in the overlap dataframe
3) Build per-method ranking tables (actual vs predicted)
4) Save two separate bar plots per overlap dataframe:
   - non_true: actual vs non-true predicted
   - true:     actual vs true predicted
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error


def load_time_polynomials() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "time_select" / "hf_selection.py"
    spec = importlib.util.spec_from_file_location("hf_selection_module", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.time_polynomials


time_polynomials = load_time_polynomials()


NON_TRUE_POLY = {
    "HF": "HF_test4",
    "MP2": "MP2_test7",
    "wB97X-V": "wB97X-V_2_le",
    "wB97X-D": "wB97X-V_2_le",
    "B3LYP-D3": "B3LYP-D3_1",
    "B2PLYP-D3": "B2PLYP-D3_2_1",
    "M05-2X": "M05-2X_2",
    "PBE-D3": "PBE-D3_2_1",
    "CCSD": "CCSD_test_1",
    "CCSD(T)": "CCSD_test_1",
    "FNO-CCSD": "CCSD_test_1",
    "FNO-CCSD(T)": "CCSD_test_1",
}

TRUE_POLY = {
    "HF": "HF_true",
    "MP2": "MP2_true",
    "wB97X-V": "wB97X-V_true",
    "wB97X-D": "wB97X-D_true",
    "B3LYP-D3": "B3LYP-D3_true",
    "B2PLYP-D3": "B2PLYP-D3_true",
    "M05-2X": "M05-2X_true",
    "PBE-D3": "PBE-D3_true",
    "CCSD": "FNO-CCSD_true",
    "CCSD(T)": "FNO-CCSD(T)_true",
    "FNO-CCSD": "FNO-CCSD_true",
    "FNO-CCSD(T)": "FNO-CCSD(T)_true",
}

N_PARAMS = {
    "HF_test4": 3,
    "MP2_test7": 6,
    "wB97X-V_2_le": 5,
    "B3LYP-D3_1": 4,
    "B2PLYP-D3_2_1": 7,
    "M05-2X_2": 4,
    "PBE-D3_2_1": 4,
    "CCSD_test_1": 8,
    "HF_true": 3,
    "MP2_true": 6,
    "wB97X-V_true": 5,
    "wB97X-D_true": 4,
    "B3LYP-D3_true": 4,
    "B2PLYP-D3_true": 7,
    "M05-2X_true": 4,
    "PBE-D3_true": 3,
    "FNO-CCSD_true": 8,
    "FNO-CCSD(T)_true": 9,
}

VAR_MAPPING = {
    "n_occ": "nocc",
    "n_virt": "nvirt",
    "nbf_aux": "nbf_aux",
    "np_total": "np_total",
    "nbf": "nbf",
}


def get_polynomial_variables(poly_name: str) -> list[str]:
    poly_func = time_polynomials[poly_name]
    sig = inspect.signature(poly_func)
    return [p for p in sig.parameters.keys() if p != "p"]


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "n_occupied": "nocc",
            "n_virtual": "nvirt",
            "user_time (parse)": "time(s)",
            "basis_set": "basis",
        }
    )

    if "nocc" not in out.columns or "nvirt" not in out.columns:
        raise KeyError("Missing n_occupied/n_virtual columns.")

    mult_col = "multiplicity (parse)" if "multiplicity (parse)" in out.columns else None
    if mult_col is not None:
        mask = out[mult_col].notna() & (out[mult_col] != 1)
        if mask.any():
            out.loc[mask, ["nocc", "nvirt"]] = out.loc[mask, ["nocc", "nvirt"]] * 2

    out["nbf"] = out["nocc"] + out["nvirt"]
    out["log(time(s))"] = np.nan
    pos_mask = out["time(s)"].notna() & (out["time(s)"] > 0)
    out.loc[pos_mask, "log(time(s))"] = np.log10(out.loc[pos_mask, "time(s)"])
    return out


def extract_var_arrays(df: pd.DataFrame, var_names: list[str]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for var in var_names:
        col = VAR_MAPPING.get(var, var)
        if col not in df.columns:
            raise KeyError(f"Required variable '{var}' mapped to '{col}' missing.")
        arrays.append(df[col].to_numpy())
    return arrays


def predict_log_time(params: np.ndarray, df: pd.DataFrame, poly_name: str) -> np.ndarray:
    poly_func = time_polynomials[poly_name]
    var_names = get_polynomial_variables(poly_name)
    var_arrays = extract_var_arrays(df, var_names)
    raw = poly_func(params, *var_arrays)
    safe_abs = np.maximum(np.abs(raw), 1e-300)
    return np.log10(safe_abs)


def fit_params(train_df: pd.DataFrame, poly_name: str) -> np.ndarray:
    n_params = N_PARAMS[poly_name]
    params0 = np.ones(n_params, dtype=float)
    bounds = [(0.0, None)] * n_params

    def objective(params: np.ndarray) -> float:
        pred = predict_log_time(params, train_df, poly_name)
        return float(root_mean_squared_error(train_df["log(time(s))"], pred))

    result = minimize(objective, x0=params0, method="L-BFGS-B", bounds=bounds)
    return result.x


def get_poly_name(method: str, fit_type: str) -> str | None:
    if fit_type == "non_true":
        return NON_TRUE_POLY.get(method)
    if fit_type == "true":
        return TRUE_POLY.get(method)
    return None


def fit_all_methods(input_dir: Path, methods: list[str], fit_type: str) -> dict[str, tuple[str, np.ndarray]]:
    fitted: dict[str, tuple[str, np.ndarray]] = {}
    for method in sorted(set(methods)):
        poly_name = get_poly_name(method, fit_type)
        if poly_name is None:
            print(f"[{fit_type}] {method}: no polynomial mapping, skipping")
            continue

        pkl_path = input_dir / f"{method}.pkl"
        if not pkl_path.exists():
            print(f"[{fit_type}] {method}: missing {pkl_path.name}, skipping")
            continue

        print(f"[{fit_type}] fitting {method} with {poly_name} ...", flush=True)
        df = pd.read_pickle(pkl_path)
        df = prepare_df(df)
        train_df = df[df["time(s)"].notna() & (df["time(s)"] > 6)].copy()
        if len(train_df) < 10:
            print(f"[{fit_type}] {method}: insufficient rows ({len(train_df)}), skipping")
            continue

        try:
            params = fit_params(train_df, poly_name)
        except Exception as exc:
            print(f"[{fit_type}] {method}: fit failed: {exc}")
            continue

        fitted[method] = (poly_name, params)
    return fitted


def rank_overlap_df(
    overlap_df: pd.DataFrame,
    fitted: dict[str, tuple[str, np.ndarray]],
    fit_type: str,
) -> pd.DataFrame:
    df = prepare_df(overlap_df)
    if "source_method" not in df.columns:
        raise KeyError("Overlap dataframe is missing 'source_method'.")

    rows: list[dict[str, Any]] = []
    for method, grp in df.groupby("source_method", sort=True):
        if method not in fitted:
            continue

        poly_name, params = fitted[method]
        use = grp[grp["time(s)"] > 0].copy()
        if use.empty:
            continue

        try:
            pred_log = predict_log_time(params, use, poly_name)
        except Exception:
            continue

        actual_log = use["log(time(s))"].to_numpy()
        rows.append(
            {
                "method": method,
                "fit_type": fit_type,
                "poly_name": poly_name,
                "n_rows": len(use),
                "actual_log_time": float(np.mean(actual_log)),
                "predicted_log_time": float(np.mean(pred_log)),
                "mae_log_time": float(np.mean(np.abs(actual_log - pred_log))),
            }
        )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked

    ranked.sort_values("actual_log_time", ascending=False, inplace=True)
    ranked.reset_index(drop=True, inplace=True)
    ranked["actual_rank"] = ranked["actual_log_time"].rank(ascending=False, method="dense").astype(int)
    ranked["predicted_rank"] = ranked["predicted_log_time"].rank(ascending=False, method="dense").astype(int)
    return ranked


def plot_ranked(df_ranked: pd.DataFrame, out_path: Path, title: str) -> None:
    if df_ranked.empty:
        return

    methods = df_ranked["method"].tolist()
    x = np.arange(len(methods))
    width = 0.38

    actual_color = "#2a9d8f"
    pred_color = "#e76f51"

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(methods)), 6.2))
    ax.bar(x - width / 2, df_ranked["actual_log_time"], width, label="actual", color=actual_color)
    ax.bar(x + width / 2, df_ranked["predicted_log_time"], width, label="predicted", color=pred_color)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("log10(time(s))")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    all_vals = np.concatenate(
        [
            df_ranked["actual_log_time"].to_numpy(dtype=float),
            df_ranked["predicted_log_time"].to_numpy(dtype=float),
        ]
    )
    shortest = float(np.min(all_vals))
    longest = float(np.max(all_vals))
    y_min = float(np.ceil(shortest) - 1.0)
    y_max = float(np.floor(longest) + 1.0)
    if y_max <= y_min:
        y_max = y_min + 1.0
    ax.set_ylim(y_min, y_max)

    actual_order = ", ".join(
        df_ranked.sort_values("actual_log_time", ascending=False)["method"].tolist()
    )
    pred_order = ", ".join(
        df_ranked.sort_values("predicted_log_time", ascending=False)["method"].tolist()
    )

    wrapped_pred = textwrap.fill(f"Predicted order: {pred_order}", width=120)
    wrapped_actual = textwrap.fill(f"Actual order: {actual_order}", width=120)

    fig.text(0.01, 0.035, wrapped_pred, ha="left", va="bottom", fontsize=9, color=pred_color)
    fig.text(0.01, 0.005, wrapped_actual, ha="left", va="bottom", fontsize=9, color=actual_color)

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_system_id(path: Path) -> str:
    m = re.search(r"system_(\d+)\.pkl$", path.name)
    return m.group(1) if m else path.stem


def safe_filename_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    token = token.strip("._-")
    return token or "unknown"


def load_formula_map(ranked_csv: Path) -> dict[str, str]:
    if not ranked_csv.exists():
        return {}
    ranked = pd.read_csv(ranked_csv)
    if "system_id" not in ranked.columns or "formula" not in ranked.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in ranked.iterrows():
        out[str(int(row["system_id"]))] = str(row["formula"])
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Rank overlap timings for true/non-true polynomial fits.")
    parser.add_argument(
        "--overlap-dir",
        type=Path,
        default=here / "overlap_dfs",
        help="Directory containing overlap dataframes (system_*.pkl)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=here.parent / "unfiltered_method_dfs",
        help="Directory containing source method dataframes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "overlap_dfs" / "timing_eval",
        help="Output directory for ranking tables and plots",
    )
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=here / "system_overlap_ranked.csv",
        help="Ranked overlap CSV used to map system_id -> molecular formula",
    )
    args = parser.parse_args()

    overlap_dir = args.overlap_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    ranked_csv = args.ranked_csv.resolve()
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    overlap_files = sorted(overlap_dir.glob("system_*.pkl"))
    if not overlap_files:
        raise FileNotFoundError(f"No overlap files found in {overlap_dir}")

    formula_map = load_formula_map(ranked_csv)

    methods_in_overlaps: list[str] = []
    for path in overlap_files:
        df = pd.read_pickle(path)
        if "source_method" in df.columns:
            methods_in_overlaps.extend(df["source_method"].dropna().astype(str).unique().tolist())

    methods = sorted(set(methods_in_overlaps))
    print(f"Methods found in overlap sets: {methods}")

    fitted_non_true = fit_all_methods(input_dir, methods, fit_type="non_true")
    fitted_true = fit_all_methods(input_dir, methods, fit_type="true")

    for idx, overlap_path in enumerate(overlap_files, start=1):
        system_id = parse_system_id(overlap_path)
        print(f"\n[{idx}/{len(overlap_files)}] Processing system_{system_id} ...", flush=True)
        overlap_df = pd.read_pickle(overlap_path)

        non_true_ranked = rank_overlap_df(overlap_df, fitted_non_true, fit_type="non_true")
        true_ranked = rank_overlap_df(overlap_df, fitted_true, fit_type="true")

        non_true_csv = table_dir / f"system_{system_id}_non_true_ranking.csv"
        true_csv = table_dir / f"system_{system_id}_true_ranking.csv"
        non_true_ranked.to_csv(non_true_csv, index=False)
        true_ranked.to_csv(true_csv, index=False)

        formula = formula_map.get(system_id, f"system_{system_id}")
        formula_token = safe_filename_token(formula)

        non_true_plot = plot_dir / f"{formula_token}_non_true.png"
        true_plot = plot_dir / f"{formula_token}_true.png"

        plot_ranked(
            non_true_ranked,
            non_true_plot,
            title=f"{formula} non-true fit: actual vs predicted log time",
        )
        plot_ranked(
            true_ranked,
            true_plot,
            title=f"{formula} true fit: actual vs predicted log time",
        )

        print(f"  wrote {non_true_csv}")
        print(f"  wrote {true_csv}")
        print(f"  wrote {non_true_plot}")
        print(f"  wrote {true_plot}")


if __name__ == "__main__":
    main()
