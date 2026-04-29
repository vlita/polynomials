#!/usr/bin/env python

import inspect
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.optimize import minimize
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "filtered_method_dfs"
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots_latex")
BLOCK_PLOTS_DIR = os.path.join(PLOTS_DIR, "split_blocks")
LATEX_FORMAT = "png"
LATEX_TABLE_NAME = "time_fit_template.tex"
INFERENCE_DF_NAME = "time_fit_inference_df.pkl"
INFERENCE_CSV_NAME = "time_fit_inference_df.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BLOCK_PLOTS_DIR, exist_ok=True)


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 26,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    )


time_polynomials = {
    "HF": {
        "poly": lambda p, n_occ, n_virt, nbf_aux: (
            p[0]
            # + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)
            # + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux**2) ** 0.65
            # + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux) ** 0.80
            # + p[1] * (nbf_aux * (n_occ + n_virt) ** p[5])                # DFH: AO Construction
            + p[1] * ((n_occ + n_virt) ** p[4] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[2] * ((n_occ + n_virt) ** p[4] * nbf_aux)                # J
            + p[3] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
        ),
        "coeffs": {
            "n_params": 5,
            "guess": [
                3.20909883e01, 
                3.81966012e-08,
                3.81966012e-08,
                5.67912758e-11, # sweeping from e-0.8 to e-0.11 and tuning tight_delta within +-1 OM
                1.50
                ],
            "bounds": [
                (1.0, 100.0), 
                (1.0e-14, 1.0e-6), 
                (1.0e-14, 1.0e-6),
                (1.0e-14, 1.0e-6),  
                (1.0, 2.0),
                ],
            "coarse_method": "Powell",
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-7,
                1.0e-12,
                0.5,
            ],
        },
    },
    "PBE-D3": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[4]) # RV: Form V
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)
            + p[3] * ((n_occ + n_virt) ** p[5] * nbf_aux ** 2) # DFH: AO-Met. Contraction
        ),
        "coeffs": {
            "n_params": 6,
            "guess": [
                10.00005130e+00,
                5.67912758e-09,
                3.81966012e-09,
                3.81966012e-09,
                1.00005516,
                1.00049297,
                ],
            "bounds": [
                (1.0, 100),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.5, 2.0),
                (1.0, 2.0),
            ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400,
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": False,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-7,
                1.0e-7,
                0.5,
                0.5,
            ],  
        },
    },
    "wB97X-D": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[5])
            # + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
            + p[2] * ((n_occ + n_virt) ** p[6] * nbf_aux**2)
            + p[3] * ((n_occ + n_virt) ** p[6] * nbf_aux)
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ),
        "coeffs": {
            "n_params": 7,
            "guess": [
                150.20909883e01,
                3.81966012e-10,
                3.81966012e-09,
                3.81966012e-09,
                3.81966012e-09,
                1.50049297,
                1.50049297,
                ],
            "bounds": [
                (0.0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.75, 2.0),
                (1.0, 2.0),
                ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400,
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-7,
                1.0e-7,
                1.0e-7,
                0.5,
                0.5,
            ],
        },
    },
    "wB97X-V": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[6])
            + p[2] * ((n_occ + n_virt) ** p[7] * nbf_aux**2)
            + p[3] * ((n_occ + n_virt) ** p[7] * nbf_aux)
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
            + p[5] * (np_total ** 2)
        ),
        "coeffs": {
            "n_params": 8,
            "guess": [
                125.20909883e01,
                3.81966012e-08,
                3.81966012e-05,
                3.81966012e-08,
                7.67912758e-10,
                8.97912758e-11,
                1.50049297,
                1.50000343,
                ],
            "bounds": [
                (0.0, None),
                # (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.75, 2.0),
                (1.0, 2.0),
                ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400, 
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-7,
                1.0e-6,
                1.0e-11,
                1.0e-12,
                0.5,
                0.5, # fine, doesnt really change that much anyways
            ],
        },
    },
    "MP2": {
        "poly": lambda p, n_occ, n_virt, nbf_aux: (
            p[0]
            + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
            + p[2] * ((n_occ + n_virt) ** p[3] * nbf_aux)
        ),
        "coeffs": {
            "n_params": 4,
            "guess": [
                30.20909883e01,
                # 3.81966012e-05,
                # 3.81966012e-08,
                1.67912758e-10,
                1.97912758e-8,
                1.50049297,
                ],
            "bounds": [
                (0.0, None),
                # (1.0e-15, 1.0e-6),
                # (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0, 2.0),
                ],
            "coarse_method": "Powell",
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                # 1.0e-7,
                # 1.0e-6,
                1.0e-11,
                1.0e-19,
                0.5, # fine, doesnt really change that much anyways
            ],
        },
    },
    "B3LYP-D3": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            # p[0]
            # + p[1] * (np_total * (n_occ + n_virt) ** 1.0)

            # + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** 0.75
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[5])
            + p[2] * ((n_occ + n_virt) ** p[6] * nbf_aux**2)
            + p[3] * ((n_occ + n_virt) ** p[6] * nbf_aux)
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ),
        "coeffs": {
            "n_params": 7,
            "guess": [
                3.20909883e01,
                3.81966012e-09,
                3.81966012e-09,
                3.81966012e-09,
                1.21966012e-10,
                1.50049297,
                1.500093,
                ],
            "bounds": [
                (0.0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.75, 2.0),
                (1.0, 2.0),
                ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400, 
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-10,
                1.0e-7,
                1.0e-7,
                1.0e-11,
                0.5,
                0.5,
            ],
        },
    },
    "B2PLYP-D3": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[3])
            # + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux**2)   # contraction
            # + p[4] * ((n_occ + n_virt) ** 2 * nbf_aux)      # DFMP2 term 
            + p[2] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ),
        "coeffs": {
            "n_params": 4,
            "guess": [
                4.63274504e-02,
                1.01980502e-08,
                1.7050187e-10,
                1.50049297,
                ],
            "bounds": [
                (0.0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.75, 2.0),
                ],
            "coarse_method": "Powell",
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-9,
                1.0e-11,
                0.5,
            ],
        },
    },
    "M05-2X": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux**2)
            + p[3] * ((n_occ + n_virt) ** p[5] * nbf_aux)
            + p[3] * (np_total * (n_occ + n_virt) ** p[6])
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ),
        "coeffs": {
            "n_params": 7,
            "guess": [
                30,
                3.81966012e-09,
                3.81966012e-09,
                1.01980502e-08,
                0.950187e-10,
                1.50049297,
                1.50049297,
                ],
            "bounds": [
                (0.0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0, 2.0),
                (0.75, 2.0),
                ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400, 
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-7,
                1.0e-9,
                1.0e-11,
                0.5,
                0.5,
            ],
        },
    },
    "FNO-CCSD": {
        "poly": lambda p, n_occ, n_virt, nbf_aux: (
            p[0] 
            + p[3] * ((n_occ + n_virt) ** 5) 
            + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
            + p[2] * (n_occ**2 * n_virt**4)
        ),
        "coeffs": {
            "n_params": 4,
            "guess": [
                1,
                5.67912758e-7, 
                0.763534e-11,
                1.23534e-12,                
                ],
            "bounds": [
                (0.0, 50),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                ],
            "coarse_method": "Powell",
            "coarse_options": {
                "maxiter": 400,
                "maxfev": 6000, 
                "xtol": 1e-10, 
                "ftol": 1e-10, 
                "disp": True
                },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta":[
                10,
                1.0e-7,
                1.0e-12,
                1.0e-12,
                ],
        },
    },
    "FNO-CCSD(T)": {
        "poly": lambda p, n_occ, n_virt, nbf_aux: (
            p[0]
            + p[4] * ((n_occ + n_virt) ** 5)
            + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
            + p[2] * (n_occ**2 * n_virt**4)
            + p[3] * (n_occ**3 * n_virt**4)
        ),
        "coeffs": {
            "n_params": 5,
            "guess": [
                3.20909883e01, 
                5.67912758e-7, 
                1.67912758e-12, 
                2.67912758e-12,
                1.23534e-12,
                ],
            "bounds": [
                (0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                ],
            "coarse_method": "Powell",
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-7,
                1.0e-13,
                1.0e-13,
                1.0e-12,
                ],
        },
    },
}


def get_polynomial_variables(poly_func):
    sig = inspect.signature(poly_func)
    return [name for name in sig.parameters if name != "p"]


def extract_variables_from_df(df, var_names):
    mapping = {
        "n_occ": "nocc",
        "n_virt": "nvirt",
        "nbf_aux": "nbf_aux",
        "np_total": "np_total",
        "nbf": "nbf",
    }
    arrs = []
    for var in var_names:
        col = mapping.get(var, var)
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' for variable '{var}'.")
        arrs.append(df[col].values)
    return arrs


def predict_log_time(coeffs, df, poly_func):
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df, var_names)
    raw = poly_func(coeffs, *var_arrays)
    return np.log10(np.clip(np.abs(raw), 1.0e-12, None))


def as_bounds(bounds_cfg, n):
    if bounds_cfg is None:
        return [(None, None)] * n
    if isinstance(bounds_cfg, tuple) and len(bounds_cfg) == 2:
        return [bounds_cfg] * n
    bounds = list(bounds_cfg)
    if len(bounds) != n:
        raise ValueError("Bounds list length must match parameter count.")
    return bounds


def bounds_around(center, delta, absolute_bounds=None):
    c = np.asarray(center, dtype=float)
    d = np.broadcast_to(np.asarray(delta, dtype=float), c.shape)
    bounds = [(v - dv, v + dv) for v, dv in zip(c, d)]
    if absolute_bounds is None:
        return bounds
    out = []
    abs_bounds = as_bounds(absolute_bounds, len(c))
    for (lo, hi), (alo, ahi) in zip(bounds, abs_bounds):
        lo2 = lo if alo is None else max(lo, alo)
        hi2 = hi if ahi is None else min(hi, ahi)
        out.append((lo2, hi2))
    return out


def minimize_with_best(objective, x0, method, bounds, stage="opt", options=None):
    best = {"x": np.asarray(x0, dtype=float).copy(), "loss": np.inf}
    eval_count = {"n": 0}
    iter_count = {"n": 0}

    def wrapped(x):
        eval_count["n"] += 1
        loss = float(objective(x))
        if np.isfinite(loss) and loss < best["loss"]:
            best["loss"] = loss
            best["x"] = np.asarray(x, dtype=float).copy()
        return loss

    def callback(xk):
        iter_count["n"] += 1
        loss_k = float(objective(np.asarray(xk, dtype=float)))
        print(f"    {stage}:{method} iter {iter_count['n']:4d} | RMSE={loss_k:.6f}")

    default_options = {
        "Powell": {"maxiter": 400, "maxfev": 6000, "xtol": 1e-4, "ftol": 1e-4, "disp": True},
        "L-BFGS-B": {"disp": True},
    }
    method_options = options if options is not None else default_options.get(method)
    result = minimize(
        wrapped,
        x0=np.asarray(x0, dtype=float),
        method=method,
        bounds=bounds,
        options=method_options,
        callback=callback,
    )
    print(f"    {stage}:{method} raw scipy result:\n{result}")
    print(f"    {stage}:{method} wrapped objective evaluations: {eval_count['n']}")
    result_fun = float(result.fun)
    if np.isfinite(result_fun) and (best["loss"] + 1e-12 < result_fun):
        print(
            f"Warning: {method} returned a non-best iterate; "
            f"using best evaluated params with RMSE={best['loss']:.6f} instead of {result_fun:.6f}."
        )
        return best["x"], best["loss"], False
    return np.asarray(result.x, dtype=float), result_fun, bool(result.success)


def optimize_method(df_train, cfg):
    y_train = df_train["log(time(s))"].values
    poly_func = cfg["poly"]
    coeff_cfg = cfg["coeffs"]

    n_params = int(coeff_cfg["n_params"])
    coeffs0 = np.asarray(coeff_cfg.get("guess", [1.0] * n_params), dtype=float)
    coeff_bounds = as_bounds(coeff_cfg.get("bounds"), n_params)

    def coeff_obj(coeffs):
        pred = predict_log_time(coeffs, df_train, poly_func)
        return root_mean_squared_error(y_train, pred)

    coeffs_coarse, coarse_loss, _ = minimize_with_best(
        coeff_obj,
        x0=coeffs0,
        method=coeff_cfg.get("coarse_method", "Powell"),
        bounds=coeff_bounds,
        stage="coarse",
        options=coeff_cfg.get("coarse_options"),
    )

    run_tight = coeff_cfg.get("run_tight", True) and (not coeff_cfg.get("skip_tight", False))
    if not run_tight:
        print(f"  coarse coefficient RMSE: {coarse_loss:.6f}")
        print("  tight coefficient step: skipped")
        return coeffs_coarse

    tight_bounds = bounds_around(
        coeffs_coarse,
        coeff_cfg.get("tight_delta", 0.2),
        coeff_cfg.get("bounds"),
    )
    coeffs_tight, tight_loss, _ = minimize_with_best(
        coeff_obj,
        x0=coeffs_coarse,
        method=coeff_cfg.get("tight_method", "L-BFGS-B"),
        bounds=tight_bounds,
        stage="tight",
        options=coeff_cfg.get("tight_options"),
    )

    print(f"  coarse coefficient RMSE: {coarse_loss:.6f}")
    print(f"  tight coefficient RMSE:  {tight_loss:.6f}")
    return coeffs_tight


def save_plot(fig, path_base):
    path = f"{path_base}.{LATEX_FORMAT}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")


def plot_combined(method, y_train, y_test, y_train_pred, y_test_pred, train_mae, test_mae):
    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])

    for ax, y, y_pred, title, mae in [
        (ax_train, y_train, y_train_pred, "- Train", train_mae),
        (ax_test, y_test, y_test_pred, "- Test", test_mae),
    ]:
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)
        ax.scatter(y, y_pred, s=50, alpha=1.0, color="steelblue", edgecolors="none")
        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.text(
            0.95,
            0.05,
            f"MAE = {mae:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.set_title(f"{method} {title}", fontsize=plt.rcParams["axes.labelsize"])

    plt.tight_layout()
    save_plot(fig, os.path.join(PLOTS_DIR, f"{method}_combined"))
    plt.close(fig)


def plot_split_block(rows, split):
    if not rows:
        return

    n = len(rows)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, row in enumerate(rows):
        ax = axes[i]
        if split == "train":
            y = row["y_train"]
            y_pred = row["y_train_pred"]
            mae = row["train_mae"]
        else:
            y = row["y_test"]
            y_pred = row["y_test_pred"]
            mae = row["test_mae"]

        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)
        ax.scatter(y, y_pred, s=50, alpha=1.0, color="steelblue", edgecolors="none")

        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.text(
            0.95,
            0.05,
            f"MAE = {mae:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.set_title(f"{row['method']}", fontsize=plt.rcParams["axes.labelsize"])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_plot(fig, os.path.join(BLOCK_PLOTS_DIR, f"all_methods_{split}"))
    plt.close(fig)


def latex_escape(text):
    return text.replace("_", "\\_")


def format_coefficients(coeffs):
    lines = [f"\\(c_{{{i}}}={v:.3e}\\)" for i, v in enumerate(coeffs)]
    return "\\begin{tabular}{l}" + " \\\\ ".join(lines) + "\\end{tabular}"


def format_polynomial_expression(poly_func):
    def strip_outer_parens(text):
        text = text.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            wraps_all = True
            for i, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(text) - 1:
                        wraps_all = False
                        break
            if wraps_all and depth == 0:
                text = text[1:-1].strip()
            else:
                break
        return text

    src = inspect.getsource(poly_func).strip()
    if "lambda" not in src:
        return ""
    expr = src.split(":", 1)[1].strip()
    if expr.startswith("lambda"):
        expr = expr.split(":", 1)[1].strip()
    expr = re.sub(r"#.*", "", expr)
    expr = " ".join(expr.split())
    expr = re.sub(r",\s*$", "", expr)
    expr = strip_outer_parens(expr)
    expr = re.sub(r"\(n_occ\s*\+\s*n_virt\)", "N_{bf}", expr)
    expr = re.sub(r"n_occ\s*\+\s*n_virt", "N_{bf}", expr)
    expr = re.sub(r"p\[(\d+)\]", r"c_{\1}", expr)
    replacements = {
        "n_occ": "N_{occ}",
        "n_virt": "N_{virt}",
        "nbf_aux": "N_{aux}",
        "np_total": "N_{g}",
        "nbf": "N_{bf}",
    }
    for key, val in replacements.items():
        expr = expr.replace(key, val)
    expr = expr.replace("**", "^")
    expr = re.sub(r"\^\s*(c_\{\d+\}|N_\{[^}]+\}|\d+(?:\.\d+)?)", r"^{\1}", expr)
    expr = expr.replace("*", " \\cdot ")
    expr = expr.replace("+ -", "- ")
    expr = expr.replace(" - ", " + - ")
    parts = [part.strip() for part in expr.split(" + ") if part.strip()]
    if not parts:
        return ""
    lines = []
    for i, part in enumerate(parts):
        if i == 0 or part.startswith("-"):
            lines.append(part)
        else:
            lines.append(f"+ {part}")
    return "$\\begin{array}{l}" + " \\\\ ".join(lines) + "\\end{array}$"


def format_expression_with_raw_coeffs(poly_func, coeffs):
    def strip_outer_parens(text):
        text = text.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            wraps_all = True
            for i, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(text) - 1:
                        wraps_all = False
                        break
            if wraps_all and depth == 0:
                text = text[1:-1].strip()
            else:
                break
        return text

    src = inspect.getsource(poly_func).strip()
    if "lambda" not in src:
        return ""
    expr = src.split(":", 1)[1].strip()
    if expr.startswith("lambda"):
        expr = expr.split(":", 1)[1].strip()
    expr = re.sub(r"#.*", "", expr)
    expr = " ".join(expr.split())
    expr = re.sub(r",\s*$", "", expr)
    expr = strip_outer_parens(expr)

    coeffs = np.asarray(coeffs, dtype=float)

    def repl(match):
        idx = int(match.group(1))
        if idx < 0 or idx >= len(coeffs):
            raise IndexError(f"Coefficient index p[{idx}] out of range for {len(coeffs)} params.")
        return np.format_float_positional(coeffs[idx], precision=17, unique=False, trim="-")

    return re.sub(r"p\[(\d+)\]", repl, expr)


def write_inference_dataframe(rows, output_path, csv_path=None):
    records = []
    for row in rows:
        records.append(
            {
                "method": row["method"],
                "variables": row["variables"],
                "coefficients": row["coefficients"],
                "polynomial_expression_raw": row["polynomial_expression_raw"],
                "prediction_expression": f"log10(abs({row['polynomial_expression_raw']}))",
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_pickle(output_path)
    print(f"Inference dataframe saved: {output_path}")
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Inference CSV saved: {csv_path}")


def write_latex_table(rows, output_path):
    header = (
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{longtable}{|l|c|c|c|c|p{3.2cm}|p{5.0cm}|}\n"
        "\\hline\n"
        "\\textbf{Method} & \\textbf{\\# Train} & \\textbf{\\# Test} "
        "& \\textbf{Train MAE} & \\textbf{Test MAE} & \\textbf{Coefficients} & \\textbf{Functional Form} \\\\ \n"
        "\\hline\n"
        "\\endhead\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{latex_escape(row['method'])} & {row['n_train']} & {row['n_test']} & "
            f"{row['train_mae']:.2f} & {row['test_mae']:.2f} & "
            f"{row['params']} & {row['func']} \\\\ \n\\hline\n"
        )
    lines.append("\\end{longtable}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


def run_method(method, cfg):
    timing_pkl = os.path.join(ROOT, DATA, f"{method}.pkl")
    print(f"Loading {method} timing data...")
    df = pd.read_pickle(timing_pkl).copy()
    df = df.rename(
        columns={
            "n_occupied": "nocc",
            "n_virtual": "nvirt",
            "basis_set": "basis",
            "user_time (parse)": "time(s)",
        }
    )

    mult_col = "multiplicity (parse)" if "multiplicity (parse)" in df.columns else None
    if mult_col is not None:
        mask = df[mult_col].notna() & (df[mult_col] != 1)
        count = int(mask.sum())
        if count:
            df.loc[mask, ["nocc", "nvirt"]] = df.loc[mask, ["nocc", "nvirt"]] * 2
        print(f"Multiplicity adjustment: {count} rows doubled (col: {mult_col}).")

    df["nbf"] = df["nocc"] + df["nvirt"]
    df["log(time(s))"] = np.log10(df["time(s)"])
    df_pool = df[df["time(s)"].notna() & (df["time(s)"] > 6)].copy()
    if len(df_pool) < 10:
        print("Insufficient data (<10 points) — skipping.")
        return None

    df_train, df_test = train_test_split(df_pool, test_size=0.10, random_state=21)
    print(f"Train {len(df_train)} Test {len(df_test)}")

    coeffs = optimize_method(df_train, cfg)
    print(f"  coefficients: {coeffs}")

    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values
    y_train_pred = predict_log_time(coeffs, df_train, cfg["poly"])
    y_test_pred = predict_log_time(coeffs, df_test, cfg["poly"])

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))

    print(f"Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}")

    plot_combined(method, y_train, y_test, y_train_pred, y_test_pred, train_mae, test_mae)

    return {
        "method": method,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "train_mae": train_mae,
        "test_mae": test_mae,
        "params": format_coefficients(coeffs),
        "func": format_polynomial_expression(cfg["poly"]),
        "coefficients": coeffs.tolist(),
        "variables": get_polynomial_variables(cfg["poly"]),
        "polynomial_expression_raw": format_expression_with_raw_coeffs(cfg["poly"], coeffs),
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


def main():
    apply_plot_style()
    rows = []
    for method, cfg in time_polynomials.items():
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        row = run_method(method, cfg)
        if row is not None:
            rows.append(row)
    if rows:
        plot_split_block(rows, split="train")
        plot_split_block(rows, split="test")
        write_latex_table(rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME))
        write_inference_dataframe(
            rows,
            os.path.join(PLOTS_DIR, INFERENCE_DF_NAME),
            csv_path=os.path.join(PLOTS_DIR, INFERENCE_CSV_NAME),
        )


if __name__ == "__main__":
    main()
