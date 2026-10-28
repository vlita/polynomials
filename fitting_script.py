#!/usr/bin/env python

import inspect
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib.transforms import offset_copy
from scipy.optimize import minimize
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = "filtered_method_dfs"
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "mult_factor_plots")
BLOCK_PLOTS_DIR = os.path.join(PLOTS_DIR, "split_blocks")
LATEX_FORMAT = "png"
LATEX_TABLE_NAME_RESTRICTED = "time_fit_template_restricted.tex"
LATEX_TABLE_NAME_UNRESTRICTED = "time_fit_template_unrestricted.tex"
LATEX_TABLE_NAME_COMBINED = "time_fit_template_combined.tex"
LATEX_TABLE_NAME_SIMPLE = "time_fit_simple_table.tex"
LATEX_TABLE_NAME_BINNED_RMSE = "time_fit_rmse_binned.tex"
INFERENCE_DF_NAME_RESTRICTED = "time_fit_inference_df_restricted.pkl"
INFERENCE_CSV_NAME_RESTRICTED = "time_fit_inference_df_restricted.csv"
INFERENCE_DF_NAME_UNRESTRICTED = "time_fit_inference_df_unrestricted.pkl"
INFERENCE_CSV_NAME_UNRESTRICTED = "time_fit_inference_df_unrestricted.csv"
SPLIT_AUG_BASIS = True
AUG_SPLIT_METHODS = {"PBE-D3", "M05-2X", "B3LYP-D3", }
SHOW_AXIS_LABELS = False
AXIS_X_LABEL = "log10 time (s)"
AXIS_Y_LABEL = "predicted log10 time (s)"
AXIS_LABEL_SIZE_OFFSET = 4
COMBINED_LABEL_SIZE_OFFSET = 0
RESTRICTED_COLOR = "steelblue"
UNRESTRICTED_COLOR = "darkred"
MAE_TEXT_X = 0.96
MAE_TEXT_Y = 0.06
MAE_TEXT_LINE_STEP = 0.055
PLOT_DATA_DIR = os.path.join(os.path.dirname(__file__), "standalone_plots", "data")

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

restricted_polynomials = {
    "HF": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux: (
            p[0]
            # + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)
            # + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux**2) ** 0.65
            # + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux) ** 0.80
            # + p[1] * (nbf_aux * (n_occ + n_virt) ** p[5])                # DFH: AO Construction
            + p[1] * ((n_occ + n_virt) ** p[4] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[2] * m[0] * ((n_occ + n_virt) ** p[4] * nbf_aux)                # J
            + p[3] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
        ),
        "coeffs": {
            "n_params": 5,
            "guess": [
                3.20909883e01, 
                3.81966012e-08,
                3.81966012e-08,
                5.67912758e-08, # sweeping from e-0.8 to e-0.11 and tuning tight_delta within +-1 OM
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
                1.0e-12,
                0.5,
            ],
            "mult_factor": [
                2.0,
                2.0
            ]
        },
    },
    "PBE-D3": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * (np_total * (n_occ + n_virt) ** p[4]) # RV: Form V
            + p[2] * m[0] * ((n_occ + n_virt) ** p[5] * nbf_aux) 
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
            "mult_factor": [
                1.0,
            ]
        },
    },
    "wB97X-D": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * m[2] * (np_total * (n_occ + n_virt) ** p[5])
            # + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
            + p[2] * ((n_occ + n_virt) ** p[6] * nbf_aux**2)
            + p[3] * m[0] * ((n_occ + n_virt) ** p[6] * nbf_aux)
            + p[4] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
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
            "mult_factor": [
                # 1.0,
                # 1.0,
                # 1.0,
                2.0,
                2.0,
                2.0,
            ]
        },
    },
    "wB97X-V": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * m[2] * (np_total * (n_occ + n_virt) ** p[6])
            + p[2] * ((n_occ + n_virt) ** p[7] * nbf_aux**2)
            + p[3] * m[0] * ((n_occ + n_virt) ** p[7] * nbf_aux)
            + p[4] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
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
            "mult_factor": [
                2.0,
                2.0,
                2.0,
                # 1.0,
                # 1.0,
                # 1.0,
            ],
        },
    },
    "MP2": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux: (
            p[0]
            + p[1] * ((n_occ + n_virt) ** p[5] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[2] * m[0] * ((n_occ + n_virt) ** p[5] * nbf_aux)                # J / contraction / (A|mn)
            + p[3] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[4] * m[2] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy

        ),
        "coeffs": {
            "n_params": 6,
            "guess": [
                10000.000000,
                3.81966012e-10,
                1.67912758e-10,
                1.97912758e-10,
                1.67912758e-12,
                1.50049297,
                ],
            "bounds": [
                (0.0, None),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
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
                100,
                1.0e-11,
                1.0e-11,
                1.0e-11,
                1.0e-12,
                0.5, # fine, doesnt really change that much anyways
            ],
            "mult_factor": [
                2.0,
                2.0,
                3.7,
            ],
        },
    },
    "B3LYP-D3": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            # p[0]
            # + p[1] * (np_total * (n_occ + n_virt) ** 1.0)

            # + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** 0.75
            p[0]
            + p[1] * m[2] * (np_total * (n_occ + n_virt) ** p[5])
            + p[2] * ((n_occ + n_virt) ** p[6] * nbf_aux**2)
            + p[3] * m[0] * ((n_occ + n_virt) ** p[6] * nbf_aux)
            + p[4] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
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
            "mult_factor": [
                2.0,
                2.0,
                2.0,
            ],
        },
    },
    "B2PLYP-D3": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * m[3] * (np_total * (n_occ + n_virt) ** p[6])
            + p[2] * ((n_occ + n_virt) ** p[7] * nbf_aux ** 2)                  # AO-Met. Contraction
            + p[3] * m[0] * ((n_occ + n_virt) ** p[8] * nbf_aux)                # J / contraction / (A|mn)
            + p[4] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[5] * m[2] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy
        ),
        "coeffs": {
            "n_params": 9,
            "guess": [
                0,
                1.01980502e-08,
                1.7050187e-10,
                1.7050187e-10,
                1.7050187e-10,
                1.67912758e-12,
                1.00049297,
                1.50049297,
                1.50049297,
                ],
            "bounds": [
                (0.0, 1.0),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (1.0e-15, 1.0e-6),
                (0.75, 2.0),
                (1.0, 2.0),
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
                1.0e-9,
                1.0e-11,
                1.0e-11,
                1.0e-11,
                1.0e-12,
                0.5,
                0.5,
                0.5,                
            ],
            "mult_factor": [
                2.0,
                2.0,
                3.7,
                2.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
            ],
        },
    },
    "M05-2X": {
        "poly": lambda p, m, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * m[2] * ((n_occ + n_virt) ** p[5] * nbf_aux**2)
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)
            + p[3] * m[0] * (np_total * (n_occ + n_virt) ** p[6])
            + p[4] * m[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ),
        "coeffs": {
            "n_params": 7,
            "guess": [
                30,
                3.81966012e-09,
                3.81966012e-10,
                1.01980502e-11,
                3.950187e-10,
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
            "mult_factor": [
                1.0,
                1.0,
                1.0,
                # 2.0,
                # 2.0,
                # 2.0,
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


def extract_variables_from_df(df, var_names, mult_factor=None, mult_col=None):
    mapping = {
        "n_occ": "nocc",
        "n_virt": "nvirt",
        "nbf_aux": "nbf_aux",
        "np_total": "np_total",
        "nbf": "nbf",
    }
    arrs = []
    for var in var_names:
        if var == "m":
            if mult_factor is None:
                raise ValueError("Missing mult_factor for polynomial with 'm' terms.")
            n_terms = len(mult_factor)
            if n_terms == 0:
                raise ValueError("mult_factor must contain at least one entry for 'm' terms.")
            n_rows = len(df)
            if mult_col is None or mult_col not in df.columns:
                scales = [np.ones(n_rows, dtype=float) for _ in range(n_terms)]
            else:
                mult_values = df[mult_col]
                is_unrestricted = mult_values.notna() & (mult_values != 1)
                scales = [
                    np.where(is_unrestricted, float(mult_factor[i]), 1.0)
                    for i in range(n_terms)
                ]
            arrs.append(np.asarray(scales, dtype=float))
            continue
        col = mapping.get(var, var)
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' for variable '{var}'.")
        arrs.append(df[col].values)
    return arrs


def predict_log_time(coeffs, df, poly_func, mult_factor=None, mult_col=None):
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(
        df,
        var_names,
        mult_factor=mult_factor,
        mult_col=mult_col,
    )
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


def optimize_method(df_train, cfg, mult_col=None):
    y_train = df_train["log(time(s))"].values
    poly_func = cfg["poly"]
    coeff_cfg = cfg["coeffs"]
    mult_factor = coeff_cfg.get("mult_factor")

    n_params = int(coeff_cfg["n_params"])
    coeffs0 = np.asarray(coeff_cfg.get("guess", [1.0] * n_params), dtype=float)
    coeff_bounds = as_bounds(coeff_cfg.get("bounds"), n_params)

    def coeff_obj(coeffs):
        pred = predict_log_time(
            coeffs,
            df_train,
            poly_func,
            mult_factor=mult_factor,
            mult_col=mult_col,
        )
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


def save_plot_data(df_train, df_test, y_train_pred, y_test_pred, method, group_name):
    os.makedirs(PLOT_DATA_DIR, exist_ok=True)
    train_out = df_train.copy()
    train_out["pred_log_time"] = y_train_pred
    test_out = df_test.copy()
    test_out["pred_log_time"] = y_test_pred
    prefix = f"{method}__{group_name}"
    train_out.to_pickle(os.path.join(PLOT_DATA_DIR, f"{prefix}_train.pkl"))
    test_out.to_pickle(os.path.join(PLOT_DATA_DIR, f"{prefix}_test.pkl"))
    print(f"  Plot data saved: {prefix}")


def draw_colored_label(ax, x, y, color, text, fontsize):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=fontsize,
        color="black",
    )
    offset_dot = offset_copy(ax.transAxes, fig=ax.figure, x=-137, y=2, units="points")
    ax.scatter(
        [x],
        [y],
        s=55,
        color=color,
        edgecolors="none",
        transform=offset_dot,
        zorder=5,
    )


def plot_combined(method, restricted, unrestricted):
    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    label_size = plt.rcParams["axes.labelsize"] + COMBINED_LABEL_SIZE_OFFSET

    def group_label(group, fallback):
        if group is None:
            return fallback
        return group.get("label", fallback)

    def collect_groups():
        if restricted is not None and unrestricted is None:
            label = group_label(restricted, "All")
            return [(label, restricted, RESTRICTED_COLOR)]
        return [
            (group_label(restricted, "R"), restricted, RESTRICTED_COLOR),
            (group_label(unrestricted, "U"), unrestricted, UNRESTRICTED_COLOR),
        ]

    def get_arrays(group, split):
        if group is None:
            return np.array([]), np.array([])
        if split == "train":
            return group["y_train"], group["y_train_pred"]
        return group["y_test"], group["y_test_pred"]

    def compute_limits(groups, split):
        values = []
        for _, group, _ in groups:
            y, y_pred = get_arrays(group, split)
            if len(y) > 0:
                values.append(y)
            if len(y_pred) > 0:
                values.append(y_pred)
        if not values:
            return None
        combined = np.concatenate(values)
        return combined.min(), combined.max()

    def format_mae_text(group, label, split):
        if group is None:
            return None
        mae = group["train_mae"] if split == "train" else group["test_mae"]
        if not np.isfinite(mae):
            return f"{label} MAE = n/a"
        return f"{label} MAE = {mae:.2f}"

    def pooled_mae(groups, split):
        ys = []
        preds = []
        for _, group, _ in groups:
            y, y_pred = get_arrays(group, split)
            if len(y) > 0 and len(y_pred) > 0:
                ys.append(y)
                preds.append(y_pred)
        if not ys:
            return None
        y_all = np.concatenate(ys)
        pred_all = np.concatenate(preds)
        return float(np.mean(np.abs(y_all - pred_all)))

    groups = collect_groups()

    for ax, split, title in [
        (ax_train, "train", "- Train"),
        (ax_test, "test", "- Test"),
    ]:
        limits = compute_limits(groups, split)
        if limits is None:
            continue
        min_val, max_val = limits
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)

        for label, group, color in groups:
            y_vals, y_pred_vals = get_arrays(group, split)
            if len(y_vals) > 0:
                ax.scatter(
                    y_vals,
                    y_pred_vals,
                    s=50,
                    alpha=1.0,
                    color=color,
                    edgecolors="none",
                    label=label,
                )

        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        mae_lines = []
        for label, group, color in groups:
            text = format_mae_text(group, label, split)
            if text is not None:
                mae_lines.append((label, color, text.split(label, 1)[-1].lstrip()))
        total_mae = pooled_mae(groups, split)
        if total_mae is not None and np.isfinite(total_mae):
            mae_lines.append((None, None, f"Total MAE = {total_mae:.2f}"))

        base_x = MAE_TEXT_X
        base_y = MAE_TEXT_Y
        line_step = MAE_TEXT_LINE_STEP
        for idx, (label, color, text) in enumerate(mae_lines):
            y_pos = base_y + (len(mae_lines) - 1 - idx) * line_step
            if label is None:
                ax.text(
                    base_x,
                    y_pos,
                    text,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=plt.rcParams["xtick.labelsize"],
                )
            else:
                draw_colored_label(
                    ax,
                    base_x,
                    y_pos,
                    color,
                    f"{label} {text}",
                    fontsize=plt.rcParams["xtick.labelsize"],
                )
        ax.set_title(f"{method} {title}", fontsize=plt.rcParams["axes.labelsize"])

    if SHOW_AXIS_LABELS:
        fig.supxlabel(AXIS_X_LABEL, fontsize=label_size)
        fig.supylabel(AXIS_Y_LABEL, fontsize=label_size)
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
    label_size = plt.rcParams["axes.labelsize"] + AXIS_LABEL_SIZE_OFFSET

    def collect_groups(row):
        if row.get("augmented") is not None or row.get("non_augmented") is not None:
            return [
                (row.get("non_augmented"), RESTRICTED_COLOR),
                (row.get("augmented"), RESTRICTED_COLOR),
            ]
        return [
            (row.get("restricted"), RESTRICTED_COLOR),
            (row.get("unrestricted"), UNRESTRICTED_COLOR),
        ]

    for i, row in enumerate(rows):
        ax = axes[i]
        groups = collect_groups(row)

        def get_arrays(group):
            if group is None:
                return np.array([]), np.array([])
            if split == "train":
                return group["y_train"], group["y_train_pred"]
            return group["y_test"], group["y_test_pred"]

        def compute_limits(groups):
            values = []
            for group, _ in groups:
                y, y_pred = get_arrays(group)
                if len(y) > 0:
                    values.append(y)
                if len(y_pred) > 0:
                    values.append(y_pred)
            if not values:
                return None
            combined = np.concatenate(values)
            return combined.min(), combined.max()

        def pooled_mae(groups):
            ys = []
            preds = []
            for group, _ in groups:
                y, y_pred = get_arrays(group)
                if len(y) > 0 and len(y_pred) > 0:
                    ys.append(y)
                    preds.append(y_pred)
            if not ys:
                return None
            y_all = np.concatenate(ys)
            pred_all = np.concatenate(preds)
            return float(np.mean(np.abs(y_all - pred_all)))

        limits = compute_limits(groups)
        if limits is None:
            ax.set_visible(False)
            continue
        min_val, max_val = limits
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)

        for group, color in groups:
            y_vals, y_pred_vals = get_arrays(group)
            if len(y_vals) > 0:
                ax.scatter(y_vals, y_pred_vals, s=50, alpha=1.0, color=color, edgecolors="none")

        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        total_mae = pooled_mae(groups)
        if total_mae is not None and np.isfinite(total_mae):
            ax.text(
                MAE_TEXT_X,
                MAE_TEXT_Y,
                f"MAE = {total_mae:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=plt.rcParams["axes.labelsize"],
            )
        ax.set_title(f"{row['method']}", fontsize=plt.rcParams["axes.labelsize"])

        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if SHOW_AXIS_LABELS:
        fig.supxlabel(AXIS_X_LABEL, fontsize=label_size)
        fig.supylabel(AXIS_Y_LABEL, fontsize=label_size)
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
    expr = re.sub(r"m\[(\d+)\]", r"m_{\1}", expr)
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
    aligned_lines = [f"& {line}" for line in lines]
    return "\\parbox[t]{4.8cm}{\\raggedright$\\begin{aligned}[t]\n" + " \\\\ ".join(aligned_lines) + "\n\\end{aligned}$}"


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


def write_inference_dataframe_combined(rows, output_path, csv_path=None):
    records = []

    def add_record(group, fit_label):
        if group is None:
            return
        records.append(
            {
                "method": group["method"],
                "fit_label": fit_label,
                "variables": group["variables"],
                "coefficients": group["coefficients"],
                "polynomial_expression_raw": group["polynomial_expression_raw"],
                "prediction_expression": f"log10(abs({group['polynomial_expression_raw']}))",
            }
        )

    for row in rows:
        non_aug = row.get("non_augmented")
        aug = row.get("augmented")
        if non_aug is not None or aug is not None:
            add_record(non_aug, "Non-augmented")
            add_record(aug, "Augmented")
            continue

        restricted = row.get("restricted")
        add_record(restricted, restricted.get("label", "All data") if restricted else "All data")

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


def write_binned_normalized_rmse_table(rows, output_path):
    def pooled_arrays(row):
        y_train = []
        y_train_pred = []
        y_test = []
        y_test_pred = []
        n_train = 0
        n_test = 0
        for key in ("restricted", "unrestricted"):
            group = row.get(key)
            if group is None:
                continue
            n_train += int(group.get("n_train", 0))
            n_test += int(group.get("n_test", 0))
            if len(group.get("y_train", [])):
                y_train.append(group["y_train"])
                y_train_pred.append(group["y_train_pred"])
            if len(group.get("y_test", [])):
                y_test.append(group["y_test"])
                y_test_pred.append(group["y_test_pred"])
        y_train_all = np.concatenate(y_train) if y_train else np.array([])
        y_train_pred_all = np.concatenate(y_train_pred) if y_train_pred else np.array([])
        y_test_all = np.concatenate(y_test) if y_test else np.array([])
        y_test_pred_all = np.concatenate(y_test_pred) if y_test_pred else np.array([])
        return y_train_all, y_train_pred_all, y_test_all, y_test_pred_all, n_train, n_test

    def rmse(y_true, y_pred):
        if len(y_true) == 0:
            return None
        return float(root_mean_squared_error(y_true, y_pred))

    def rmse_by_bins(y_true, y_pred, bins):
        out = []
        for lo, hi in bins:
            mask = (y_true >= lo) & (y_true < hi)
            if not np.any(mask):
                out.append(None)
            else:
                out.append(float(root_mean_squared_error(y_true[mask], y_pred[mask])))
        return out

    def fmt_pair(train_val, test_val):
        if train_val is None and test_val is None:
            return "---"
        if train_val is None:
            return f"--- ({test_val:.2f})"
        if test_val is None:
            return f"{train_val:.2f} (---)"
        return f"{train_val:.2f} ({test_val:.2f})"

    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    header = (
        "\\begin{table*}[ht]\n"
        "    \\centering\n"
        "    \\caption{\n"
        "        RMSEs of predicted log10(time) outputted by compute time estimators on the train and test sets with predictions from restricted and unrestricted polynomial classes pooled, binned."
        "\\label{tab:timing-rmses}\n"
        "    }\n"
        "    \\adjustbox{max width=\\textwidth}{%\n"
        "    \\begin{tabular}{lc|c ccccc}\n"
        "        \\hline\n"
        "        Method "
        "& \\# Train (\\# Test) "
        "& \\rotatebox{60}{Overall RMSE} "
        "& \\rotatebox{60}{RMSE ($0$--$10^1$ s)} "
        "& \\rotatebox{60}{RMSE ($10^1$--$10^2$ s)} "
        "& \\rotatebox{60}{RMSE ($10^2$--$10^3$ s)} "
        "& \\rotatebox{60}{RMSE ($10^3$--$10^4$ s)} "
        "& \\rotatebox{60}{RMSE ($10^4$--$10^5$ s)} \\\n"
        "        \\hline\n"
    )

    lines = [header]
    total_train = 0
    total_test = 0
    overall_train_vals = []
    overall_test_vals = []
    binned_train_vals = [[] for _ in bins]
    binned_test_vals = [[] for _ in bins]
    for row in rows:
        method = row["method"]
        y_train, y_train_pred, y_test, y_test_pred, n_train, n_test = pooled_arrays(row)
        total_train += n_train
        total_test += n_test

        overall_train = rmse(y_train, y_train_pred) if len(y_train) else None
        overall_test = rmse(y_test, y_test_pred) if len(y_test) else None
        if overall_train is not None:
            overall_train_vals.append(overall_train)
        if overall_test is not None:
            overall_test_vals.append(overall_test)
        binned_train = rmse_by_bins(y_train, y_train_pred, bins) if len(y_train) else [None] * len(bins)
        binned_test = rmse_by_bins(y_test, y_test_pred, bins) if len(y_test) else [None] * len(bins)
        for idx, val in enumerate(binned_train):
            if val is not None:
                binned_train_vals[idx].append(val)
        for idx, val in enumerate(binned_test):
            if val is not None:
                binned_test_vals[idx].append(val)

        line = (
            f"        {method} & {n_train} ({n_test}) & {fmt_pair(overall_train, overall_test)} "
            f"& {fmt_pair(binned_train[0], binned_test[0])} & {fmt_pair(binned_train[1], binned_test[1])} "
            f"& {fmt_pair(binned_train[2], binned_test[2])} & {fmt_pair(binned_train[3], binned_test[3])} "
            f"& {fmt_pair(binned_train[4], binned_test[4])} \\\n"
        )
        lines.append(line)

    lines.append("        \\hline\n")
    lines.append(f"        Total & {total_train} ({total_test}) &  &  &  &  &  &  \\\n")
    lines.append("        \\hline\n")
    if overall_train_vals or overall_test_vals:
        def avg_std(vals):
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        def fmt_avg(mean_val, std_val):
            if mean_val is None or std_val is None:
                return "---"
            return f"{mean_val:.2f}±{std_val:.2f}"

        mean_train, std_train = avg_std(overall_train_vals)
        mean_test, std_test = avg_std(overall_test_vals)
        avg_cells = [f"{fmt_avg(mean_train, std_train)} ({fmt_avg(mean_test, std_test)})"]
        for idx in range(len(bins)):
            b_mean_train, b_std_train = avg_std(binned_train_vals[idx])
            b_mean_test, b_std_test = avg_std(binned_test_vals[idx])
            avg_cells.append(f"{fmt_avg(b_mean_train, b_std_train)} ({fmt_avg(b_mean_test, b_std_test)})")

        lines.append(
            "        Avg ± SD &  & " + " & ".join(avg_cells) + " \\\n"
        )
        lines.append("        \\hline\n")
    lines.append("    \\end{tabular}\n")
    lines.append("    }\n")
    lines.append("\\end{table*}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


def write_latex_table_combined(rows, output_path):
    header = (
        "\\small\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{longtable}{|p{1.7cm}|p{1.2cm}|p{1.1cm}|p{1.4cm}|p{1.3cm}|p{6.6cm}|p{4.8cm}|}\n"
        "\\hline\n"
        "\\textbf{Method} & \\textbf{\\# Train} & \\textbf{\\# Test} "
        "& \\textbf{Train MAE} & \\textbf{Test MAE} & \\textbf{Parameters} "
        "& \\textbf{Functional Form} \\\\ \n"
        "\\hline\n"
        "\\endhead\n"
    )

    def format_count(value):
        if value is None:
            return "--"
        return str(int(value))

    def format_metric(value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.2f}"

    def format_params(value):
        if not value:
            return "\\textit{n/a}"
        return value

    def format_coeff_lines(coeffs):
        if not coeffs:
            return []
        return [f"\\(c_{{{i}}}={v:.2e}\\)" for i, v in enumerate(coeffs)]

    def format_mult_lines(mult_factor):
        if not mult_factor:
            return []
        return [f"\\(m_{{{i}}}={v:.3g}\\)" for i, v in enumerate(mult_factor)]

    def format_parameters_block(row):
        blocks = []
        non_aug = row.get("non_augmented")
        aug = row.get("augmented")
        if non_aug is not None:
            blocks.append(("Non-augmented", format_coeff_lines(non_aug.get("coefficients"))))
        if aug is not None:
            blocks.append(("Augmented", format_coeff_lines(aug.get("coefficients"))))

        base = row.get("restricted") or row.get("unrestricted")
        if not blocks and base is not None:
            blocks.append(("Coefficients", format_coeff_lines(base.get("coefficients"))))

        if base is not None:
            mult_lines = format_mult_lines(base.get("mult_factor"))
            if mult_lines:
                blocks.append(("Multiplicity", mult_lines))

        if not blocks:
            return "\\textit{n/a}"

        out_lines = []
        for title, lines in blocks:
            out_lines.append(f"\\textbf{{{title}}}")
            if lines:
                out_lines.extend(lines)
            else:
                out_lines.append("\\textit{n/a}")
        return "\\parbox[t]{\\linewidth}{\\raggedright\\noindent " + " \\\\ ".join(out_lines) + "}"

    lines = [header]
    def pooled_groups(row):
        if row.get("non_augmented") is not None or row.get("augmented") is not None:
            return [row.get("non_augmented"), row.get("augmented")]
        return [row.get("restricted"), row.get("unrestricted")]

    def pooled_counts(row):
        groups = pooled_groups(row)
        n_train = sum(int(g.get("n_train", 0)) for g in groups if g is not None)
        n_test = sum(int(g.get("n_test", 0)) for g in groups if g is not None)
        if n_train == 0 and n_test == 0:
            return None, None
        return n_train, n_test

    def pooled_mae(row, split):
        ys = []
        preds = []
        for group in pooled_groups(row):
            if group is None:
                continue
            y = group.get("y_train" if split == "train" else "y_test")
            y_pred = group.get("y_train_pred" if split == "train" else "y_test_pred")
            if y is None or y_pred is None:
                continue
            if len(y) > 0 and len(y_pred) > 0:
                ys.append(y)
                preds.append(y_pred)
        if not ys:
            return None
        y_all = np.concatenate(ys)
        pred_all = np.concatenate(preds)
        return float(np.mean(np.abs(y_all - pred_all)))

    for row in rows:
        restricted = row.get("restricted")
        unrestricted = row.get("unrestricted")
        base = restricted if restricted is not None else unrestricted
        if base is None:
            continue
        pooled_train, pooled_test = pooled_counts(row)
        pooled_train_mae = pooled_mae(row, "train")
        pooled_test_mae = pooled_mae(row, "test")
        lines.append(
            f"{latex_escape(row['method'])} & {format_count(pooled_train)} & {format_count(pooled_test)} & "
            f"{format_metric(pooled_train_mae)} & {format_metric(pooled_test_mae)} & "
            f"{format_parameters_block(row)} & {base.get('func', '')} \\\\ \n\\hline\n"
        )
    lines.append("\\end{longtable}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


def write_simple_latex_table(rows, output_path):
    def pooled_groups(row):
        if row.get("non_augmented") is not None or row.get("augmented") is not None:
            return [row.get("non_augmented"), row.get("augmented")]
        return [row.get("restricted"), row.get("unrestricted")]

    def pooled_arrays(row):
        y_train = []
        y_train_pred = []
        y_test = []
        y_test_pred = []
        n_train = 0
        n_test = 0
        for group in pooled_groups(row):
            if group is None:
                continue
            n_train += int(group.get("n_train", 0))
            n_test += int(group.get("n_test", 0))
            if len(group.get("y_train", [])):
                y_train.append(group["y_train"])
                y_train_pred.append(group["y_train_pred"])
            if len(group.get("y_test", [])):
                y_test.append(group["y_test"])
                y_test_pred.append(group["y_test_pred"])

        y_train_all = np.concatenate(y_train) if y_train else np.array([])
        y_train_pred_all = np.concatenate(y_train_pred) if y_train_pred else np.array([])
        y_test_all = np.concatenate(y_test) if y_test else np.array([])
        y_test_pred_all = np.concatenate(y_test_pred) if y_test_pred else np.array([])
        return y_train_all, y_train_pred_all, y_test_all, y_test_pred_all, n_train, n_test

    def mae(y_true, y_pred):
        if len(y_true) == 0:
            return None
        return float(np.mean(np.abs(y_true - y_pred)))

    def rmse(y_true, y_pred):
        if len(y_true) == 0:
            return None
        return float(root_mean_squared_error(y_true, y_pred))

    def format_metric(value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.2f}"

    lines = [
        "\\begin{table*}\n",
        "    \\centering\n",
        "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} lrrrrrr}\n",
        "        \\hline\n",
        "        Method & N train & N test & Train MAE & Train RMSE & Test MAE & Test RMSE \\\\ \n",
        "        \\hline\n",
    ]

    for row in rows:
        y_train, y_train_pred, y_test, y_test_pred, n_train, n_test = pooled_arrays(row)
        lines.append(
            f"        {latex_escape(row['method'])} & {n_train} & {n_test} & "
            f"{format_metric(mae(y_train, y_train_pred))} & {format_metric(rmse(y_train, y_train_pred))} & "
            f"{format_metric(mae(y_test, y_test_pred))} & {format_metric(rmse(y_test, y_test_pred))} \\\\ \n"
        )

    lines.extend(
        [
            "        \\hline\n",
            "    \\end{tabular*}\n",
            "\\end{table*}\n",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Simple LaTeX table saved: {output_path}")


def write_binned_rmse_table(rows, output_path):
    def pooled_arrays(row):
        y_train = []
        y_train_pred = []
        y_test = []
        y_test_pred = []
        n_train = 0
        n_test = 0
        for key in ("restricted", "unrestricted"):
            group = row.get(key)
            if group is None:
                continue
            n_train += int(group.get("n_train", 0))
            n_test += int(group.get("n_test", 0))
            if len(group.get("y_train", [])):
                y_train.append(group["y_train"])
                y_train_pred.append(group["y_train_pred"])
            if len(group.get("y_test", [])):
                y_test.append(group["y_test"])
                y_test_pred.append(group["y_test_pred"])
        y_train_all = np.concatenate(y_train) if y_train else np.array([])
        y_train_pred_all = np.concatenate(y_train_pred) if y_train_pred else np.array([])
        y_test_all = np.concatenate(y_test) if y_test else np.array([])
        y_test_pred_all = np.concatenate(y_test_pred) if y_test_pred else np.array([])
        return y_train_all, y_train_pred_all, y_test_all, y_test_pred_all, n_train, n_test

    def rmse(y_true, y_pred):
        if len(y_true) == 0:
            return None
        return float(root_mean_squared_error(y_true, y_pred))

    def normalized_rmse(y_true, y_pred, denom):
        if len(y_true) == 0 or denom is None or denom == 0:
            return None
        return float(root_mean_squared_error(y_true, y_pred)) / denom

    def rmse_by_bins(y_true, y_pred, bins, denom):
        out = []
        for lo, hi in bins:
            mask = (y_true >= lo) & (y_true < hi)
            if not np.any(mask):
                out.append(None)
            else:
                out.append(float(root_mean_squared_error(y_true[mask], y_pred[mask])) / denom)
        return out

    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    header = (
        "\\begin{table*}[ht]\n"
        "    \\centering\n"
        "    \\caption{\n"
        "        RMSEs of compute time estimators on the train and test sets with predictions from restricted and unrestricted polynomial classes pooled, binned."
        "\\label{tab:timing-rmses}\n"
        "}"
        "\n"
        "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} lccccccc}\n"
        "        \\hline\n"
        "        Method & \\# Train & \\# Test & Overall RMSE & RMSE ($0-10^1$ s) & RMSE ($10^1-10^2$ s) & RMSE ($10^2-10^3$ s) & RMSE ($10^3-10^4$ s) & RMSE ($10^4-10^5$ s) \\\\n"
        "        \\hline\n"
    )

    lines = [header]
    total_train = 0
    total_test = 0
    for row in rows:
        method = row["method"]
        y_train, y_train_pred, y_test, y_test_pred, n_train, n_test = pooled_arrays(row)
        total_train += n_train
        total_test += n_test
        y_all = np.concatenate([y_train, y_test]) if len(y_train) or len(y_test) else np.array([])
        y_pred_all = np.concatenate([y_train_pred, y_test_pred]) if len(y_train_pred) or len(y_test_pred) else np.array([])
        overall = rmse(y_all, y_pred_all)
        binned = rmse_by_bins(y_all, y_pred_all, bins) if len(y_all) else [None] * len(bins)

        def fmt(val):
            return f"{val:.2f}" if val is not None else ""

        line = (
            f"        {method} & {n_train} & {n_test} & {fmt(overall)} & {fmt(binned[0])} "
            f"& {fmt(binned[1])} & {fmt(binned[2])} & {fmt(binned[3])} & {fmt(binned[4])} \\\\n"
        )
        lines.append(line)

    lines.append("        \\hline\n")
    lines.append(f"        Total & {total_train} & {total_test} &  &  &  &  &  &  \\\\n")
    lines.append("        \\hline\n")
    lines.append("    \\end{tabular*}\n")
    lines.append("\\end{table*}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


def preprocess_dataframe(df):
    df = df.rename(
        columns={
            "n_occupied": "nocc",
            "n_virtual": "nvirt",
            "user_time (parse)": "time(s)",
        }
    )
    df["nbf"] = df["nocc"] + df["nvirt"]
    df["log(time(s))"] = np.log10(df["time(s)"])
    return df[df["time(s)"].notna() & (df["time(s)"] > 6)].copy()


def is_augmented_basis(basis_series):
    basis = basis_series.fillna("").astype(str).str.lower()
    return basis.str.contains("aug")


def split_augmented_basis(df):
    if "basis" not in df.columns:
        return df.iloc[0:0].copy(), df.copy()
    aug_mask = is_augmented_basis(df["basis"])
    return df.loc[aug_mask].copy(), df.loc[~aug_mask].copy()


def train_test(df_pool, min_points=10):
    if len(df_pool) < min_points:
        return None, None
    return train_test_split(df_pool, test_size=0.10, random_state=21)


def fit_group(method, cfg, df_pool, group_name, mult_col=None):
    df_train, df_test = train_test(df_pool)
    if df_train is None:
        print(f"{group_name}: Insufficient data (<10 points) — skipping fit.")
        return None

    print(f"{group_name}: Train {len(df_train)} Test {len(df_test)}")
    coeffs = optimize_method(df_train, cfg, mult_col=mult_col)
    print(f"  {group_name} coefficients: {coeffs}")

    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values
    mult_factor = cfg["coeffs"].get("mult_factor")
    y_train_pred = predict_log_time(
        coeffs,
        df_train,
        cfg["poly"],
        mult_factor=mult_factor,
        mult_col=mult_col,
    )
    y_test_pred = predict_log_time(
        coeffs,
        df_test,
        cfg["poly"],
        mult_factor=mult_factor,
        mult_col=mult_col,
    )

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))

    print(f"{group_name} Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"{group_name} Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}")

    save_plot_data(df_train, df_test, y_train_pred, y_test_pred, method, group_name)

    return {
        "method": method,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "label": group_name,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "params": format_coefficients(coeffs),
        "func": format_polynomial_expression(cfg["poly"]),
        "coefficients": coeffs.tolist(),
        "mult_factor": mult_factor,
        "variables": get_polynomial_variables(cfg["poly"]),
        "polynomial_expression_raw": format_expression_with_raw_coeffs(cfg["poly"], coeffs),
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


def plot_with_predictions(df_pool, coeffs, poly_func, mult_factor=None, mult_col=None):
    if df_pool is None or len(df_pool) == 0:
        return np.array([]), np.array([])
    y = df_pool["log(time(s))"].values
    y_pred = predict_log_time(
        coeffs,
        df_pool,
        poly_func,
        mult_factor=mult_factor,
        mult_col=mult_col,
    )
    return y, y_pred


def run_method(method, cfg):
    timing_pkl = os.path.join(ROOT, DATA, f"{method}.pkl")
    print(f"Loading {method} timing data...")
    df = pd.read_pickle(timing_pkl).copy()
    mult_col = "multiplicity" if "multiplicity" in df.columns else None

    if method == "PBE-D3" and mult_col is not None:
        df = df.loc[df[mult_col].isna() | (df[mult_col] == 1)].copy()

    if method == "wB97X-D" and mult_col is not None:
        df = df.loc[df[mult_col].isna() | (df[mult_col] == 1)].copy()

    if method == "wB97X-V" and mult_col is not None:
        df = df.loc[df[mult_col].isna() | (df[mult_col] == 1)].copy()

    if method == "M05-2X" and mult_col is not None:
        df = df.loc[df[mult_col].isna() | (df[mult_col] == 1)].copy()

    df_pool = preprocess_dataframe(df)

    if SPLIT_AUG_BASIS and method in AUG_SPLIT_METHODS:
        df_aug, df_non_aug = split_augmented_basis(df_pool)
        if len(df_aug) < 10 and len(df_non_aug) < 10:
            print("All data: Insufficient data (<10 points) — skipping.")
            return None

        aug_row = None
        non_aug_row = None
        if len(df_aug) >= 10:
            aug_row = fit_group(method, cfg, df_aug, "Augmented", mult_col=mult_col)
        else:
            print("Augmented: Insufficient data (<10 points) — skipping fit.")

        if len(df_non_aug) >= 10:
            non_aug_row = fit_group(method, cfg, df_non_aug, "Non-augmented", mult_col=mult_col)
        else:
            print("Non-augmented: Insufficient data (<10 points) — skipping fit.")

        plot_combined(method, non_aug_row, aug_row)

        base_row = non_aug_row if non_aug_row is not None else aug_row
        if base_row is None:
            return None
        return {
            "method": method,
            "restricted": base_row,
            "unrestricted": None,
            "augmented": aug_row,
            "non_augmented": non_aug_row,
        }

    if len(df_pool) < 10:
        print("All data: Insufficient data (<10 points) — skipping.")
        return None

    row = fit_group(method, cfg, df_pool, "All data", mult_col=mult_col)
    if row is None:
        return None

    plot_combined(method, row, None)

    return {
        "method": method,
        "restricted": row,
        "unrestricted": None,
    }


def main():
    apply_plot_style()
    combined_rows = []
    restricted_rows = []
    for method, cfg in restricted_polynomials.items():
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        row = run_method(method, cfg)
        if row is None:
            continue
        combined_rows.append(row)
        if row.get("restricted") is not None:
            restricted_rows.append(row["restricted"])
    if combined_rows:
        plot_split_block(combined_rows, split="train")
        plot_split_block(combined_rows, split="test")
    if restricted_rows:
        write_latex_table(restricted_rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_RESTRICTED))
        write_inference_dataframe(
            restricted_rows,
            os.path.join(PLOTS_DIR, INFERENCE_DF_NAME_RESTRICTED),
            csv_path=os.path.join(PLOTS_DIR, INFERENCE_CSV_NAME_RESTRICTED),
        )
    if combined_rows:
        write_latex_table_combined(combined_rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_COMBINED))
        write_simple_latex_table(combined_rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_SIMPLE))
        write_binned_normalized_rmse_table(
            combined_rows,
            os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_BINNED_RMSE),
        )
        write_inference_dataframe_combined(
            combined_rows,
            os.path.join(PLOTS_DIR, INFERENCE_DF_NAME_UNRESTRICTED),
            csv_path=os.path.join(PLOTS_DIR, INFERENCE_CSV_NAME_UNRESTRICTED),
        )


if __name__ == "__main__":
    main()
