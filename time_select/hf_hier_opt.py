#!/usr/bin/env python
"""
Hierarchical timing polynomial fit (coefficients -> exponents).

Workflow:
1) Load timing df from filtered_method_dfs/<METHOD>.pkl
2) Compute log10(time(s)) target
3) Two-step optimization: coefficients (exponents fixed), then exponents
4) Plot train/test and combined predictions with LaTeX-ready styling

Plots saved to time_select/plots_hier_latex (or plots_hier if LATEX_READY_PLOTS=0).
"""

import inspect
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Options (keep only currently selected ones)
# ---------------------------------------------------------------------------
ADJUST_FOR_MULTIPLICITY = 1
LATEX_READY_PLOTS = 1
LATEX_READY_FORMAT = "png"

DATA = "filtered_method_dfs"

PLOTS_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "plots_hier")
PLOTS_DIR_LATEX = os.path.join(os.path.dirname(__file__), "plots_hier_latex")
PLOTS_DIR = PLOTS_DIR_LATEX if LATEX_READY_PLOTS else PLOTS_DIR_DEFAULT

os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
def apply_plot_style():
    if not LATEX_READY_PLOTS:
        return
    plt.rcParams.update({
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
    })


# ---------------------------------------------------------------------------
# Configuration (coeff bounds + exponent bounds live here)
# ---------------------------------------------------------------------------
METHOD_CONFIGS = {
    "HF": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux: (
                p[0]
                + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)
                + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2) ** e[0]
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux) ** e[1]
                + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[2]
            )
        ),
        "coeffs": {"n_params": 5, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.65, 0.80, 0.80]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "PBE-D3": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0)
                + p[2] * (nbf_aux * (n_occ + n_virt) ** 2) ** e[0]
            )
        ),
        "coeffs": {"n_params": 3, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.90]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "wB97X-D": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0)
                + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2) ** e[0]
                + p[4] * ((n_occ + n_virt) ** 2 * nbf_aux) ** e[1]
                + p[5] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[2]
            )
        ),
        "coeffs": {"n_params": 6, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.65, 0.80, 0.80]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "wB97X-V": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0)
                + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2) ** e[0]
                + p[4] * ((n_occ + n_virt) ** 2 * nbf_aux) ** e[1]
                + p[5] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[2]
                + p[6] * (np_total ** 2) ** e[3]
            )
        ),
        "coeffs": {"n_params": 7, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.65, 0.80, 0.80, 0.70]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "MP2": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux: (
                p[0]
                + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux) ** e[0]
                + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux) ** e[1]
            )
        ),
        "coeffs": {"n_params": 3, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.85, 0.70]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "B3LYP-D3": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0)
                + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2)
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux)
                + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[0]
            )
        ),
        "coeffs": {"n_params": 5, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.75]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "B2PLYP-D3": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0) ** e[0]
                + p[2] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[1]
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2)
                + p[4] * ((n_occ + n_virt) ** 2 * nbf_aux) ** e[2]
            )
        ),
        "coeffs": {"n_params": 5, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.75, 0.80, 0.80]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "M05-2X": {
        "poly": (
            lambda p, e, n_occ, n_virt, nbf_aux, np_total: (
                p[0]
                + p[1] * (np_total * (n_occ + n_virt) ** 1.0)
                + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2)
                + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux)
                + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2) ** e[0]
            )
        ),
        "coeffs": {"n_params": 5, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.75]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "FNO-CCSD": {
        "poly": (
            lambda p, e, n_occ, n_virt: (
                p[0]
                + p[1] * ((n_occ + n_virt) ** 5)
                + p[2] * (n_occ ** 2 * n_virt ** 4) ** e[0]
            )
        ),
        "coeffs": {"n_params": 3, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.6]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
    "FNO-CCSD(T)": {
        "poly": (
            lambda p, e, n_occ, n_virt: (
                p[0]
                + p[1] * ((n_occ + n_virt) ** 5)
                + p[2] * (n_occ ** 2 * n_virt ** 4) ** e[0]
                + p[2] * (n_occ ** 3 * n_virt ** 4) ** e[1]
            )
        ),
        "coeffs": {"n_params": 4, "bounds": (0.0, None)},
        "exponent_opt": {
            "guess": np.array([0.6, 0.7]),
            "delta": 0.5,
            "bounds": None,
            "optimize_mask": None,
        },
        "joint_opt": {
            "enabled": False,
            "coeff_delta": None,
            "coeff_bounds": None,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_polynomial_variables(poly_func):
    sig = inspect.signature(poly_func)
    return [param for param in sig.parameters.keys() if param not in {"p", "e"}]


def extract_variables_from_df(df, var_names):
    var_mapping = {
        "n_occ": "nocc",
        "n_virt": "nvirt",
        "nbf_aux": "nbf_aux",
        "np_total": "np_total",
        "nbf": "nbf",
    }
    extracted = []
    for var in var_names:
        col = var_mapping.get(var, var)
        if col in df.columns:
            extracted.append(df[col].values)
        else:
            raise KeyError(
                f"Variable '{var}' (mapped to '{col}') not found in dataframe. "
                f"Available columns: {df.columns.tolist()}"
            )
    return extracted


def predict_log_time(params, exp_params, df, poly_func):
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df, var_names)
    raw = poly_func(params, exp_params, *var_arrays)
    safe = np.clip(np.abs(raw), 1.0e-12, None)
    return np.log10(safe)


def rmse_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def step1_objective(coeffs, exp_params, df, poly_func, y):
    y_pred = predict_log_time(coeffs, exp_params, df, poly_func)
    return rmse_loss(y, y_pred)


def optimize_coefficients(df, y, poly_func, exp_params, coeffs_guess,
                          coeffs_bounds=None, method="L-BFGS-B", options=None):
    if options is None:
        options = {"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8}

    if coeffs_bounds is None:
        coeffs_bounds = [(None, None)] * len(coeffs_guess)

    result = minimize(
        step1_objective,
        x0=coeffs_guess,
        args=(exp_params, df, poly_func, y),
        method=method,
        bounds=coeffs_bounds,
        options=options,
    )

    if not result.success:
        print(f"[Step 1] Warning: {result.message}")
    else:
        print(f"[Step 1] Converged. Loss = {result.fun:.6e}")

    return result


def step2_objective(exp_params, coeffs_fixed, df, poly_func, y):
    y_pred = predict_log_time(coeffs_fixed, exp_params, df, poly_func)
    return rmse_loss(y, y_pred)


def joint_objective(params, df, poly_func, n_coeffs, y):
    coeffs = params[:n_coeffs]
    exp_params = params[n_coeffs:]
    y_pred = predict_log_time(coeffs, exp_params, df, poly_func)
    return rmse_loss(y, y_pred)


def build_exponent_bounds(exp_init, delta=None, bounds=None, optimize_mask=None):
    n = len(exp_init)
    if bounds is not None:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bounds_list = [bounds] * n
        else:
            bounds_list = list(bounds)
            if len(bounds_list) != n:
                raise ValueError("bounds must be a tuple or list of tuples matching exponents")
    else:
        if delta is None:
            delta = 0.5
        delta_flat = np.broadcast_to(delta, exp_init.shape)
        bounds_list = [(v - d, v + d) for v, d in zip(exp_init, delta_flat)]

    final_bounds = []
    for v, bnd, opt in zip(exp_init, bounds_list, optimize_mask):
        if not opt:
            final_bounds.append((v, v))
        else:
            final_bounds.append(bnd)
    return final_bounds


def optimize_exponents(df, y, coeffs_fixed, poly_func,
                       delta=0.5, bounds=None, optimize_mask=None, guess=None,
                       method="L-BFGS-B", options=None):
    if options is None:
        options = {"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8}

    if guess is None:
        raise ValueError("exponent guess is required")
    exp_init = np.asarray(guess, dtype=float)
    if optimize_mask is None:
        opt_mask = np.ones_like(exp_init, dtype=bool)
    else:
        opt_mask = np.asarray(optimize_mask, dtype=bool)
        if len(opt_mask) != len(exp_init):
            raise ValueError("optimize_mask must match exponent guess length")
    bounds_list = build_exponent_bounds(exp_init, delta=delta,
                                        bounds=bounds, optimize_mask=opt_mask)

    result = minimize(
        step2_objective,
        x0=exp_init,
        args=(coeffs_fixed, df, poly_func, y),
        method=method,
        bounds=bounds_list,
        options=options,
    )

    if not result.success:
        print(f"[Step 2] Warning: {result.message}")
    else:
        print(f"[Step 2] Converged. Loss = {result.fun:.6e}")

    return result


def build_coeff_bounds_from_delta(coeffs, coeff_delta):
    coeffs = np.asarray(coeffs, dtype=float)
    if coeff_delta is None:
        return [(None, None)] * len(coeffs)
    if np.isscalar(coeff_delta):
        deltas = np.full_like(coeffs, float(coeff_delta), dtype=float)
    else:
        deltas = np.asarray(coeff_delta, dtype=float)
        if len(deltas) != len(coeffs):
            raise ValueError("coeff_delta must match number of coefficients")
    return [(c - d, c + d) for c, d in zip(coeffs, deltas)]


def optimize_joint(df, y, poly_func, coeffs_init, exp_init,
                   coeff_bounds=None, exp_bounds=None,
                   method="L-BFGS-B", options=None):
    if options is None:
        options = {"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8}

    n_coeffs = len(coeffs_init)
    params0 = np.concatenate([coeffs_init, exp_init])
    if coeff_bounds is None:
        coeff_bounds = [(None, None)] * n_coeffs
    if exp_bounds is None:
        exp_bounds = [(None, None)] * len(exp_init)
    bounds = list(coeff_bounds) + list(exp_bounds)

    result = minimize(
        joint_objective,
        x0=params0,
        args=(df, poly_func, n_coeffs, y),
        method=method,
        bounds=bounds,
        options=options,
    )

    if not result.success:
        print(f"[Step 2] Warning: {result.message}")
    else:
        print(f"[Step 2] Converged. Loss = {result.fun:.6e}")

    coeffs_opt = result.x[:n_coeffs]
    exp_opt = result.x[n_coeffs:]
    return result, coeffs_opt, exp_opt


def save_plot(fig, path_base):
    if LATEX_READY_PLOTS:
        ext = LATEX_READY_FORMAT
        path = f"{path_base}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        path = f"{path_base}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")


def plot_combined(method, df_train, df_test, y_train, y_test,
                  y_train_pred, y_test_pred, train_rmse, test_rmse,
                  train_r2, test_r2, train_mae, test_mae):
    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])

    for ax, y, y_pred, title, rmse, r2, mae, n in [
        (ax_train, y_train, y_train_pred, "- Train",
         train_rmse, train_r2, train_mae, len(df_train)),
        (ax_test, y_test, y_test_pred, "- Test",
         test_rmse, test_r2, test_mae, len(df_test)),
    ]:
        if LATEX_READY_PLOTS:
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
            ax.set_title(
                f"{method} {title}",
                fontsize=plt.rcParams["axes.labelsize"],
            )
        else:
            ax.scatter(y, y_pred, s=50, alpha=0.6, color="steelblue")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        if not LATEX_READY_PLOTS:
            ax.set_xlabel("Actual log10(time(s))", fontsize=12)
            ax.set_ylabel("Predicted log10(time(s))", fontsize=12)
        if not LATEX_READY_PLOTS:
            ax.set_title(
                f"{method} combined {title}\n"
                f"n={n}, RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}",
                fontsize=13,
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path_base = os.path.join(PLOTS_DIR, f"{method}_combined")
    save_plot(fig, path_base)
    plt.close(fig)


def run_method(method):
    cfg = METHOD_CONFIGS.get(method)
    if cfg is None:
        print(f"No METHOD_CONFIGS entry for method '{method}'. Skipping.")
        return

    timing_pkl = os.path.join(ROOT, DATA, f"{method}.pkl")
    print(f"Loading {method} timing data...")
    df_time = pd.read_pickle(timing_pkl)

    df = df_time.copy()
    df = df.rename(columns={
        "n_occupied": "nocc",
        "n_virtual": "nvirt",
        "basis_set": "basis",
        "user_time (parse)": "time(s)",
    })

    if ADJUST_FOR_MULTIPLICITY:
        mult_col = "multiplicity (parse)" if "multiplicity (parse)" in df.columns else None
        if mult_col is None:
            print("Multiplicity column not found; skipping multiplicity adjustment.")
        else:
            mask = df[mult_col].notna() & (df[mult_col] != 1)
            count = int(mask.sum())
            if count:
                df.loc[mask, ["nocc", "nvirt"]] = df.loc[mask, ["nocc", "nvirt"]] * 2
            print(f"Multiplicity adjustment: {count} rows doubled (col: {mult_col}).")

    df["nbf"] = df["nocc"] + df["nvirt"]
    df["log(time(s))"] = np.log10(df["time(s)"])

    df_pool = df[df["time(s)"].notna() & (df["time(s)"] > 6)].copy()
    if len(df_pool) < 10:
        print("Insufficient data (< 10 points) — skipping.")
        return

    df_train, df_test = train_test_split(df_pool, test_size=0.10, random_state=21)
    print(f"Train {len(df_train)} Test {len(df_test)}")

    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values

    coeff_cfg = cfg["coeffs"]
    n_params = coeff_cfg["n_params"]
    coeffs_guess = np.asarray(coeff_cfg.get("guess", [1.0] * n_params), dtype=float)
    if len(coeffs_guess) != n_params:
        raise ValueError("coeffs guess length must match n_params")

    bounds = coeff_cfg.get("bounds")
    if bounds is None:
        coeffs_bounds = [(None, None)] * n_params
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        coeffs_bounds = [bounds] * n_params
    else:
        coeffs_bounds = list(bounds)

    poly_func = cfg["poly"]
    exp_cfg = cfg["exponent_opt"]

    res1 = optimize_coefficients(
        df_train,
        y_train,
        poly_func,
        exp_cfg["guess"],
        coeffs_guess,
        coeffs_bounds=coeffs_bounds,
    )
    optimal_coeffs = res1.x
    print(f"Optimal coefficients: {optimal_coeffs}")

    joint_cfg = cfg.get("joint_opt", {})
    if joint_cfg.get("enabled"):
        exp_init = np.asarray(exp_cfg.get("guess"), dtype=float)
        exp_bounds = build_exponent_bounds(
            exp_init,
            delta=exp_cfg.get("delta", 0.5),
            bounds=exp_cfg.get("bounds"),
            optimize_mask=exp_cfg.get("optimize_mask"),
        )

        coeff_bounds = joint_cfg.get("coeff_bounds")
        if coeff_bounds is None:
            coeff_bounds = build_coeff_bounds_from_delta(
                optimal_coeffs, joint_cfg.get("coeff_delta")
            )

        res2, optimal_coeffs, optimal_exp_params = optimize_joint(
            df_train,
            y_train,
            poly_func,
            optimal_coeffs,
            exp_init,
            coeff_bounds=coeff_bounds,
            exp_bounds=exp_bounds,
        )
    else:
        res2 = optimize_exponents(
            df_train,
            y_train,
            optimal_coeffs,
            poly_func,
            delta=exp_cfg.get("delta", 0.5),
            bounds=exp_cfg.get("bounds"),
            optimize_mask=exp_cfg.get("optimize_mask"),
            guess=exp_cfg.get("guess"),
        )
        optimal_exp_params = res2.x

    print(f"Optimal coefficients: {optimal_coeffs}")
    print(f"Optimal exponents (e): {optimal_exp_params}")

    y_train_pred = predict_log_time(
        optimal_coeffs, optimal_exp_params, df_train, poly_func
    )
    y_test_pred = predict_log_time(
        optimal_coeffs, optimal_exp_params, df_test, poly_func
    )

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))

    print(f"Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}")

    plot_combined(
        method,
        df_train,
        df_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        train_rmse,
        test_rmse,
        train_r2,
        test_r2,
        train_mae,
        test_mae,
    )


def main():
    apply_plot_style()
    for method in METHOD_CONFIGS:
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        run_method(method)


if __name__ == "__main__":
    main()
