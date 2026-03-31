#!/home/vlita3/miniconda3/envs/p4_qcml/bin/python
"""
Polynomial fitting script for per-method timing dataframes.

Loads each method's dataframe from INPUT_DIR/{method}.pkl (configurable below),
trains a polynomial on log10(user time) for rows with time > 6s, and saves
plots to plots/. No dataframes are written by this script.

To switch between input sources set INPUT_DIR to either:
  "output"                  — raw consolidated dataframes
  "unfiltered_method_dfs"   — dataframes with a pre-existing predicted col
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import inspect
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Switch INPUT_DIR to read from a different source:
#   "output"                — raw consolidated per-method dataframes
#   "unfiltered_method_dfs" — dataframes that may already have a predicted col
INPUT_DIR  = "unfiltered_method_dfs"
PLOTS_DIR = os.path.join("plots", INPUT_DIR)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting flags
# ---------------------------------------------------------------------------
COLOR_BY_BASIS = 0   # 1 to color scatter points by basis set
COLOR_BY_MEMORY = 0  # 1 to color scatter points by memory usage

# ---------------------------------------------------------------------------
# Polynomial definitions
# (copied verbatim from AI4Science_QC/timings/p2_user.py)
# ---------------------------------------------------------------------------
polynomials = {
    "MP2_test5": lambda p, n_occ, n_virt, nbf_aux: p[0] + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux) + p[2] * (n_occ * n_virt * (n_occ + n_virt) * nbf_aux) + p[3] * (n_occ ** 2 * n_virt ** 2 * nbf_aux),
    "HF_test1": lambda p, n_occ, n_virt, nbf_aux: p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[2] * (n_occ * (n_occ + n_virt) * nbf_aux**2) + p[3] * (((n_occ + n_virt)**2 * nbf_aux) + (n_occ * (n_occ + n_virt)**2 * nbf_aux)) + p[4] * (((n_occ + n_virt)**2 * nbf_aux) + ((n_occ + n_virt)**2 * nbf_aux) + nbf_aux**2) + p[5] * (nbf_aux**2 + nbf_aux**3),
    "FNO-CCSD_1": lambda p, n_occ, n_virt: p[0] + p[1] * (n_occ ** 2 * n_virt ** 4) + p[2] * (n_occ ** 3 * n_virt ** 3),
    "FNO-CCSD(T)_1": lambda p, n_occ, n_virt: p[0] + p[1] * (n_occ ** 2 * n_virt ** 4) + p[2] * (n_occ ** 3 * n_virt ** 3) + p[3] * (n_occ ** 3 * n_virt ** 4),
    "CCSD_test_1": lambda p, n_occ, n_virt: p[0] + p[1] * (n_occ ** 1 * n_virt ** 1) + p[2] * (n_occ ** 3 * n_virt ** 2) + p[3] * (n_occ ** 2 * n_virt ** 2) + p[4] * (n_occ ** 2 * n_virt ** 3) + p[5] * (n_occ ** 3 * n_virt ** 3) + p[6] * (n_occ ** 2 * n_virt ** 4) + p[7] * (n_occ ** 4 * n_virt ** 2),
    "PBE-D3_1": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * ((n_occ + n_virt) * np_total) + p[2] * (((n_occ + n_virt)**2 * nbf_aux) + ((n_occ + n_virt)**2 * nbf_aux) + nbf_aux**2) + p[3] * (nbf_aux**2 + nbf_aux**3),
    "B3LYP-D3_1": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * ((n_occ + n_virt) * np_total) + p[2] * (((n_occ + n_virt)**2 * nbf_aux) + ((n_occ + n_virt)**2 * nbf_aux) + nbf_aux**2) + p[3] * (nbf_aux**2 + nbf_aux**3),
    "B2PLYP-D3_1": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * ((n_occ + n_virt) * np_total) + p[2] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux) + p[3] * (n_occ * n_virt * (n_occ + n_virt) * nbf_aux) + p[4] * (n_occ ** 2 * n_virt ** 2 * nbf_aux),
    "M05-2X_1": lambda p, nbf, np_total: p[0] + p[1] * (nbf * np_total) + p[2] * nbf**4,
    "wB97X-V_1": lambda p, nbf, np_total: p[0] + p[1] * (nbf * np_total) + p[2] * nbf**4,
    "wB97X-V_2": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[2] * (nbf_aux * (n_occ + n_virt)**2) + p[3] * ((n_occ + n_virt) * np_total) + p[4] * ((0.5 * np_total) ** 2),

    "wB97X-D_1": lambda p, nbf, np_total: p[0] + p[1] * (nbf * np_total) + p[2] * nbf**4,

    # new polys from jacket
    "PBE-D3_2": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * ((n_occ + n_virt) * np_total) + p[2] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[3] * (nbf_aux * (n_occ + n_virt)**2),
    "B3LYP-D3_2": lambda p, n_occ, n_virt, nbf_aux, np_total: p[0] + p[1] * ((n_occ + n_virt) * np_total) + p[2] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[3] * (nbf_aux * (n_occ + n_virt)**2),
    "HF_test4": lambda p, n_occ, n_virt, nbf_aux: p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[2] * (nbf_aux * (n_occ + n_virt)**2),
    "HF_test5": lambda p, n_occ, n_virt, nbf_aux: p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux) + p[2] * (nbf_aux * (n_occ + n_virt)**2) + p[3] * ((n_occ + n_virt)**2 * nbf_aux**2),
}

POLYNOMIAL_CONFIG = {
    "MP2_test5":       {"n_params": 4, "bounds": (0.0, None)},
    "HF_test1":        {"n_params": 6, "bounds": (0.0, None)},
    "HF_test4":        {"n_params": 3, "bounds": (0.0, None)},
    "HF_test5":        {"n_params": 4, "bounds": (0.0, None)},
    "FNO-CCSD_1":      {"n_params": 3, "bounds": (0.0, None)},
    "FNO-CCSD(T)_1":   {"n_params": 4, "bounds": (0.0, None)},
    "CCSD_test_1":     {"n_params": 8, "bounds": (0.0, None)},
    "PBE-D3_1":        {"n_params": 4, "bounds": (0.0, None)},
    "PBE-D3_2":        {"n_params": 4, "bounds": (0.0, None)},
    "B3LYP-D3_1":      {"n_params": 4, "bounds": (0.0, None)},
    "B3LYP-D3_2":      {"n_params": 4, "bounds": (0.0, None)},
    "B2PLYP-D3_1":     {"n_params": 5, "bounds": (0.0, None)},
    "M05-2X_1":        {"n_params": 3, "bounds": (0.0, None)},
    "wB97X-V_1":       {"n_params": 3, "bounds": (0.0, None)},
    "wB97X-V_2":       {"n_params": 5, "bounds": (0.0, None)},
    "wB97X-D_1":       {"n_params": 3, "bounds": (0.0, None)},
}

# Methods with no polynomial assigned — skipped during fitting
method_to_polynomial = {
    # "HF":         "HF_test4",
    # "MP2":        "MP2_test5",
    # "B2PLYP-D3":  "B2PLYP-D3_1",
    # "B3LYP-D3":   "B3LYP-D3_2",
    # "PBE-D3":     "PBE-D3_2",
    # "M05-2X":     "M05-2X_1",
    "wB97X-V":    "wB97X-V_2",
    # "wB97X-D":    "wB97X-D_1",
    # "FNO-CCSD":   "CCSD_test_1",
    # "FNO-CCSD(T)":"CCSD_test_1",
    # "CCSD":     not assigned
    # "CCSD(T)":  not assigned
}

ALL_METHODS = [
    # "HF", 
    # "MP2", 
    # "FNO-CCSD", 
    # "FNO-CCSD(T)",
    # "B2PLYP-D3", 
    # "B3LYP-D3", 
    # "M05-2X", 
    # "PBE-D3", 
    # "wB97X-D", 
    "wB97X-V",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_polynomial_variables(polynomial_name):
    poly_func = polynomials[polynomial_name]
    sig = inspect.signature(poly_func)
    return [param for param in sig.parameters.keys() if param != 'p']


def extract_variables_from_df(df, var_names):
    """
    Extract variable arrays from dataframe. Supports internal column names
    (nocc, nvirt, nbf, nbf_aux, np_total) used by the polynomial lambdas.

    Lambda parameter names like 'n_occ'/'n_virt' are mapped to the df
    column names 'nocc'/'nvirt' (after the rename step in process_method).
    """
    var_mapping = {
        'n_occ':    'nocc',
        'n_virt':   'nvirt',
        'nbf_aux':  'nbf_aux',
        'np_total': 'np_total',
        'nbf':      'nbf',
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


def compute_rmse(params, df, polynomial_name):
    poly_func = polynomials[polynomial_name]
    var_names = get_polynomial_variables(polynomial_name)
    var_arrays = extract_variables_from_df(df, var_names)
    pred = poly_func(params, *var_arrays)
    return root_mean_squared_error(
        df["log(time(s))"], np.log10(np.abs(pred))
    )


def predict_log(params, df, polynomial_name):
    poly_func = polynomials[polynomial_name]
    var_names = get_polynomial_variables(polynomial_name)
    var_arrays = extract_variables_from_df(df, var_names)
    return np.log10(np.abs(poly_func(params, *var_arrays)))


def optimize(df_train, polynomial_name):
    config = POLYNOMIAL_CONFIG[polynomial_name]
    n_params = config["n_params"]
    bounds = [config["bounds"]] * n_params
    params0 = [1.0] * n_params

    print(f"Polynomial: {polynomial_name}, n_params: {n_params}")
    print(f"Variables: {get_polynomial_variables(polynomial_name)}")

    ret = minimize(
        compute_rmse,
        args=(df_train, polynomial_name),
        x0=params0,
        method="L-BFGS-B",
        bounds=bounds,
    )
    return ret.x


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def get_basis_colors(basis_sets):
    n = len(basis_sets)
    cmap = plt.cm.tab10 if n <= 10 else (plt.cm.tab20 if n <= 20 else plt.cm.viridis)
    return {b: cmap(i / max(n - 1, 1)) for i, b in enumerate(sorted(basis_sets))}


def plot_with_basis_colors(ax, x, y, basis_values, basis_colors):
    for basis in sorted(set(basis_values)):
        mask = basis_values == basis
        ax.scatter(x[mask], y[mask], c=[basis_colors[basis]], alpha=0.6, label=basis, s=50)


def plot_with_memory_colors(ax, x, y, memory_values):
    scatter = ax.scatter(x, y, c=memory_values, cmap='viridis', alpha=0.6, s=50)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Memory (GB)', fontsize=10)


def make_plot(method, df_train, df_test, y_train, y_test,
              y_train_pred, y_test_pred, train_rmse, test_rmse,
              train_r2, test_r2, train_mae, test_mae, basis_col):

    if COLOR_BY_BASIS and COLOR_BY_MEMORY:
        color_basis = True
        color_memory = False
    else:
        color_basis = bool(COLOR_BY_BASIS and basis_col)
        color_memory = bool(COLOR_BY_MEMORY and 'memory_GB (provenance)' in df_train.columns)

    basis_colors = None
    if color_basis:
        basis_colors = get_basis_colors(df_train[basis_col].dropna().unique())

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, x, y, df_sub, label, color in [
        (ax_train, y_train, y_train_pred, df_train, 'Training', 'steelblue'),
        (ax_test,  y_test,  y_test_pred,  df_test,  'Test',     'orange'),
    ]:
        if color_basis and basis_col in df_sub.columns:
            plot_with_basis_colors(ax, x, y, df_sub[basis_col].values, basis_colors)
        elif color_memory and 'memory_GB (provenance)' in df_sub.columns:
            plot_with_memory_colors(ax, x, y, df_sub['memory_GB (provenance)'].values)
        else:
            ax.scatter(x, y, alpha=0.6, label=label, color=color, s=50)

        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=2)
        ax.set_xlabel('Actual log₁₀(time(s))', fontsize=12)
        ax.set_ylabel('Predicted log₁₀(time(s))', fontsize=12)

        if ax is ax_train:
            rmse, r2, mae, n = train_rmse, train_r2, train_mae, len(df_train)
        else:
            rmse, r2, mae, n = test_rmse, test_r2, test_mae, len(df_test)

        ax.set_title(
            f'{method} {label}\nn={n}, RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}',
            fontsize=13
        )
        if not color_memory or color_basis:
            ax.legend(loc='best', fontsize=9, ncol=2 if basis_colors else 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    suffix = "_colored" if COLOR_BY_BASIS else ("_memory" if COLOR_BY_MEMORY else "")
    path = f"{PLOTS_DIR}/{method}_user{suffix}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_method(method):
    pkl_path = os.path.join(INPUT_DIR, f"{method}.pkl")
    df_full = pd.read_pickle(pkl_path)

    if method not in method_to_polynomial:
        print(f"  No polynomial assigned — skipping (no prediction column added).")
        return

    polynomial = method_to_polynomial[method]

    # Rename columns for internal use
    df_full = df_full.rename(columns={
        'user_time (parse)': 'time(s)',
        'n_occupied':        'nocc',
        'n_virtual':         'nvirt',
        'basis_set':         'basis',
    })
    df_full['nbf'] = df_full['nocc'] + df_full['nvirt']
    df_full['log(time(s))'] = np.log10(df_full['time(s)'])

    # Training subset: rows with valid timing data and time > 6s
    df_train_pool = df_full[
        df_full['time(s)'].notna() & (df_full['time(s)'] > 6)
    ].copy()
    df_train_pool.sort_values('time(s)', inplace=True)

    print(f"  Total rows: {len(df_full)}, training-eligible (time>6s): {len(df_train_pool)}")

    if len(df_train_pool) < 10:
        print(f"  Insufficient data (< 10 points) — skipping fit.")
        return

    df_train, df_test = train_test_split(df_train_pool, test_size=0.10, random_state=21)
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

    # Fit polynomial
    params = optimize(df_train, polynomial)
    print(f"  Optimized params: {params}")

    # Metrics
    y_train      = df_train['log(time(s))'].values
    y_test       = df_test['log(time(s))'].values
    y_train_pred = predict_log(params, df_train, polynomial)
    y_test_pred  = predict_log(params, df_test,  polynomial)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse  = root_mean_squared_error(y_test,  y_test_pred)
    train_r2   = r2_score(y_train, y_train_pred)
    test_r2    = r2_score(y_test,  y_test_pred)
    train_mae  = np.mean(np.abs(y_train - y_train_pred))
    test_mae   = np.mean(np.abs(y_test  - y_test_pred))

    print(f"  Train — RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"  Test  — RMSE: {test_rmse:.6f},  R²: {test_r2:.6f},  MAE: {test_mae:.6f}")

    # Add 'predicted log user time' only to rows with valid timing data (time(s) not NaN)
    valid_mask = df_full['time(s)'].notna()
    df_full.loc[valid_mask, 'predicted log user time'] = predict_log(
        params, df_full[valid_mask], polynomial
    )

    # Plot
    basis_col = 'basis' if 'basis' in df_train.columns else None
    make_plot(
        method, df_train, df_test,
        y_train, y_test, y_train_pred, y_test_pred,
        train_rmse, test_rmse, train_r2, test_r2, train_mae, test_mae,
        basis_col,
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY: {method}")
    print(f"{'='*60}")
    print(f"{'N_train':<10} {'N_test':<10} {'Train RMSE':<14} {'Test RMSE':<14} {'Test R²':<10}")
    print(f"{'-'*60}")
    print(f"{len(df_train):<10} {len(df_test):<10} {train_rmse:<14.6f} {test_rmse:<14.6f} {test_r2:<10.6f}")


def main():
    for method in ALL_METHODS:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")
        process_method(method)


if __name__ == "__main__":
    main()
