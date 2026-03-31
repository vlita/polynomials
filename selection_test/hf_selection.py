#!/home/vlita3/miniconda3/envs/p4_qcml/bin/python
"""
HF selection test: use memory polynomial to select timing polynomial.

Workflow:
1) Load HF timing df from unfiltered_method_dfs/HF.pkl
2) Load HF memory df from memory/mem_dfs/HF.pkl
3) Merge on qcfractal id
4) Train memory polynomial (log10(GiB) target) on in-core rows
5) Predict memory for all rows and classify predicted_df_type
6) Train timing polynomial separately for predicted_df_type mem vs disk
7) Plot each subset and a combined plot using the assigned polynomial

Outputs saved to selection_test/plots/.
"""

import inspect
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
COLOR_BY_POLY = 0   # color final combined plot by predicted polynomial type
COLOR_BY_ERROR = 0  # color final combined plot by error_type
COLOR_BY_BASIS = 1  # color final combined plot by basis set

# ---------------------------------------------------------------------------
# Polynomial definitions
# ---------------------------------------------------------------------------

# Memory polynomial (from memory/p2_mem.py)
mem_polynomials = {
    "mem_test": lambda p, n_occ, n_virt, nbf_aux: (
        p[0] + p[1] * (n_occ + n_virt) ** 2 + p[2] * nbf_aux ** 2
        + p[3] * nbf_aux * (n_occ + n_virt) ** 2
    ),
}

MEM_POLY = "mem_test"
MEM_CONFIG = {"n_params": 4, "bounds": (0.0, None)}

# Timing polynomial (from p2_user.py)
time_polynomials = {
    "HF_test4": lambda p, n_occ, n_virt, nbf_aux: (
        p[0] + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
        + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
    ),
    "MP2_test5": lambda p, n_occ, n_virt, nbf_aux: (
        p[0] + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
        + p[2] * (n_occ * n_virt * (n_occ + n_virt) * nbf_aux)
        + p[3] * (n_occ ** 2 * n_virt ** 2 * nbf_aux)
    ),
}

TIME_CONFIGS = {
    "HF": {"poly": "HF_test4", "n_params": 3, "bounds": (0.0, None)},
    "MP2": {"poly": "MP2_test5", "n_params": 4, "bounds": (0.0, None)},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_polynomial_variables(poly_func):
    sig = inspect.signature(poly_func)
    return [param for param in sig.parameters.keys() if param != "p"]


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


def scale_to_gib(raw):
    return raw * 8 / (1024 * 1024 * 1024.0)


def predict_log_mem(params, df):
    poly_func = mem_polynomials[MEM_POLY]
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df, var_names)
    raw = poly_func(params, *var_arrays)
    return np.log10(np.abs(scale_to_gib(raw)))


def predict_log_time(params, df, poly_name):
    poly_func = time_polynomials[poly_name]
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df, var_names)
    raw = poly_func(params, *var_arrays)
    return np.log10(np.abs(raw))


def optimize_mem(df_train):
    n_params = MEM_CONFIG["n_params"]
    bounds = [MEM_CONFIG["bounds"]] * n_params
    params0 = [1.0] * n_params

    def rmse(params):
        pred_log = predict_log_mem(params, df_train)
        return root_mean_squared_error(df_train["log(mem)"], pred_log)

    print(f"Memory polynomial: {MEM_POLY}, n_params: {n_params}")
    print(f"Variables: {get_polynomial_variables(mem_polynomials[MEM_POLY])}")

    ret = minimize(rmse, x0=params0, method="L-BFGS-B", bounds=bounds)
    return ret.x


def optimize_time(df_train, poly_name, config):
    n_params = config["n_params"]
    bounds = [config["bounds"]] * n_params
    params0 = [1.0] * n_params

    def rmse(params):
        pred_log = predict_log_time(params, df_train, poly_name)
        return root_mean_squared_error(df_train["log(time(s))"], pred_log)

    print(f"Timing polynomial: {poly_name}, n_params: {n_params}")
    print(f"Variables: {get_polynomial_variables(time_polynomials[poly_name])}")

    ret = minimize(rmse, x0=params0, method="L-BFGS-B", bounds=bounds)
    return ret.x


def get_basis_colors(basis_sets):
    n = len(basis_sets)
    cmap = plt.cm.tab10 if n <= 10 else (plt.cm.tab20 if n <= 20 else plt.cm.viridis)
    return {b: cmap(i / max(n - 1, 1)) for i, b in enumerate(sorted(basis_sets))}


def plot_train_test(method, label, df_train, df_test, y_train, y_test,
                    y_train_pred, y_test_pred, train_rmse, test_rmse,
                    train_r2, test_r2, train_mae, test_mae):
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, x, y, title, color, rmse, r2, mae, n in [
        (ax_train, y_train, y_train_pred, "Training", "steelblue",
         train_rmse, train_r2, train_mae, len(df_train)),
        (ax_test,  y_test,  y_test_pred,  "Test",     "orange",
         test_rmse,  test_r2,  test_mae,  len(df_test)),
    ]:
        ax.scatter(x, y, alpha=0.6, color=color, s=50)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], "r--", lw=2)
        ax.set_xlabel("Actual log10(time(s))", fontsize=12)
        ax.set_ylabel("Predicted log10(time(s))", fontsize=12)
        ax.set_title(
            f"{method} {label} {title}\n"
            f"n={n}, RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{method}_{label}_subset.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_combined(method, df, y, y_pred):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if COLOR_BY_POLY:
        color_map = {"mem": "steelblue", "disk": "orange"}
        labels_map = {"mem": "mem poly", "disk": "disk poly"}
        for key in ["mem", "disk"]:
            mask = df["predicted_df_type"] == key
            if mask.any():
                ax.scatter(y[mask], y_pred[mask], s=50, alpha=0.6,
                           color=color_map[key], label=labels_map[key])
    elif COLOR_BY_ERROR:
        err_labels = np.where(pd.isna(df["error_type"]), "correct", df["error_type"])
        color_map = {
            "correct": "gray",
            "false_positive": "orange",
            "false_negative": "blue",
        }
        labels_map = {
            "correct": "Correct",
            "false_positive": "False positive",
            "false_negative": "False negative",
        }
        for key in ["correct", "false_positive", "false_negative"]:
            mask = err_labels == key
            if mask.any():
                ax.scatter(y[mask], y_pred[mask], s=50, alpha=0.6,
                           color=color_map[key], label=labels_map[key])
    elif COLOR_BY_BASIS:
        basis_colors = get_basis_colors(df["basis"].dropna().unique())
        for basis in sorted(set(df["basis"].dropna().values)):
            mask = df["basis"] == basis
            ax.scatter(y[mask], y_pred[mask], s=50, alpha=0.6,
                       color=basis_colors[basis], label=basis)
    else:
        ax.scatter(y, y_pred, s=50, alpha=0.6, color="steelblue")

    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax.set_xlabel("Actual log10(time(s))", fontsize=12)
    ax.set_ylabel("Predicted log10(time(s))", fontsize=12)
    ax.set_title(f"{method} combined timing fit", fontsize=13)
    ax.grid(True, alpha=0.3)
    if COLOR_BY_POLY or COLOR_BY_ERROR or COLOR_BY_BASIS:
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{method}_combined.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def train_time_subset(df_subset, method, label, poly_name, config):
    df_pool = df_subset[df_subset["time(s)"].notna() & (df_subset["time(s)"] > 6)].copy()
    if len(df_pool) < 10:
        print(f"  {label}: insufficient data (< 10 points) — skipping.")
        return None

    df_train, df_test = train_test_split(df_pool, test_size=0.10, random_state=21)
    print(f"  {label}: Train {len(df_train)} Test {len(df_test)}")

    params = optimize_time(df_train, poly_name, config)
    print(f"  {label} params: {params}")
    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values
    y_train_pred = predict_log_time(params, df_train, poly_name)
    y_test_pred = predict_log_time(params, df_test, poly_name)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))

    print(f"  {label} — Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"  {label} — Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}")

    plot_train_test(
        method, label, df_train, df_test,
        y_train, y_test, y_train_pred, y_test_pred,
        train_rmse, test_rmse, train_r2, test_r2, train_mae, test_mae,
    )

    return {
        "params": params,
        "df_train": df_train,
        "df_test": df_test,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


def run_method(method):
    timing_pkl = os.path.join(ROOT, "unfiltered_method_dfs", f"{method}.pkl")
    mem_pkl = os.path.join(ROOT, "memory", "mem_dfs", f"{method}.pkl")

    print(f"Loading {method} timing and memory data...")
    df_time = pd.read_pickle(timing_pkl)
    df_mem = pd.read_pickle(mem_pkl)

    keep_cols = ["qcfractal id", "ao_memory_GiB", "scaled set mem (GiB)"]
    df = pd.merge(df_time, df_mem[keep_cols], on="qcfractal id", how="inner")

    # Rename for modeling
    df = df.rename(columns={
        "n_occupied": "nocc",
        "n_virtual": "nvirt",
        "basis_set": "basis",
        "user_time (parse)": "time(s)",
    })
    df["nbf"] = df["nocc"] + df["nvirt"]

    # Actual df_type based on ao_memory_GiB
    df["df_type"] = np.where(df["ao_memory_GiB"].isna(), "disk", "mem")

    # Train memory polynomial on in-core rows
    df_mem_train = df[df["ao_memory_GiB"].notna()].copy()
    df_mem_train["log(mem)"] = np.log10(df_mem_train["ao_memory_GiB"])
    print(f"Memory train rows: {len(df_mem_train)}")
    mem_params = optimize_mem(df_mem_train)
    print(f"Memory params: {mem_params}")

    # Predict memory for all rows and classify
    df["pred_log_mem"] = predict_log_mem(mem_params, df)
    pred_gib = 10 ** df["pred_log_mem"]
    df["predicted_df_type"] = np.where(
        pred_gib > df["scaled set mem (GiB)"], "disk", "mem"
    )

    # error_type based on predicted vs actual
    df["error_type"] = pd.Series([pd.NA] * len(df), dtype="object")
    wrong_mask = df["predicted_df_type"] != df["df_type"]
    df.loc[wrong_mask, "error_type"] = np.where(
        (df.loc[wrong_mask, "predicted_df_type"] == "disk") &
        (df.loc[wrong_mask, "df_type"] == "mem"),
        "false_positive",
        "false_negative",
    )

    # Timing targets
    df["log(time(s))"] = np.log10(df["time(s)"])

    # Partition by predicted_df_type
    df_mem_pred = df[df["predicted_df_type"] == "mem"].copy()
    df_disk_pred = df[df["predicted_df_type"] == "disk"].copy()
    print(f"Predicted mem rows: {len(df_mem_pred)}")
    print(f"Predicted disk rows: {len(df_disk_pred)}")

    # Train timing polynomials for each subset
    time_cfg = TIME_CONFIGS[method]
    print("\nFitting timing polynomial for predicted mem subset")
    mem_fit = train_time_subset(df_mem_pred, method, "pred_mem",
                                time_cfg["poly"], time_cfg)
    print("\nFitting timing polynomial for predicted disk subset")
    disk_fit = train_time_subset(df_disk_pred, method, "pred_disk",
                                 time_cfg["poly"], time_cfg)

    if mem_fit is None or disk_fit is None:
        print("One of the subsets did not have enough data to fit. Skipping combined plot.")
        return

    # Combined prediction
    df_valid = df[df["time(s)"].notna()].copy()
    pred_combined = pd.Series(index=df_valid.index, dtype=float)
    mask_mem = df_valid["predicted_df_type"] == "mem"
    mask_disk = df_valid["predicted_df_type"] == "disk"
    if mask_mem.any():
        pred_combined.loc[mask_mem] = predict_log_time(mem_fit["params"], df_valid[mask_mem], time_cfg["poly"])
    if mask_disk.any():
        pred_combined.loc[mask_disk] = predict_log_time(disk_fit["params"], df_valid[mask_disk], time_cfg["poly"])

    plot_combined(method, df_valid, df_valid["log(time(s))"].values, pred_combined.values)


def main():
    for method in ["HF", "MP2"]:
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        run_method(method)


if __name__ == "__main__":
    main()
