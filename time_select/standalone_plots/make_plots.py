#!/usr/bin/env python

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib.transforms import offset_copy

# ── Constants ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BLOCK_PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
LATEX_FORMAT = "png"
SHOW_AXIS_LABELS = False
AXIS_X_LABEL = "log10 time (s)"
AXIS_Y_LABEL = "predicted log10 time (s)"
AXIS_LABEL_SIZE_OFFSET = 4
RESTRICTED_COLOR = "steelblue"
UNRESTRICTED_COLOR = "darkred"
MAE_TEXT_X = 0.96
MAE_TEXT_Y = 0.06
MAE_TEXT_LINE_STEP = 0.055
AUG_SPLIT_METHODS = {"PBE-D3", "M05-2X", "B3LYP-D3"}

os.makedirs(BLOCK_PLOTS_DIR, exist_ok=True)


# ── Plot helpers (copied from const_mult.py) ───────────────────────────────
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


def save_plot(fig, path_base):
    path = f"{path_base}.{LATEX_FORMAT}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")


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

    for i, row in enumerate(rows):
        ax = axes[i]
        groups = collect_groups(row)

        limits = compute_limits(groups)
        if limits is None:
            ax.set_visible(False)
            continue
        min_val, max_val = limits
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)

        for group, color in groups:
            y_vals, y_pred_vals = get_arrays(group)
            if len(y_vals) > 0:
                ax.scatter(
                    y_vals,
                    y_pred_vals,
                    s=50,
                    alpha=1.0,
                    color=color,
                    edgecolors="none",
                )

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


def main():
    apply_plot_style()

    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "*__*_train.pkl")))
    if not train_files:
        print(f"No data files found in {DATA_DIR}. Run const_mult.py first to generate them.")
        return

    method_groups = {}
    for tf in train_files:
        basename = os.path.basename(tf)
        rest = basename.rsplit("_train.pkl", 1)[0]
        method, group = rest.split("__", 1)

        method_groups.setdefault(method, {})[group] = {"train_df": pd.read_pickle(tf)}

        test_path = tf.replace("_train.pkl", "_test.pkl")
        if os.path.exists(test_path):
            method_groups[method][group]["test_df"] = pd.read_pickle(test_path)

    combined_rows = []
    method_order = [
        "HF", "PBE-D3", "wB97X-D", "wB97X-V", "MP2",
        "B3LYP-D3", "B2PLYP-D3", "M05-2X", "FNO-CCSD", "FNO-CCSD(T)",
    ]
    for method in method_order:
        if method not in method_groups:
            continue
        groups = method_groups[method]

        if method in AUG_SPLIT_METHODS:
            non_aug = groups.get("Non-augmented")
            aug = groups.get("Augmented")

            def make_group_dict(g):
                if g is None:
                    return None
                train_df = g.get("train_df")
                test_df = g.get("test_df")
                if train_df is None or test_df is None:
                    return None
                return {
                    "y_train": train_df["log(time(s))"].values,
                    "y_train_pred": train_df["pred_log_time"].values,
                    "y_test": test_df["log(time(s))"].values,
                    "y_test_pred": test_df["pred_log_time"].values,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }

            non_aug_dict = make_group_dict(non_aug)
            aug_dict = make_group_dict(aug)

            if non_aug_dict is None and aug_dict is None:
                continue

            base = non_aug_dict if non_aug_dict is not None else aug_dict
            combined_rows.append(
                {
                    "method": method,
                    "restricted": base,
                    "unrestricted": None,
                    "non_augmented": non_aug_dict,
                    "augmented": aug_dict,
                }
            )
        else:
            group = groups.get("All data")
            if group is None:
                continue
            train_df = group.get("train_df")
            test_df = group.get("test_df")
            if train_df is None or test_df is None:
                continue

            row_dict = {
                "y_train": train_df["log(time(s))"].values,
                "y_train_pred": train_df["pred_log_time"].values,
                "y_test": test_df["log(time(s))"].values,
                "y_test_pred": test_df["pred_log_time"].values,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
            combined_rows.append(
                {
                    "method": method,
                    "restricted": row_dict,
                    "unrestricted": None,
                }
            )

    if combined_rows:
        plot_split_block(combined_rows, split="train")
        plot_split_block(combined_rows, split="test")
    else:
        print("No valid data to plot.")


if __name__ == "__main__":
    main()
