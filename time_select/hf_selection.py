#!/home/vlita3/miniconda3/envs/p4_qcml/bin/python
"""
HF selection test: split by a hardcoded log(time) threshold.

Workflow:
1) Load HF timing df from unfiltered_method_dfs/HF.pkl
2) Compute log10(time(s))
3) Split rows by a hardcoded log(time) threshold
4) Train timing polynomial separately for each subset
5) Plot each subset and a combined plot using the assigned polynomial

Outputs saved to time_select/plots/.
"""

import inspect
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

# choose a set of systems (5) and put actual v predicted  for timing difference.
# Need to make sure timing difference is captured by timing polynomials, not just MAE looks good.

COLOR_BY_POLY = 1  # color final combined plot by predicted polynomial type
COLOR_BY_BASIS = 0  # color final combined plot by basis set
COLOR_BY_SIZE = 0  # color final combined plot by SZ_VAR
SZ_VAR = "nbf_aux"

ADJUST_FOR_MULTIPLICITY = 1  # double nocc/nvirt when multiplicity != 1
LATEX_READY_PLOTS = 1  # apply rcParams + save vector format
LATEX_READY_FORMAT = "png"
WRITE_LATEX_TABLE = 1
LATEX_TABLE_NAME = "time_fit_table.tex"

COMPOSITE_POLYS = (
    0  # set to 1 to use polynomials that divide timing data into 2 subsets
)

FIT_LINEAR_TIME = 0  # set to 1 to fit in linear time space
SCALE_FEATURES = 0  # set to 1 to scale polynomial input variables
SCALE_TARGET = 0  # set to 1 to scale time(s) target when FIT_LINEAR_TIME=1
SCALER_TYPE = "standard"  # "standard" or "minmax"

DATA = "filtered_method_dfs"

PLOTS_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "plots")
PLOTS_DIR_LATEX = os.path.join(os.path.dirname(__file__), "plots_latex")
PLOTS_DIR = PLOTS_DIR_LATEX if LATEX_READY_PLOTS else PLOTS_DIR_DEFAULT

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------


def apply_plot_style():
    if not LATEX_READY_PLOTS:
        return
    plt.rcParams.update(
        {
            # "font.family": "serif",
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


# ---------------------------------------------------------------------------
# Polynomial definitions
# ---------------------------------------------------------------------------

# Timing polynomial (from p2_user.py)
time_polynomials = {
    # "HF_test4": lambda p, n_occ, n_virt, nbf_aux: (
    #     p[0]
    #     + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux) # K
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2) # J
    # ),
    # "MP2_test6": lambda p, n_occ, n_virt, nbf_aux: (
    #     p[0]
    #     + p[1] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # MP2 transformation: O(N_aux × N_bf^2 × N_occ)
    #     + p[2] * (nbf_aux**2 * n_occ * n_virt)  # MP2 fitting: O(N_occ × N_virt × N_aux^2)
    #     + p[3] * (nbf_aux * n_occ**2 * n_virt**2)  # MP2 form 2e int: O(N_occ^2 × N_virt^2 × N_aux)
    # ),
    # "MP2_test7": lambda p, n_occ, n_virt, nbf_aux: (
    #     p[0]
    #     + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # MP2 transformation: O(N_aux × N_bf^2 × N_occ)
    #     + p[4] * (nbf_aux**2 * n_occ * n_virt)  # MP2 fitting: O(N_occ × N_virt × N_aux^2)
    #     + p[5] * (nbf_aux * n_occ**2 * n_virt**2)  # MP2 energy: O(N_occ^2 × N_virt^2 × N_aux)
    # ),
    # "wB97X-V_2_le": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)
    #     + p[2] * (nbf_aux * (n_occ + n_virt)**2)
    #     + p[3] * ((n_occ + n_virt) * np_total)
    #     + p[4] * ((0.5 * np_total) ** 2)
    # ),
    # "wB97X-V_2_gt": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0] + p[1] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)
    #     + p[2] * (nbf_aux * (n_occ + n_virt)**2)
    #     + p[3] * ((n_occ + n_virt) ** 1.1 * np_total)
    #     + p[4] * ((0.8*np_total) ** 2)
    # ),
    # "B3LYP-D3_1": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0] + p[1] * ((n_occ + n_virt) * np_total)
    #     + p[2] * (((n_occ + n_virt)**2 * nbf_aux) + ((n_occ + n_virt)**2 * nbf_aux) + nbf_aux**2)
    #     + p[3] * (nbf_aux**2 + nbf_aux**3)
    # ),
    # "B2PLYP-D3_2": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]  # Constant overhead
    #     + p[1] * (np_total * (n_occ + n_virt))  # DFT XC (B88+LYP): O(N_grid × N_bf^1.4)
    #     + p[2] * (nbf_aux * (n_occ + n_virt)**2)  # J matrix: O(N_aux × N_bf^2)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # K matrix: O(N_bf^2 × N_occ × N_aux)
    #     + p[4] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # MP2 transformation: O(N_aux × N_bf^2 × N_occ)
    #     + p[5] * (nbf_aux**2 * n_occ * n_virt)  # MP2 fitting: O(N_occ × N_virt × N_aux^2)
    #     + p[6] * (nbf_aux * n_occ**2 * n_virt**2)  # MP2 energy: O(N_occ^2 × N_virt^2 × N_aux)
    # ),
    # "B2PLYP-D3_2_1": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]  # Constant overhead
    #     + p[1] * (np_total * (n_occ + n_virt)**1.2)  # DFT XC (B88+LYP): O(N_grid × N_bf^1.4)
    #     + p[2] * (nbf_aux * (n_occ + n_virt)**2)  # J matrix: O(N_aux × N_bf^2)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # K matrix: O(N_bf^2 × N_occ × N_aux)
    #     + p[4] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # MP2 transformation: O(N_aux × N_bf^2 × N_occ)
    #     + p[5] * (nbf_aux**2 * n_occ * n_virt)  # MP2 fitting: O(N_occ × N_virt × N_aux^2)
    #     + p[6] * (nbf_aux * n_occ**2 * n_virt**2)  # MP2 energy: O(N_occ^2 × N_virt^2 × N_aux)
    # ),
    # "M05-2X_2": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]  # Constant overhead
    #     + p[1] * (np_total * (n_occ + n_virt)**1.2)  # Meta-GGA XC: O(N_grid × N_bf^1.4)
    #     + p[2] * (nbf_aux * (n_occ + n_virt)**2)  # J matrix: O(N_aux × N_bf^2)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2 * n_occ)  # K matrix: O(N_bf^2 × N_occ × N_aux)
    # ),
    # "PBE-D3_2": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * ((n_occ + n_virt)**1.25 * np_total)
    #     + p[2] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2)
    # ),
    # "PBE-D3_2_1": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * ((n_occ + n_virt)**1.25 * np_total)
    #     + p[2] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)
    #     + p[3] * (nbf_aux * (n_occ + n_virt)**2)
    # ),
    # # "PBE0": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    # #     p[0]
    # #     + p[1] * nbf_aux * (n_occ + n_virt)**2                 # J matrix
    # #     + p[2] * nbf_aux * (n_occ + n_virt)**2 * n_occ         # 25% exact exchange
    # #     + p[3] * np_total * (n_occ + n_virt)**1.5              # GGA grid integration
    # # ),
    # # "PBE-D3_1": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    # #     p[0]
    # #     + p[1] * ((n_occ + n_virt) * np_total)
    # #     + p[2] * (((n_occ + n_virt)**2 * nbf_aux) + ((n_occ + n_virt)**2 * nbf_aux) + nbf_aux**2)
    # #     + p[3] * (nbf_aux**2 + nbf_aux**3)
    # # ),
    # "CCSD_test_1": lambda p, n_occ, n_virt: (
    #     p[0]
    #     + p[1] * (n_occ ** 1 * n_virt ** 1)
    #     + p[2] * (n_occ ** 3 * n_virt ** 2)
    #     + p[3] * (n_occ ** 2 * n_virt ** 2)
    #     + p[4] * (n_occ ** 2 * n_virt ** 3)
    #     + p[5] * (n_occ ** 3 * n_virt ** 3)
    #     + p[6] * (n_occ ** 2 * n_virt ** 4)
    #     + p[7] * (n_occ ** 4 * n_virt ** 2)
    # ),
    # true fits below
    # "HF_true": lambda p, n_occ, n_virt, nbf_aux: (
    #     p[0]
    #     + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)          # J Build
    #     + p[2] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)  # K Build
    # ),
    # "MP2_true": lambda p, n_occ, n_virt, nbf_aux: (
    #     p[0]
    #     + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)               # HF J Build
    #     + p[2] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)       # HF K Build
    #     + p[3] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)         # 3-index int 1
    #     + p[4] * (n_occ * n_virt * (n_occ + n_virt) * nbf_aux)   # 3-index int 2
    #     + p[5] * (n_occ**2 * n_virt**2 * nbf_aux)                # int transform
    # ),
    # "B3LYP-D3_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)              # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)          # HF J Build
    #     + p[3] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)  # HF K Build
    # ),
    # "B2PLYP-D3_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)                   # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)               # HF J Build
    #     + p[3] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)       # HF K Build
    #     + p[4] * (n_occ * (n_occ + n_virt)**2 * nbf_aux)         # MP2 3-index int 1
    #     + p[5] * (n_occ * n_virt * (n_occ + n_virt) * nbf_aux)   # MP2 3-index int 2
    #     + p[6] * (n_occ**2 * n_virt**2 * nbf_aux)                # MP2 int transform
    # ),
    # "M05-2X_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)              # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)          # HF J Build
    #     + p[3] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)  # HF K Build
    # ),
    # "PBE-D3_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)              # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)          # HF J Build
    # ),
    # "FNO-CCSD_true": lambda p, n_occ, n_virt: (
    #     p[0]
    #     + p[1] * (n_occ * n_virt)             # J-term
    #     + p[2] * (n_occ ** 3 * n_virt ** 2)   # H-term
    #     + p[3] * (n_occ ** 2 * n_virt ** 2)   # F,I-term
    #     + p[4] * (n_occ ** 2 * n_virt ** 3)   # G-term
    #     + p[5] * (n_occ ** 3 * n_virt ** 3)   # C,D-term
    #     + p[6] * (n_occ ** 2 * n_virt ** 4)   # B-term
    #     + p[7] * (n_occ ** 4 * n_virt ** 2)   # A-term
    # ),
    # "FNO-CCSD(T)_true": lambda p, n_occ, n_virt: (
    #     p[0]
    #     + p[1] * (n_occ * n_virt)             # J-term
    #     + p[2] * (n_occ ** 3 * n_virt ** 2)   # H-term
    #     + p[3] * (n_occ ** 2 * n_virt ** 2)   # F,I-term
    #     + p[4] * (n_occ ** 2 * n_virt ** 3)   # G-term
    #     + p[5] * (n_occ ** 3 * n_virt ** 3)   # C,D-term
    #     + p[6] * (n_occ ** 2 * n_virt ** 4)   # B-term
    #     + p[7] * (n_occ ** 4 * n_virt ** 2)   # A-term
    #     + p[8] * (n_occ ** 3 * n_virt ** 4)   # T algorithm?
    # ),
    # "wB97X-V_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)              # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)          # HF J Build
    #     + p[3] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)  # HF K Build
    #     + p[4] * (np_total ** 2)                            # VV10
    # ),
    # "wB97X-D_true": lambda p, n_occ, n_virt, nbf_aux, np_total: (
    #     p[0]
    #     + p[1] * (np_total * (n_occ + n_virt)**2)              # DFT exchange SHOULD THIS BE Nao^1 ?
    #     + p[2] * (nbf_aux * (n_occ + n_virt) ** 2)          # HF J Build
    #     + p[3] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux)  # HF K Build
    # ),
    # .dat fits below
    "HF_dat": lambda p, n_occ, n_virt, nbf_aux: (
        p[0]
        # + p[1] * (nbf_aux * (n_occ + n_virt) ** 2)                    # DFH: AO Construction, 3-idx eris in AO basis  (psi4/src/psi4/lib3index/dfhelper.cc:1284-1347) (symmetric case)
        # + p[2] * ((n_occ + n_virt) ** 2 * nbf_aux ** 2)**0.65         # AO-Met. Contraction     (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678) (symmetric case)
        # + p[3] * ((n_occ + n_virt) ** 2 * nbf_aux)**0.80              # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223) (symmetric case)
        # + p[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)      # K                       (psi4/src/psi4/lib3index/dfhelper.cc:3350-3377) (generic)
        # + p[1] * (nbf_aux * (n_occ + n_virt) ** 1.5)                # DFH: AO Construction
        # + p[2] * ((n_occ + n_virt) ** 1.5 * nbf_aux ** 2)           # AO-Met. Contraction
        # + p[3] * ((n_occ + n_virt) ** 1.5 * nbf_aux)                # J
        + p[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)  # K
    ),
    "PBE-D3_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (np_total)
        ** 1.0  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:202-221) & (GGA)
        + p[2]
        * (nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.90  # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223)
    ),
    "wB97X-D_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (
            np_total * (n_occ + n_virt) ** 1.0
        )  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:202-221) (GGA)
        + p[2]
        * (
            nbf_aux * (n_occ + n_virt) ** 2
        )  # (w)AO construction      (psi4/src/psi4/lib3index/dfhelper.cc:1284-1347) & (psi4/src/psi4/lib3index/dfhelper.cc:1349-1417)
        + p[3]
        * ((n_occ + n_virt) ** 2 * nbf_aux**2)
        ** 0.65  # (w)AO-Met. Contraction  (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678)
        + p[4]
        * ((n_occ + n_virt) ** 2 * nbf_aux)
        ** 0.80  # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223) (symmetric case)
        + p[5]
        * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.80  # (w)K                    (psi4/src/psi4/lib3index/dfhelper.cc:3378-3438)
    ),
    "wB97X-V_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (
            np_total * (n_occ + n_virt) ** 1.0
        )  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:202-221) (GGA)
        + p[2]
        * (
            nbf_aux * (n_occ + n_virt) ** 2
        )  # (w)AO construction      (psi4/src/psi4/lib3index/dfhelper.cc:1284-1347) & (psi4/src/psi4/lib3index/dfhelper.cc:1349-1417)
        + p[3]
        * ((n_occ + n_virt) ** 2 * nbf_aux**2)
        ** 0.65  # (w)AO-Met. Contraction  (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678)
        + p[4]
        * ((n_occ + n_virt) ** 2 * nbf_aux)
        ** 0.80  # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223) (symmetric case)
        + p[5]
        * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.80  # (w)K                    (psi4/src/psi4/lib3index/dfhelper.cc:3378-3438)
        + p[6]
        * (np_total**2)
        ** 0.70  # RV: Form V, VV10                    (psi4/src/psi4/libfunctional/superfunctional.cc:928-1065) VV10 uses separate grid with flat pruning scheme followed by additional pruning
    ),
    "MP2_dat": lambda p, n_occ, n_virt, nbf_aux: (
        p[0]
        + p[1] * (n_occ * (n_occ + n_virt) ** 2 * nbf_aux) ** 0.85  # HF K Build
        + p[2]
        * ((n_occ + n_virt) ** 2 * nbf_aux)
        ** 0.70  # DFMP2 (A|mn), construct 3-index ints
    ),
    "B3LYP-D3_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (
            np_total * (n_occ + n_virt) ** 1.0
        )  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:223-298) (meta GGA)
        + p[2]
        * (
            (n_occ + n_virt) ** 2 * nbf_aux**2
        )  # AO-Met. Contraction     (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678) (symmetric case)
        + p[3]
        * (
            (n_occ + n_virt) ** 2 * nbf_aux
        )  # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223) (symmetric case)
        + p[4]
        * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.75  # K                       (psi4/src/psi4/lib3index/dfhelper.cc:3350-3377) (generic)
    ),
    "B2PLYP-D3_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (np_total * (n_occ + n_virt) ** 1.0)
        ** 0.75  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:223-298) (GGA)
        + p[2]
        * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.80  # K                       (psi4/src/psi4/lib3index/dfhelper.cc:3350-3377) (generic)
        + p[3]
        * (
            (n_occ + n_virt) ** 2 * nbf_aux**2
        )  # AO-Met. Contraction     (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678) (symmetric case)
        + p[4]
        * ((n_occ + n_virt) ** 2 * nbf_aux)
        ** 0.80  # DFMP2 (A|mn), construct 3-index ints
    ),
    "M05-2X_dat": lambda p, n_occ, n_virt, nbf_aux, np_total: (
        p[0]
        + p[1]
        * (
            np_total * (n_occ + n_virt) ** 1.0
        )  # RV: Form V, DFT XC matrix           (psi4/src/psi4/libfock/v.cc:1261-1433) & (psi4/src/psi4/libfock/points.cc:223-298) (meta GGA)
        + p[2]
        * (
            (n_occ + n_virt) ** 2 * nbf_aux**2
        )  # AO-Met. Contraction     (psi4/src/psi4/lib3index/dfhelper.cc:1653-1678) (symmetric case)
        + p[3]
        * (
            (n_occ + n_virt) ** 2 * nbf_aux
        )  # J                       (psi4/src/psi4/lib3index/dfhelper.cc:3162-3223) (symmetric case)
        + p[4]
        * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)
        ** 0.75  # K                       (psi4/src/psi4/lib3index/dfhelper.cc:3350-3377) (generic)
    ),
    "FNO-CCSD_dat": lambda p, n_occ, n_virt: (
        p[0]
        + p[1]
        * (
            (n_occ + n_virt) ** 5
        )  # FNOCC: Two Elec Int Trans   (psi4/src/psi4/libtrans/integraltransform_tei_1st_half.cc)
        + p[2]
        * (n_occ**2 * n_virt**4)
        ** 1.0  # FNOCC: CCSD                 (psi4/src/psi4/fnocc/ccsd.cc:1414-1480) & (psi4/src/psi4/fnocc/ccsd.cc:1485-1560). DGEMM @ 1454-1461, 1526-1533
    ),
    "FNO-CCSD(T)_dat": lambda p, n_occ, nbf_aux, n_virt: (
        p[0]
        # + p[1] * ((n_occ + n_virt)**5)                               # FNOCC: Two Elec Int Trans   (psi4/src/psi4/libtrans/integraltransform_tei_1st_half.cc)
        + p[1] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)  # K                    # FNOCC: CCSD                 (psi4/src/psi4/fnocc/ccsd.cc:1414-1480) & (psi4/src/psi4/fnocc/ccsd.cc:1485-1560). DGEMM @ 1454-1461, 1526-1533
        + p[1] * (n_occ**3 * n_virt**4)** 1.0  # FNOCC: triples              (psi4/src/psi4/fnocc/triples.cc:180-378)
    ),
}

if COMPOSITE_POLYS:
    raise ValueError

else:
    POLY_CONFIGS = {
        # "wB97X-V": {
        #     "split": None,
        #     "split_type": "nbf_aux",
        #     "le": ("wB97X-V_dat", {"n_params": 7, "bounds": (0.0, None)}),
        #     "gt": ("wB97X-V_2_gt", {"n_params": 5, "bounds": (0.0, None)}),
        # },
        # "wB97X-D": {
        #     "split": None,
        #     "split_type": "nbf_aux",
        #     "le": ("wB97X-D_dat", {"n_params": 6, "bounds": (0.0, None)}),
        #     "gt": ("wB97X-V_2_gt", {"n_params": 5, "bounds": (0.0, None)}),
        # },
        # "B3LYP-D3": {
        #     "split": None,
        #     "split_type": "np_total",
        #     "le": ("B3LYP-D3_dat", {"n_params": 5, "bounds": (0.0, None)}),
        #     "gt": ("B3LYP-D3_1", {"n_params": 4, "bounds": (0.0, None)}),
        # },
        # "B2PLYP-D3": {
        #     "split": None,
        #     "split_type": "np_total",
        #     "le": ("B2PLYP-D3_dat", {"n_params": 5, "bounds": (0.0, None)}),
        #     "gt": ("B2PLYP-D3_2_1", {"n_params": 7, "bounds": (0.0, None)}),
        # },
        # "MP2": {
        #     "split": None,
        #     "split_type": "nbf_aux",
        #     "le": ("MP2_dat", {"n_params": 3, "bounds": (0.0, None)}),
        #     "gt": ("MP2_test7", {"n_params": 6, "bounds": (0.0, None)}),
        # },
        "HF": {
            "split": None,
            "split_type": "nbf_aux",
            "le": ("HF_dat", {"n_params": 2, "bounds": (0.0, None), "params0": [1.0, 1.0]}),
            "gt": ("HF_test4", {"n_params": 3, "bounds": (0.0, None)}),
        },
        # "M05-2X": {
        #     "split": None,
        #     "split_type": "nbf_aux",
        #     "le": ("M05-2X_dat", {"n_params": 5, "bounds": (0.0, None)}),
        #     "gt": ("M05-2X_2", {"n_params": 4, "bounds": (0.0, None)}),
        # },
        # "PBE-D3": {
        #     "split": None,
        #     "split_type": "basis",
        #     "le": ("PBE-D3_dat", {"n_params": 3, "bounds": (0.0, None)}),
        #     "gt": ("PBE-D3_2_1", {"n_params": 4, "bounds": (0.0, None)}),
        # },
        "FNO-CCSD(T)": {
            "split": None,
            "split_type": "np_total",
            "le": (
                "FNO-CCSD(T)_dat",
                {"n_params": 3, "bounds": [(0.1, 100), (1e-14, 1e-8), (1e-14, 1e-8)],
                }
            ),
            "gt": ("M05-2X_1", {"n_params": 3, "bounds": (0.0, None)}),
        },
        # pre powell fit: grid search on coefficients grid must match feasible numerical ranges
        # "FNO-CCSD": {
        #     "split": None,
        #     "le": ("FNO-CCSD_dat", {"n_params": 3, "bounds": (0.0, None)}),
        #     "gt": ("M05-2X_1", {"n_params": 3, "bounds": (0.0, None)}),
        # },
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


def fit_scaler(values, scaler_type, enforce_nonnegative=False):
    arr = np.asarray(values, dtype=float)
    if scaler_type == "minmax":
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        return {"type": "minmax", "min": vmin, "max": vmax}
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    scaler = {"type": "standard", "mean": mean, "std": std}
    if enforce_nonnegative:
        denom = std if std != 0 else 1.0
        z_min = np.nanmin((arr - mean) / denom)
        if z_min < 0:
            scaler["shift"] = -z_min
            print(
                "Warning: standard scaling produced negative values; "
                f"applying shift={scaler['shift']:.6f} to keep inputs nonnegative."
            )
    return scaler


def scale_values(values, scaler):
    arr = np.asarray(values, dtype=float)
    if scaler is None:
        return arr
    if scaler["type"] == "minmax":
        denom = scaler["max"] - scaler["min"]
        if denom == 0:
            return np.zeros_like(arr)
        return (arr - scaler["min"]) / denom
    denom = scaler["std"]
    if denom == 0:
        return np.zeros_like(arr)
    scaled = (arr - scaler["mean"]) / denom
    if "shift" in scaler:
        scaled = scaled + scaler["shift"]
    return scaled


def get_df_columns_for_poly(poly_func):
    var_mapping = {
        "n_occ": "nocc",
        "n_virt": "nvirt",
        "nbf_aux": "nbf_aux",
        "np_total": "np_total",
        "nbf": "nbf",
    }
    var_names = get_polynomial_variables(poly_func)
    return [var_mapping.get(var, var) for var in var_names]


def apply_feature_scaling(df, feature_scalers, poly_func):
    if not feature_scalers:
        return df
    df_scaled = df.copy()
    for col in get_df_columns_for_poly(poly_func):
        if col in df_scaled.columns and col in feature_scalers:
            df_scaled[col] = scale_values(df_scaled[col].values, feature_scalers[col])
    return df_scaled


def get_split_series(df, split_type):
    allowed = {"n_occ", "n_virt", "nbf_aux", "np_total", "nbf", "time", "basis"}
    if split_type not in allowed:
        raise ValueError(
            f"Invalid split_type '{split_type}'. Allowed values: {sorted(allowed)}"
        )
    var_mapping = {
        "n_occ": "nocc",
        "n_virt": "nvirt",
        "nbf_aux": "nbf_aux",
        "np_total": "np_total",
        "nbf": "nbf",
        "time": "log(time(s))",
        "basis": "basis",
    }
    col = var_mapping[split_type]
    if col not in df.columns:
        raise KeyError(
            f"split_type '{split_type}' mapped to '{col}' not found in dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )
    return df[col]


def predict_log_time(params, df, poly_name, feature_scalers=None):
    poly_func = time_polynomials[poly_name]
    df_use = apply_feature_scaling(df, feature_scalers, poly_func)
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df_use, var_names)
    raw = poly_func(params, *var_arrays)
    eps = 1e-12
    return np.log10(np.clip(np.abs(raw), eps, None))


def predict_time(params, df, poly_name, feature_scalers=None):
    poly_func = time_polynomials[poly_name]
    df_use = apply_feature_scaling(df, feature_scalers, poly_func)
    var_names = get_polynomial_variables(poly_func)
    var_arrays = extract_variables_from_df(df_use, var_names)
    raw = poly_func(params, *var_arrays)
    return np.abs(raw)


def optimize_time(
    df_train, poly_name, config, feature_scalers=None, target_scaler=None
):
    n_params = config["n_params"]
    if isinstance(config["bounds"], tuple):
        bounds = [config["bounds"]] * n_params

    else:
        bounds = config["bounds"]
        assert len(bounds) == n_params, (
            "interpreting bounds as a list, bounds given dont match number of parameters"
        )
    if "params0" in config.keys():
        params0 = config["params0"] 
    
    else:
        params0 = [5.67912758e-12] * n_params
        params0[0] = 3.20909883e01

    best = {"params": np.array(params0, dtype=float), "loss": np.inf}

    def rmse(params):
        if FIT_LINEAR_TIME:
            pred = predict_time(
                params, df_train, poly_name, feature_scalers=feature_scalers
            )
            if target_scaler is not None:
                pred = scale_values(pred, target_scaler)
                target = scale_values(df_train["time(s)"].values, target_scaler)
            else:
                target = df_train["time(s)"].values
            v = root_mean_squared_error(target, pred)
        else:
            pred_log = predict_log_time(
                params, df_train, poly_name, feature_scalers=feature_scalers
            )
            v = root_mean_squared_error(df_train["log(time(s))"], pred_log)
        if v < best["loss"]:
            best["loss"] = float(v)
            best["params"] = np.array(params, dtype=float).copy()
        print(params, v)
        return v

    print(f"Timing polynomial: {poly_name}, n_params: {n_params}")
    print(f"Variables: {get_polynomial_variables(time_polynomials[poly_name])}")

    ret = minimize(rmse, x0=params0, method="powell", bounds=bounds)
    if best["loss"] + 1e-12 < ret.fun:
        print(
            "Warning: Powell returned a non-best iterate; "
            f"using best evaluated params with RMSE={best['loss']:.6f} instead of {ret.fun:.6f}."
        )
        return best["params"]
    return ret.x


def get_basis_colors(basis_sets):
    n = len(basis_sets)
    cmap = plt.cm.tab10 if n <= 10 else (plt.cm.tab20 if n <= 20 else plt.cm.viridis)
    return {b: cmap(i / max(n - 1, 1)) for i, b in enumerate(sorted(basis_sets))}


def latex_escape(text):
    return text.replace("_", "\\_")


def format_coefficients(params):
    lines = [f"\\(c_{{{i}}}={val:.3e}\\)" for i, val in enumerate(params)]
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
    expr = re.sub(r"\(n_occ\s*\+\s*n_virt\)", "N_{ao}", expr)
    expr = re.sub(r"n_occ\s*\+\s*n_virt", "N_{ao}", expr)
    expr = re.sub(r"p\[(\d+)\]", r"c_{\1}", expr)
    replacements = {
        "n_occ": "N_{occ}",
        "n_virt": "N_{virt}",
        "nbf_aux": "N_{aux}",
        "np_total": "N_{p}",
        "nbf": "N_{bf}",
    }
    for key, val in replacements.items():
        expr = expr.replace(key, val)
    expr = expr.replace("**", "^")
    expr = re.sub(r"\^(\d*\.\d+)", r"^{\1}", expr)
    expr = expr.replace("*", " \\cdot ")
    expr = expr.replace("+ -", "- ")
    expr = expr.replace(" - ", " + - ")
    parts = [part.strip() for part in expr.split(" + ") if part.strip()]
    if not parts:
        return ""
    lines = []
    for i, part in enumerate(parts):
        if i == 0:
            lines.append(part)
        elif part.startswith("-"):
            lines.append(part)
        else:
            lines.append(f"+ {part}")
    expr_lines = " \\\\ ".join(lines)
    return "$\\begin{array}{l}" + expr_lines + "\\end{array}$"


def write_latex_table(rows, output_path):
    header = (
        "\\section{Time Estimation Equations}\n\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{longtable}{|l|c|c|c|c|p{3.2cm}|p{5.0cm}|}\n"
        "\\hline\n"
        "\\textbf{Method} & \\textbf{\# train} & \\textbf{\# test} "
        "& \\textbf{Train MAE} & \\textbf{Test MAE} & \\textbf{Coeffs} & \\textbf{Functional Form} \\\\ \n"
        "\\hline\n"
        "\\endhead\n"
    )

    lines = [header]
    for row in rows:
        method = latex_escape(row["method"]) if row.get("method") else ""
        split_label = row.get("split_label", "")
        if COMPOSITE_POLYS and split_label in {"high", "low"}:
            method = latex_escape(f"{row['method']} ({split_label})")
        lines.append(
            f"{method} & {row['n_train']} & {row['n_test']} & "
            f"{row['train_mae']:.2f} & {row['test_mae']:.2f} & "
            f"{row['coeffs']} & {row['func']} \\\\ \n\\hline\n"
        )

    lines.append("\\end{longtable}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


def save_plot(fig, path_base):
    if LATEX_READY_PLOTS:
        ext = LATEX_READY_FORMAT
        path = f"{path_base}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        path = f"{path_base}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")


def build_table_row(method, split_label, fit, poly_name):
    poly_func = time_polynomials[poly_name]
    return {
        "method": method,
        "split_label": split_label,
        "n_train": len(fit["df_train"]),
        "n_test": len(fit["df_test"]),
        "train_rmse": fit["train_rmse"],
        "test_rmse": fit["test_rmse"],
        "train_mae": fit["train_mae"],
        "test_mae": fit["test_mae"],
        "min_abs_err": fit["min_abs_err"],
        "max_abs_err": fit["max_abs_err"],
        "coeffs": format_coefficients(fit["params"]),
        "func": format_polynomial_expression(poly_func),
    }


def plot_train_test(
    method,
    label,
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
):
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, x, y, title, color, rmse, r2, mae, n in [
        (
            ax_train,
            y_train,
            y_train_pred,
            "Training",
            "steelblue",
            train_rmse,
            train_r2,
            train_mae,
            len(df_train),
        ),
        (
            ax_test,
            y_test,
            y_test_pred,
            "Test",
            "orange",
            test_rmse,
            test_r2,
            test_mae,
            len(df_test),
        ),
    ]:
        if LATEX_READY_PLOTS:
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot(
                [min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1
            )
            ax.scatter(x, y, alpha=1.0, color=color, s=50, edgecolors="none", zorder=2)
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
        else:
            ax.scatter(x, y, alpha=0.6, color=color, s=50)
            ax.plot([x.min(), x.max()], [x.min(), x.max()], "r--", lw=2)
            ax.text(
                0.02,
                0.98,
                f"RMSE={rmse:.4f}\nR2={r2:.4f}\nMAE={mae:.4f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
            ax.set_title(
                f"{method} {label} {title}\n"
                f"n={n}, RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}",
                fontsize=13,
            )
            ax.grid(True, alpha=0.3)

        if not LATEX_READY_PLOTS:
            ax.set_xlabel("Actual log10(time(s))", fontsize=12)
            ax.set_ylabel("Predicted log10(time(s))", fontsize=12)

    plt.tight_layout()
    path_base = os.path.join(PLOTS_DIR, f"{method}_{label}_subset")
    save_plot(fig, path_base)
    plt.close(fig)


def plot_combined(
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
):
    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    norm = None
    cmap = None
    if COLOR_BY_SIZE:
        nbf_min = min(df_train[SZ_VAR].min(), df_test[SZ_VAR].min())
        nbf_max = max(df_train[SZ_VAR].max(), df_test[SZ_VAR].max())
        norm = Normalize(vmin=nbf_min, vmax=nbf_max)
        cmap = plt.cm.viridis

    def scatter_by_mode(ax, df_subset, y_subset, y_pred_subset):
        alpha = 1.0 if LATEX_READY_PLOTS else 0.6
        edgecolors = "none" if LATEX_READY_PLOTS else None
        if COLOR_BY_SIZE:
            return ax.scatter(
                y_subset,
                y_pred_subset,
                s=50,
                alpha=alpha,
                c=df_subset[SZ_VAR].values,
                cmap=cmap,
                norm=norm,
                edgecolors=edgecolors,
            )
        if COLOR_BY_POLY:
            color_map = {"high": "steelblue", "low": "orange"}
            labels_map = {"high": "split high", "low": "split low"}
            for key in ["high", "low"]:
                mask = df_subset["time_group"] == key
                if mask.any():
                    ax.scatter(
                        y_subset[mask],
                        y_pred_subset[mask],
                        s=50,
                        alpha=alpha,
                        color=color_map[key],
                        label=labels_map[key],
                        edgecolors=edgecolors,
                    )
            return None
        if COLOR_BY_BASIS:
            basis_colors = get_basis_colors(df_subset["basis"].dropna().unique())
            for basis in sorted(set(df_subset["basis"].dropna().values)):
                mask = df_subset["basis"] == basis
                ax.scatter(
                    y_subset[mask],
                    y_pred_subset[mask],
                    s=50,
                    alpha=alpha,
                    color=basis_colors[basis],
                    label=basis,
                    edgecolors=edgecolors,
                )
            return None
        ax.scatter(
            y_subset,
            y_pred_subset,
            s=50,
            alpha=alpha,
            color="steelblue",
            edgecolors=edgecolors,
        )
        return None

    for ax, df_sub, y, y_pred, title, rmse, r2, mae, n in [
        (
            ax_train,
            df_train,
            y_train,
            y_train_pred,
            "- Train",
            train_rmse,
            train_r2,
            train_mae,
            len(df_train),
        ),
        (
            ax_test,
            df_test,
            y_test,
            y_test_pred,
            "- Test",
            test_rmse,
            test_r2,
            test_mae,
            len(df_test),
        ),
    ]:
        if LATEX_READY_PLOTS:
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot(
                [min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1
            )
            scatter_by_mode(ax, df_sub, y, y_pred)
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
            scatter_by_mode(ax, df_sub, y, y_pred)
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

    if COLOR_BY_SIZE and not LATEX_READY_PLOTS:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, cax=cax, label=SZ_VAR)
    else:
        cax.set_visible(False)

    plt.tight_layout()
    path_base = os.path.join(PLOTS_DIR, f"{method}_combined")
    save_plot(fig, path_base)
    plt.close(fig)


def train_time_subset(df_subset, method, label, poly_name, config):
    df_pool = df_subset[df_subset["time(s)"].notna()].copy()
    if len(df_pool) < 10:
        print(f"  {label}: insufficient data (< 10 points) — skipping.")
        return None

    df_train, df_test = train_test_split(df_pool, test_size=0.10, random_state=21)
    print(f"  {label}: Train {len(df_train)} Test {len(df_test)}")

    feature_scalers = None
    target_scaler = None
    poly_func = time_polynomials[poly_name]
    if SCALE_FEATURES:
        feature_scalers = {}
        for col in get_df_columns_for_poly(poly_func):
            if col in df_train.columns:
                feature_scalers[col] = fit_scaler(
                    df_train[col].values,
                    SCALER_TYPE,
                    enforce_nonnegative=True,
                )
    if FIT_LINEAR_TIME and SCALE_TARGET:
        target_scaler = fit_scaler(df_train["time(s)"].values, SCALER_TYPE)

    params = optimize_time(
        df_train,
        poly_name,
        config,
        feature_scalers=feature_scalers,
        target_scaler=target_scaler,
    )
    print(f"  {label} params: {params}")
    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values
    y_train_pred = predict_log_time(
        params, df_train, poly_name, feature_scalers=feature_scalers
    )
    y_test_pred = predict_log_time(
        params, df_test, poly_name, feature_scalers=feature_scalers
    )

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    abs_err_train = np.abs(y_train - y_train_pred)
    abs_err_test = np.abs(y_test - y_test_pred)
    train_mae = np.mean(abs_err_train)
    test_mae = np.mean(abs_err_test)

    print(
        f"  {label} — Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}"
    )
    print(
        f"  {label} — Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}"
    )

    plot_train_test(
        method,
        label,
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

    return {
        "params": params,
        "feature_scalers": feature_scalers,
        "df_train": df_train,
        "df_test": df_test,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "min_abs_err": min(abs_err_train.min(), abs_err_test.min()),
        "max_abs_err": max(abs_err_train.max(), abs_err_test.max()),
    }


def run_method(method):
    timing_pkl = os.path.join(ROOT, DATA, f"{method}.pkl")

    print(f"Loading {method} timing data...")
    df_time = pd.read_pickle(timing_pkl)

    df = df_time.copy()

    # Rename for modeling
    df = df.rename(
        columns={
            "n_occupied": "nocc",
            "n_virtual": "nvirt",
            "basis_set": "basis",
            "user_time (parse)": "time(s)",
            # "wall_time (parse)": "time(s)",
        }
    )

    if ADJUST_FOR_MULTIPLICITY:
        mult_col = (
            "multiplicity (parse)" if "multiplicity (parse)" in df.columns else None
        )
        if mult_col is None:
            print("Multiplicity column not found; skipping multiplicity adjustment.")
        else:
            mask = df[mult_col].notna() & (df[mult_col] != 1)
            count = int(mask.sum())
            if count:
                df.loc[mask, ["nocc", "nvirt"]] = df.loc[mask, ["nocc", "nvirt"]] * 2
            print(f"Multiplicity adjustment: {count} rows doubled (col: {mult_col}).")

    df["nbf"] = df["nocc"] + df["nvirt"]

    # Timing targets
    df["log(time(s))"] = np.log10(df["time(s)"])

    # Partition by log(time) threshold (if configured)
    df_valid = df[df["time(s)"].notna()].copy()
    df_valid = df_valid[df_valid["time(s)"] > 10**1.0].copy()
    poly_cfg = POLY_CONFIGS.get(method)
    if poly_cfg is None:
        print(f"No POLY_CONFIGS entry for method '{method}'. Skipping.")
        return

    split_val = poly_cfg.get("split")
    split_type = poly_cfg.get("split_type")
    if split_val is None:
        df_valid["time_group"] = "low"
        df_low = df_valid.copy()
        df_high = df_valid.iloc[0:0].copy()
        print("No split configured; fitting all data with 'le' polynomial.")
    else:
        if split_type is None:
            raise KeyError(
                f"POLY_CONFIGS for method '{method}' must define 'split_type' when 'split' is set."
            )
        if split_type == "basis" or split_val == "basis":
            if "basis" not in df_valid.columns:
                raise KeyError(
                    f"split_type 'basis' requires a 'basis' column for method '{method}'."
                )
            is_aug = (
                df_valid["basis"].astype(str).str.contains("aug", case=False, na=False)
            )
            df_valid["time_group"] = np.where(is_aug, "high", "low")
            df_high = df_valid[df_valid["time_group"] == "high"].copy()
            df_low = df_valid[df_valid["time_group"] == "low"].copy()
            print(f"Split: basis contains 'aug' rows: {len(df_high)}")
            print(f"Split: basis without 'aug' rows: {len(df_low)}")
        else:
            split_series = get_split_series(df_valid, split_type)
            df_valid["time_group"] = np.where(split_series > split_val, "high", "low")
            df_high = df_valid[df_valid["time_group"] == "high"].copy()
            df_low = df_valid[df_valid["time_group"] == "low"].copy()
            print(f"Split: {split_type} > {split_val} rows: {len(df_high)}")
            print(f"Split: {split_type} <= {split_val} rows: {len(df_low)}")

    # Train timing polynomials for each subset
    poly_le, cfg_le = poly_cfg["le"]
    poly_gt, cfg_gt = poly_cfg["gt"]

    high_fit = None
    if split_val is not None:
        print(f"\nFitting timing polynomial for split high subset: {poly_gt}")
        high_fit = train_time_subset(df_high, method, "log_gt", poly_gt, cfg_gt)
    print(f"\nFitting timing polynomial for split low subset: {poly_le}")
    low_fit = train_time_subset(df_low, method, "log_le", poly_le, cfg_le)

    if low_fit is None or (split_val is not None and high_fit is None):
        print(
            "One of the subsets did not have enough data to fit. Skipping combined plot."
        )
        return None

    df_train, df_test = train_test_split(
        df_valid,
        test_size=0.10,
        random_state=21,
        stratify=df_valid["time_group"],
    )

    pred_train = pd.Series(
        index=df_train.index, dtype=float
    )  # LOOK HERE FOR HOW OVERALL MAE CALCULATED
    pred_test = pd.Series(index=df_test.index, dtype=float)

    mask_train_high = df_train["time_group"] == "high"
    mask_train_low = df_train["time_group"] == "low"
    mask_test_high = df_test["time_group"] == "high"
    mask_test_low = df_test["time_group"] == "low"

    if mask_train_high.any() and high_fit is not None:
        pred_train.loc[mask_train_high] = predict_log_time(
            high_fit["params"],
            df_train[mask_train_high],
            poly_gt,
            feature_scalers=high_fit.get("feature_scalers"),
        )
    if mask_train_low.any():
        pred_train.loc[mask_train_low] = predict_log_time(
            low_fit["params"],
            df_train[mask_train_low],
            poly_le,
            feature_scalers=low_fit.get("feature_scalers"),
        )

    if mask_test_high.any() and high_fit is not None:
        pred_test.loc[mask_test_high] = predict_log_time(
            high_fit["params"],
            df_test[mask_test_high],
            poly_gt,
            feature_scalers=high_fit.get("feature_scalers"),
        )
    if mask_test_low.any():
        pred_test.loc[mask_test_low] = predict_log_time(
            low_fit["params"],
            df_test[mask_test_low],
            poly_le,
            feature_scalers=low_fit.get("feature_scalers"),
        )

    y_train = df_train["log(time(s))"].values
    y_test = df_test["log(time(s))"].values
    y_train_pred = pred_train.values
    y_test_pred = pred_test.values

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    if split_val is None and low_fit is not None:
        low_fit["train_rmse"] = train_rmse
        low_fit["test_rmse"] = test_rmse
        low_fit["train_mae"] = train_mae
        low_fit["test_mae"] = test_mae

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

    rows = []
    if split_val is not None and high_fit is not None:
        rows.append(build_table_row(method, "high", high_fit, poly_gt))
    split_label = "low" if split_val is not None else "all"
    rows.append(build_table_row(method, split_label, low_fit, poly_le))
    return rows


def main():
    apply_plot_style()
    latex_rows = []
    method_order = [
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
    ordered_methods = [m for m in method_order if m in POLY_CONFIGS]
    remaining_methods = sorted(
        m for m in POLY_CONFIGS.keys() if m not in ordered_methods
    )
    for method in ordered_methods + remaining_methods:
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        method_rows = run_method(method)
        if method_rows:
            if not COMPOSITE_POLYS:
                for idx, row in enumerate(method_rows):
                    if idx > 0:
                        row["method"] = ""
            latex_rows.extend(method_rows)

    if WRITE_LATEX_TABLE and latex_rows:
        output_path = os.path.join(PLOTS_DIR, LATEX_TABLE_NAME)
        write_latex_table(latex_rows, output_path)


if __name__ == "__main__":
    main()
