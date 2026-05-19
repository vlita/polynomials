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


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "filtered_method_dfs"
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "2step_plots")
BLOCK_PLOTS_DIR = os.path.join(PLOTS_DIR, "split_blocks")
LATEX_FORMAT = "png"
LATEX_TABLE_NAME_RESTRICTED = "time_fit_template_restricted.tex"
LATEX_TABLE_NAME_UNRESTRICTED = "time_fit_template_unrestricted.tex"
LATEX_TABLE_NAME_COMBINED = "time_fit_template_combined.tex"
LATEX_TABLE_NAME_BINNED_RMSE = "time_fit_rmse_binned.tex"
INFERENCE_DF_NAME_RESTRICTED = "time_fit_inference_df_restricted.pkl"
INFERENCE_CSV_NAME_RESTRICTED = "time_fit_inference_df_restricted.csv"
INFERENCE_DF_NAME_UNRESTRICTED = "time_fit_inference_df_unrestricted.pkl"
INFERENCE_CSV_NAME_UNRESTRICTED = "time_fit_inference_df_unrestricted.csv"
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

unrestricted_polynomials = {
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
                3.0909883e01, 
                3.81966012e-08,
                3.81966012e-08,
                1.67912758e-8, # sweeping from e-0.8 to e-0.11 and tuning tight_delta within +-1 OM
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
                1.0e-9,
                1.0e-9,
                1.0e-9,
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
                0.20909883e01,
                3.81966012e-10,
                3.81966012e-09,
                3.81966012e-09,
                3.81966012e-11,
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
            # "coarse_options": {
            #     "maxiter": 400,
            #     "maxfev": 6000, 
            #     "xtol": 1e-10, 
            #     "ftol": 1e-10, 
            #     "disp": True
            #     },
            "run_tight": True,
            "tight_method": "L-BFGS-B",
            "tight_delta": [
                10,
                1.0e-11,
                1.0e-10,
                1.0e-10,
                1.0e-12,
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
                7.67912758e-8,
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
            + p[1] * ((n_occ + n_virt) ** p[5] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)                # J / contraction / (A|mn)
            + p[3] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[4] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy

        ),
        "coeffs": {
            "n_params": 6,
            "guess": [
                32.6,
                3.81966012e-10,
                1.67912758e-10,
                1.97912758e-10,
                7.67912758e-12,
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
            + p[1] * (np_total * (n_occ + n_virt) ** p[6])
            + p[2] * ((n_occ + n_virt) ** p[7] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[3] * ((n_occ + n_virt) ** p[8] * nbf_aux)                # J / contraction / (A|mn)
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[5] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy
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
        },
    },
    "M05-2X": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * ((n_occ + n_virt) ** p[5] * nbf_aux**2)
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)
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

restricted_polynomials = {
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
            + p[1] * ((n_occ + n_virt) ** p[5] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)                # J / contraction / (A|mn)
            + p[3] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[4] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy

        ),
        "coeffs": {
            "n_params": 6,
            "guess": [
                0.000000,
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
            + p[1] * (np_total * (n_occ + n_virt) ** p[6])
            + p[2] * ((n_occ + n_virt) ** p[7] * nbf_aux ** 2)           # AO-Met. Contraction
            + p[3] * ((n_occ + n_virt) ** p[8] * nbf_aux)                # J / contraction / (A|mn)
            + p[4] * (n_occ * nbf_aux * (n_occ + n_virt) ** 2)           # K
            + p[5] * (n_occ ** 2  * n_virt ** 2 * nbf_aux)               # MP2 energy
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
        },
    },
    "M05-2X": {
        "poly": lambda p, n_occ, n_virt, nbf_aux, np_total: (
            p[0]
            + p[1] * ((n_occ + n_virt) ** p[5] * nbf_aux**2)
            + p[2] * ((n_occ + n_virt) ** p[5] * nbf_aux)
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

    def get_arrays(group, split):
        if group is None:
            return np.array([]), np.array([])
        if split == "train":
            return group["y_train"], group["y_train_pred"]
        return group["y_test"], group["y_test_pred"]

    def compute_limits(groups, split):
        values = []
        for group in groups:
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
        for group in groups:
            y, y_pred = get_arrays(group, split)
            if len(y) > 0 and len(y_pred) > 0:
                ys.append(y)
                preds.append(y_pred)
        if not ys:
            return None
        y_all = np.concatenate(ys)
        pred_all = np.concatenate(preds)
        return float(np.mean(np.abs(y_all - pred_all)))

    for ax, split, title in [
        (ax_train, "train", "- Train"),
        (ax_test, "test", "- Test"),
    ]:
        limits = compute_limits([restricted, unrestricted], split)
        if limits is None:
            continue
        min_val, max_val = limits
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)

        y_r, y_r_pred = get_arrays(restricted, split)
        if len(y_r) > 0:
            ax.scatter(y_r, y_r_pred, s=50, alpha=1.0, color=RESTRICTED_COLOR, edgecolors="none", label="R")

        y_u, y_u_pred = get_arrays(unrestricted, split)
        if len(y_u) > 0:
            ax.scatter(
                y_u,
                y_u_pred,
                s=50,
                alpha=1.0,
                color=UNRESTRICTED_COLOR,
                edgecolors="none",
                label="U",
            )

        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        mae_lines = []
        r_text = format_mae_text(restricted, "R", split)
        if r_text is not None:
            mae_lines.append(("R", RESTRICTED_COLOR, r_text.split("R", 1)[-1].lstrip()))
        u_text = format_mae_text(unrestricted, "U", split)
        if u_text is not None:
            mae_lines.append(("U", UNRESTRICTED_COLOR, u_text.split("U", 1)[-1].lstrip()))
        total_mae = pooled_mae([restricted, unrestricted], split)
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

    for i, row in enumerate(rows):
        ax = axes[i]
        restricted = row.get("restricted")
        unrestricted = row.get("unrestricted")

        def get_arrays(group):
            if group is None:
                return np.array([]), np.array([])
            if split == "train":
                return group["y_train"], group["y_train_pred"]
            return group["y_test"], group["y_test_pred"]

        def compute_limits(groups):
            values = []
            for group in groups:
                y, y_pred = get_arrays(group)
                if len(y) > 0:
                    values.append(y)
                if len(y_pred) > 0:
                    values.append(y_pred)
            if not values:
                return None
            combined = np.concatenate(values)
            return combined.min(), combined.max()

        def format_mae_text(group, label):
            if group is None:
                return None
            mae = group["train_mae"] if split == "train" else group["test_mae"]
            if not np.isfinite(mae):
                return f"{label} MAE = n/a"
            return f"{label} MAE = {mae:.2f}"

        def pooled_mae(groups):
            ys = []
            preds = []
            for group in groups:
                y, y_pred = get_arrays(group)
                if len(y) > 0 and len(y_pred) > 0:
                    ys.append(y)
                    preds.append(y_pred)
            if not ys:
                return None
            y_all = np.concatenate(ys)
            pred_all = np.concatenate(preds)
            return float(np.mean(np.abs(y_all - pred_all)))

        limits = compute_limits([restricted, unrestricted])
        if limits is None:
            ax.set_visible(False)
            continue
        min_val, max_val = limits
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, zorder=1)

        y_r, y_r_pred = get_arrays(restricted)
        if len(y_r) > 0:
            ax.scatter(y_r, y_r_pred, s=50, alpha=1.0, color=RESTRICTED_COLOR, edgecolors="none")

        y_u, y_u_pred = get_arrays(unrestricted)
        if len(y_u) > 0:
            ax.scatter(y_u, y_u_pred, s=50, alpha=1.0, color=UNRESTRICTED_COLOR, edgecolors="none")

        ticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        mae_lines = []
        r_text = format_mae_text(restricted, "R")
        if r_text is not None:
            mae_lines.append(("R", RESTRICTED_COLOR, r_text.split("R", 1)[-1].lstrip()))
        u_text = format_mae_text(unrestricted, "U")
        if u_text is not None:
            mae_lines.append(("U", UNRESTRICTED_COLOR, u_text.split("U", 1)[-1].lstrip()))
        total_mae = pooled_mae([restricted, unrestricted])
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
        "\\begin{longtable}{|p{1.7cm}|p{1.2cm}|p{1.1cm}|p{1.4cm}|p{1.3cm}|p{3.0cm}|p{3.0cm}|p{4.8cm}|}\n"
        "\\hline\n"
        "\\textbf{Method} & \\textbf{\\# Train} & \\textbf{\\# Test} "
        "& \\textbf{Train MAE} & \\textbf{Test MAE} & \\textbf{Coeff. (R)} "
        "& \\textbf{Coeff. (U)} & \\textbf{Functional Form} \\\\ \n"
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

    lines = [header]
    for row in rows:
        restricted = row.get("restricted")
        unrestricted = row.get("unrestricted")
        base = restricted if restricted is not None else unrestricted
        if base is None:
            continue
        lines.append(
            f"{latex_escape(row['method'])} & {format_count(base.get('n_train'))} & {format_count(base.get('n_test'))} & "
            f"{format_metric(base.get('train_mae'))} & {format_metric(base.get('test_mae'))} & "
            f"{format_params(restricted['params'] if restricted else None)} & "
            f"{format_params(unrestricted['params'] if unrestricted else None)} & {base.get('func', '')} \\\\ \n\\hline\n"
        )
    lines.append("\\end{longtable}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"LaTeX table saved: {output_path}")


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
            "basis_set": "basis",
            "user_time (parse)": "time(s)",
        }
    )
    df["nbf"] = df["nocc"] + df["nvirt"]
    df["log(time(s))"] = np.log10(df["time(s)"])
    return df[df["time(s)"].notna() & (df["time(s)"] > 6)].copy()


def split_multiplicity(df, mult_col):
    if mult_col is None:
        return df.copy(), df.iloc[0:0].copy()
    restricted_mask = df[mult_col].isna() | (df[mult_col] == 1)
    unrestricted_mask = df[mult_col].notna() & (df[mult_col] != 1)
    return df.loc[restricted_mask].copy(), df.loc[unrestricted_mask].copy()


def train_test(df_pool, min_points=10):
    if len(df_pool) < min_points:
        return None, None
    return train_test_split(df_pool, test_size=0.10, random_state=21)


def fit_group(method, cfg, df_pool, group_name):
    df_train, df_test = train_test(df_pool)
    if df_train is None:
        print(f"{group_name}: Insufficient data (<10 points) — skipping fit.")
        return None

    print(f"{group_name}: Train {len(df_train)} Test {len(df_test)}")
    coeffs = optimize_method(df_train, cfg)
    print(f"  {group_name} coefficients: {coeffs}")

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

    print(f"{group_name} Train RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"{group_name} Test  RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}, MAE: {test_mae:.6f}")

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


def plot_with_predictions(df_pool, coeffs, poly_func):
    if df_pool is None or len(df_pool) == 0:
        return np.array([]), np.array([])
    y = df_pool["log(time(s))"].values
    y_pred = predict_log_time(coeffs, df_pool, poly_func)
    return y, y_pred


def run_method(method, restricted_cfg, unrestricted_cfg):
    timing_pkl = os.path.join(ROOT, DATA, f"{method}.pkl")
    print(f"Loading {method} timing data...")
    df = pd.read_pickle(timing_pkl).copy()
    mult_col = "multiplicity (parse)" if "multiplicity (parse)" in df.columns else None

    df_restricted, df_unrestricted = split_multiplicity(df, mult_col)
    df_restricted_pool = preprocess_dataframe(df_restricted)
    df_unrestricted_pool = preprocess_dataframe(df_unrestricted)

    if len(df_restricted_pool) < 10:
        print("Restricted: Insufficient data (<10 points) — skipping.")
        restricted_row = None
    else:
        restricted_row = fit_group(method, restricted_cfg, df_restricted_pool, "Restricted")

    if len(df_unrestricted_pool) < 10:
        print("Unrestricted: Insufficient data (<10 points) — skipping fit.")
        unrestricted_row = None
    else:
        unrestricted_row = fit_group(method, unrestricted_cfg, df_unrestricted_pool, "Unrestricted")

    restricted_plot = None
    if restricted_row is not None:
        restricted_plot = restricted_row

    unrestricted_plot = None
    if unrestricted_row is not None:
        unrestricted_plot = unrestricted_row
    elif len(df_unrestricted_pool) > 0 and unrestricted_cfg is not None:
        coeffs0 = np.asarray(unrestricted_cfg["coeffs"].get("guess"), dtype=float)
        if coeffs0.size:
            y_all, y_all_pred = plot_with_predictions(df_unrestricted_pool, coeffs0, unrestricted_cfg["poly"])
            unrestricted_plot = {
                "method": method,
                "y_train": y_all,
                "y_test": np.array([]),
                "y_train_pred": y_all_pred,
                "y_test_pred": np.array([]),
                "train_mae": np.nan,
                "test_mae": np.nan,
            }

    if restricted_plot is None and unrestricted_plot is None:
        return None

    plot_combined(method, restricted_plot, unrestricted_plot)

    return {
        "method": method,
        "restricted": restricted_row,
        "unrestricted": unrestricted_row,
    }


def main():
    apply_plot_style()
    combined_rows = []
    restricted_rows = []
    unrestricted_rows = []
    for method, restricted_cfg in restricted_polynomials.items():
        unrestricted_cfg = unrestricted_polynomials.get(method)
        if unrestricted_cfg is None:
            print(f"Skipping {method}: missing unrestricted polynomial config.")
            continue
        print("\n" + "=" * 60)
        print(f"METHOD: {method}")
        print("=" * 60)
        row = run_method(method, restricted_cfg, unrestricted_cfg)
        if row is None:
            continue
        combined_rows.append(row)
        if row.get("restricted") is not None:
            restricted_rows.append(row["restricted"])
        if row.get("unrestricted") is not None:
            unrestricted_rows.append(row["unrestricted"])
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
    if unrestricted_rows:
        write_latex_table(unrestricted_rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_UNRESTRICTED))
        write_inference_dataframe(
            unrestricted_rows,
            os.path.join(PLOTS_DIR, INFERENCE_DF_NAME_UNRESTRICTED),
            csv_path=os.path.join(PLOTS_DIR, INFERENCE_CSV_NAME_UNRESTRICTED),
        )
    if combined_rows:
        write_latex_table_combined(combined_rows, os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_COMBINED))
        write_binned_normalized_rmse_table(
            combined_rows,
            os.path.join(PLOTS_DIR, LATEX_TABLE_NAME_BINNED_RMSE),
        )


if __name__ == "__main__":
    main()
