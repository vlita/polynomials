"""
compute_timing_vars.py
----------------------
Compute quantum chemistry sizing variables for one (dataset x method) pair
and save the result as a new dataframe with four additional columns:

    n_occupied  – number of occupied orbitals
    n_virtual   – number of virtual orbitals
    np_total    – number of DFT grid points
    nbf_aux     – number of auxiliary basis functions (JKFIT or RIFIT)

Designed to be called by the cluster sbatch script (run_job.sh) with
--dataset and --method arguments. Each invocation handles exactly one
(dataset x method) pair. Multiple jobs can run simultaneously without
conflict since they write to separate output files.

Checkpoints are written every ROWS_PER_SYSTEM rows (default 6, one full
system across all basis sets) so interrupted jobs can resume cleanly.

Usage (direct):
    python compute_timing_vars.py --dataset 3acx --method HF
    python compute_timing_vars.py --dataset pepconf_90 --method MP2
    python compute_timing_vars.py --dataset pepconf_2333 --method FNO-CCSD
    python compute_timing_vars.py --dataset 3acx --method HF --threads 8
"""

import argparse
import contextlib
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import psi4
from psi4 import core

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# BASE_DIR is the directory containing this script (ice_timing_vars/).
# data/ holds the input .pkl files (copied from projects/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")

DATASETS = {
    "3acx": {
        "pkl":     os.path.join(DATA_DIR, "3acx_103+_20_in.pkl"),
        "mol_col": "qcel_dimer",
        "out_dir": os.path.join(OUT_DIR, "3acx"),
        "methods": ["HF", "B2PLYP-D3", "B3LYP-D3", "PBE-D3",
                    "wB97X-V", "M05-2X", "wB97X-D", "MP2"],
    },
    "pepconf_90": {
        "pkl":     os.path.join(DATA_DIR, "pepconf_90+_25.pkl"),
        "mol_col": "qcel_dimer",
        "out_dir": os.path.join(OUT_DIR, "pepconf_90"),
        "methods": ["HF", "B2PLYP-D3", "B3LYP-D3", "PBE-D3",
                    "wB97X-V", "M05-2X", "wB97X-D", "MP2"],
    },
    "pepconf_2333": {
        "pkl":     os.path.join(DATA_DIR, "pepconf_2333_20.pkl"),
        "mol_col": "qcel_dimer",
        "out_dir": os.path.join(OUT_DIR, "pepconf_2333"),
        "methods": ["FNO-CCSD"],
    },
}

# JKFIT methods (all others fall through to RIFIT)
JKFIT_METHODS = {"HF", "B2PLYP-D3", "B3LYP-D3", "PBE-D3",
                 "wB97X-V", "M05-2X", "wB97X-D"}

# Number of rows per system (one per basis set); checkpoint after each group
ROWS_PER_SYSTEM = 6

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str):
    """Timestamped log line — always written to stdout (captured by sbatch)."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stdout suppression (silences psi4 output + stray prints in wfn build)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


# ---------------------------------------------------------------------------
# Core computation (mirrors new_and_better_function from filter_dimers.py)
# ---------------------------------------------------------------------------
def compute_vars(mol_qcel, basis_set: str, method: str):
    """
    Build the wavefunction and DFT grid for mol_qcel at the given basis_set
    and return (n_occupied, n_virtual, np_total, nbf_aux).

    Auxiliary basis selection:
      - JKFIT  for HF and DFT methods (DF_BASIS_SCF)
      - RIFIT  for MP2, CCSD, FNO-CCSD, and anything else (DF_BASIS_MP2)
    """
    n_atoms = len(mol_qcel.atomic_numbers)
    mol = psi4.core.Molecule.from_schema(mol_qcel.dict())

    psi4.set_options({
        "basis": basis_set,
        "dft_pruning_scheme": "robust",
    })

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    bs  = wfn.basisset()
    grid = psi4.core.DFTGrid.build(mol, bs)

    n_occupied = math.ceil((wfn.nalpha() + wfn.nbeta()) / 2)
    n_virtual  = bs.nbf() - n_occupied
    np_total   = grid.npoints()

    if method in JKFIT_METHODS:
        aux_basis = core.BasisSet.build(
            wfn.molecule(),
            "DF_BASIS_SCF",
            core.get_option("SCF", "DF_BASIS_SCF"),
            "JKFIT",
            core.get_global_option("BASIS"),
        )
    else:
        aux_basis = core.BasisSet.build(
            wfn.molecule(),
            "DF_BASIS_MP2",
            core.get_option("DFMP2", "DF_BASIS_MP2"),
            "RIFIT",
            core.get_global_option("BASIS"),
        )

    nbf_aux = aux_basis.nbf()
    return n_occupied, n_virtual, np_total, nbf_aux


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process(dataset: str, method: str, n_threads: int):
    cfg     = DATASETS[dataset]
    mol_col = cfg["mol_col"]
    out_dir = cfg["out_dir"]

    safe_method = method.replace("/", "-")
    out_path    = os.path.join(out_dir, f"{dataset}_{safe_method}.pkl")
    ckpt_path   = os.path.join(out_dir, f"{dataset}_{safe_method}_checkpoint.pkl")

    os.makedirs(out_dir, exist_ok=True)

    # Skip if final output already exists
    if os.path.exists(out_path):
        log(f"[SKIP] Final output already exists: {out_path}")
        return

    log(f"[START] dataset={dataset}  method={method}  threads={n_threads}")
    log(f"        input pkl : {cfg['pkl']}")
    log(f"        output pkl: {out_path}")

    df_src = pd.read_pickle(cfg["pkl"])
    log(f"        dataframe shape: {df_src.shape}  |  mol column: '{mol_col}'")

    # Configure psi4
    psi4.core.be_quiet()
    psi4.set_num_threads(n_threads)
    log(f"        psi4 version: {psi4.__version__}  |  threads: {n_threads}")

    # Resume from checkpoint if available
    if os.path.exists(ckpt_path):
        ckpt       = pd.read_pickle(ckpt_path)
        done_count = len(ckpt)
        results    = ckpt.to_dict("records")
        log(f"[RESUME] Checkpoint found — {done_count}/{len(df_src)} rows already done")
    else:
        done_count = 0
        results    = []

    total_rows = len(df_src)

    for idx in range(done_count, total_rows):
        row   = df_src.iloc[idx]
        mol   = row[mol_col]
        basis = row["basis"]
        sys_id = row["id"]

        try:
            with suppress_stdout():
                n_occ, n_virt, np_tot, nbf_aux = compute_vars(mol, basis, method)
            log(f"  [{idx+1:>4}/{total_rows}] {sys_id:<20} {basis:<14} {method:<12} "
                f"nocc={n_occ}  nvirt={n_virt}  np={np_tot}  nbf_aux={nbf_aux}")
        except Exception as e:
            n_occ = n_virt = np_tot = nbf_aux = np.nan
            log(f"  [{idx+1:>4}/{total_rows}] {sys_id:<20} {basis:<14} {method:<12} "
                f"ERROR: {e}")

        record = row.to_dict()
        record["method"]     = method
        record["n_occupied"] = n_occ
        record["n_virtual"]  = n_virt
        record["np_total"]   = np_tot
        record["nbf_aux"]    = nbf_aux
        results.append(record)

        # Checkpoint every ROWS_PER_SYSTEM rows (after each system finishes all bases)
        if (idx + 1) % ROWS_PER_SYSTEM == 0:
            pd.DataFrame(results).to_pickle(ckpt_path)
            log(f"  [CHECKPOINT] {idx+1}/{total_rows} rows saved")

    df_out = pd.DataFrame(results)
    df_out.to_pickle(out_path)
    log(f"[DONE] Saved {len(df_out)} rows -> {out_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        log(f"[CLEANUP] Checkpoint removed: {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    valid_datasets = list(DATASETS.keys())
    parser = argparse.ArgumentParser(
        description="Compute psi4 timing variables for one (dataset x method) pair."
    )
    parser.add_argument(
        "--dataset", required=True, choices=valid_datasets,
        help=f"Dataset to process. One of: {valid_datasets}"
    )
    parser.add_argument(
        "--method", required=True,
        help="Method name (e.g. HF, MP2, FNO-CCSD). Must be valid for the chosen dataset."
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Number of threads for psi4 (default: 1). Match to --cpus-per-task in sbatch."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate method against dataset
    valid_methods = DATASETS[args.dataset]["methods"]
    if args.method not in valid_methods:
        print(f"ERROR: method '{args.method}' is not valid for dataset '{args.dataset}'.")
        print(f"       Valid methods: {valid_methods}")
        sys.exit(1)

    process(args.dataset, args.method, args.threads)


if __name__ == "__main__":
    main()
