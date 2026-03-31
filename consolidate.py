"""
consolidate.py

Builds one per-method pickle file combining:
  - phoenix/complete/   : actual timing outputs + psi4 output files
  - ds5/consolidated_dataframe_new_vars.pkl : correct timing vars for old (batch_0_*) datasets
  - ice_timing_vars/output/ : correct timing vars for new datasets (3acx, pepconf_90, pepconf_2333)

Output: output/{method}.pkl  (12 files, one per method)

Target schema per file:
  id, basis_set, Geometry, qcel_dimer, dimer_charge, dimer_multiplicity, n_atoms,
  qcfractal id, job status,
  n_occupied, n_virtual, np_total, nbf_aux,
  wall_time (provenance), wall_time (parse), total_time (parse),
  user_time (parse), memory_GB (provenance),
  psi4 output
"""

import os
import glob
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PHOENIX = os.path.join(BASE, "phoenix", "complete")
DS5_PATH = os.path.join(BASE, "ds5", "consolidated_dataframe_new_vars.pkl")
ICE_3ACX = os.path.join(BASE, "ice_timing_vars", "output", "3acx", "3acx_HF.pkl")
ICE_PEP90_DIR = os.path.join(BASE, "ice_timing_vars", "output", "pepconf_90")
ICE_PEP2333 = os.path.join(BASE, "ice_timing_vars", "output", "pepconf_2333", "pepconf_2333_FNO-CCSD.pkl")
OUT_DIR = os.path.join(BASE, "output")

METHODS = [
    "HF", "MP2", "CCSD", "CCSD(T)",
    "FNO-CCSD", "FNO-CCSD(T)",
    "B2PLYP-D3", "B3LYP-D3", "M05-2X",
    "PBE-D3", "wB97X-D", "wB97X-V",
]

# Dataset prefixes for old (batch_0) vs new datasets
OLD_PREFIXES = [
    "batch_0_1015_s20",
    "batch_0_1520_s20",
    "batch_0_2024_s32",
    "batch_0_2738_s37",
    "batch_0_3942_s8",
    "batch_0_3942_s13",
    "batch_0_3942_s52",
    "batch_0_3942_s94",
    "batch_0_3942_sr",
]

NEW_PREFIXES = {
    "3acx": "3acx_103+_20",
    "pepconf_90": "pepconf_90+_25",
    "pepconf_2333": "pepconf_2333_20",
}

# Final column order
TARGET_COLS = [
    "id",
    "basis_set",
    "Geometry",
    "qcel_dimer",
    "dimer_charge",
    "dimer_multiplicity",
    "n_atoms",
    "qcfractal id",
    "job status",
    "n_occupied",
    "n_virtual",
    "np_total",
    "nbf_aux",
    "wall_time (provenance)",
    "wall_time (parse)",
    "total_time (parse)",
    "user_time (parse)",
    "memory_GB (provenance)",
    "psi4 output",
]

# Columns to drop (not needed in output)
DROP_COLS = [
    "charges", "monAs", "monBs",
    "nocc", "nvirt", "nbf",
    "main_id", "System #", "System", "DB",
    "qcel_monA", "qcel_monB",
    "monA_charge", "monB_charge",
    "monA_multiplicity", "monB_multiplicity",
    "# atoms",
    "folder",
]


def load_timing_lookup_ds5():
    """
    Load ds5 timing vars. Deduplicate on (id, basis_set) since timing vars
    are method-independent. Returns a DataFrame indexed on (id, basis_set).
    """
    df = pd.read_pickle(DS5_PATH)
    cols = ["id", "basis_set", "n_occupied", "n_virtual", "np_total", "nbf_aux"]
    lookup = df[cols].drop_duplicates(subset=["id", "basis_set"]).copy()
    lookup = lookup.set_index(["id", "basis_set"])
    print(f"  DS5 timing lookup: {len(lookup)} unique (id, basis_set) pairs")
    return lookup


def load_timing_lookup_3acx():
    """
    Load 3acx timing vars from the complete HF ice file.
    Timing vars are basis-dependent only, so HF file covers all methods.
    Returns DataFrame indexed on (id, basis_set).
    """
    df = pd.read_pickle(ICE_3ACX)
    cols = ["id", "basis", "n_occupied", "n_virtual", "np_total", "nbf_aux"]
    lookup = df[cols].copy()
    lookup = lookup.rename(columns={"basis": "basis_set"})
    lookup = lookup.drop_duplicates(subset=["id", "basis_set"])
    lookup = lookup.set_index(["id", "basis_set"])
    print(f"  3acx timing lookup: {len(lookup)} unique (id, basis_set) pairs")
    return lookup


def load_timing_lookup_pepconf90():
    """
    Load pepconf_90 timing vars. Use HF file (all methods have same timing vars).
    Returns DataFrame indexed on (id, basis_set).
    """
    hf_file = os.path.join(ICE_PEP90_DIR, "pepconf_90_HF.pkl")
    df = pd.read_pickle(hf_file)
    cols = ["id", "basis", "n_occupied", "n_virtual", "np_total", "nbf_aux"]
    lookup = df[cols].copy()
    lookup = lookup.rename(columns={"basis": "basis_set"})
    lookup = lookup.drop_duplicates(subset=["id", "basis_set"])
    lookup = lookup.set_index(["id", "basis_set"])
    print(f"  pepconf_90 timing lookup: {len(lookup)} unique (id, basis_set) pairs")
    return lookup


def load_timing_lookup_pepconf2333():
    """
    Load pepconf_2333 timing vars from the FNO-CCSD ice file.
    Returns DataFrame indexed on (id, basis_set).
    """
    df = pd.read_pickle(ICE_PEP2333)
    cols = ["id", "basis", "n_occupied", "n_virtual", "np_total", "nbf_aux"]
    lookup = df[cols].copy()
    lookup = lookup.rename(columns={"basis": "basis_set"})
    lookup = lookup.drop_duplicates(subset=["id", "basis_set"])
    lookup = lookup.set_index(["id", "basis_set"])
    print(f"  pepconf_2333 timing lookup: {len(lookup)} unique (id, basis_set) pairs")
    return lookup


def rename_method_timing_cols(df, method):
    """
    Rename {method}_wall_time (provenance) etc. to generic names.
    Also rename {method}_set_memory_GB (provenance) -> memory_GB (provenance).
    """
    rename_map = {
        f"{method}_wall_time (provenance)": "wall_time (provenance)",
        f"{method}_wall_time (parse)":      "wall_time (parse)",
        f"{method}_total_time (parse)":     "total_time (parse)",
        f"{method}_user_time (parse)":      "user_time (parse)",
        f"{method}_set_memory_GB (provenance)": "memory_GB (provenance)",
    }
    return df.rename(columns=rename_map)


def drop_predicted_cols(df):
    """Drop all *_predicted_* columns."""
    pred_cols = [c for c in df.columns if "_predicted_" in c]
    return df.drop(columns=pred_cols, errors="ignore")


def process_old_batch_file(fpath, method, ds5_lookup):
    """
    Process a single old-dataset (batch_0_*) phoenix file.

    - id = qcfractal id (preserved as the primary identifier)
    - basis -> basis_set
    - rename method-specific timing cols to generic names
    - join correct np_total, nbf_aux (and n_occupied, n_virtual) from ds5_lookup
    - ensure psi4 output column exists (NaN if absent)
    """
    df = pd.read_pickle(fpath)
    if df.empty:
        return None

    df = df.copy()

    # Rename basis
    df = df.rename(columns={"basis": "basis_set"})

    # Rename method timing cols
    df = rename_method_timing_cols(df, method)

    # Drop predicted and unwanted cols
    df = drop_predicted_cols(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Ensure psi4 output column
    if "psi4 output" not in df.columns:
        df["psi4 output"] = pd.NA

    # Set id = qcfractal id (preserve qcfractal id column separately)
    # The 'id' column in old batch files is the molecular/system ID (small integer)
    # The 'qcfractal id' is the QCFractal record ID — this is the primary ID we want
    df["id"] = df["qcfractal id"]

    # Join timing vars from ds5 (correct np_total, nbf_aux, n_occupied, n_virtual)
    # Drop the (potentially wrong) nocc/nvirt/np_total/nbf_aux already in df
    # (nocc/nvirt already match ds5 n_occupied/n_virtual but we standardise names)
    for col in ["nocc", "nvirt", "np_total", "nbf_aux", "n_occupied", "n_virtual"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.reset_index(drop=True)
    df = df.join(
        ds5_lookup[["n_occupied", "n_virtual", "np_total", "nbf_aux"]],
        on=["id", "basis_set"],
        how="left",
    )

    return df


def process_new_dataset_file(fpath, method, timing_lookup):
    """
    Process a single new-dataset (3acx / pepconf_90 / pepconf_2333) phoenix file.

    - id = string identifier (e.g. '3ACX_103', 'GPQGIAG_0')
    - basis -> basis_set
    - rename method-specific timing cols
    - join timing vars from ice lookup
    - ensure psi4 output column exists
    """
    df = pd.read_pickle(fpath)
    if df.empty:
        return None

    df = df.copy()

    # Rename basis
    df = df.rename(columns={"basis": "basis_set"})

    # Rename method timing cols
    df = rename_method_timing_cols(df, method)

    # Drop unwanted cols
    df = drop_predicted_cols(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Ensure psi4 output column
    if "psi4 output" not in df.columns:
        df["psi4 output"] = pd.NA

    # New datasets don't have n_occupied etc. — join from ice timing lookup
    df = df.reset_index(drop=True)
    df = df.join(
        timing_lookup[["n_occupied", "n_virtual", "np_total", "nbf_aux"]],
        on=["id", "basis_set"],
        how="left",
    )

    return df


def build_pepconf2333_rows(method, pep2333_lookup):
    """
    For pepconf_2333: the phoenix file is empty (no completed jobs).
    Build rows from the ice timing var lookup with all output columns as NaN.
    Only relevant for FNO-CCSD.
    """
    if method != "FNO-CCSD":
        return None

    df = pep2333_lookup.reset_index().copy()
    # df has: id, basis_set, n_occupied, n_virtual, np_total, nbf_aux
    # All output columns will be NaN
    for col in [
        "Geometry", "qcel_dimer", "dimer_charge", "dimer_multiplicity", "n_atoms",
        "qcfractal id", "job status",
        "wall_time (provenance)", "wall_time (parse)", "total_time (parse)",
        "user_time (parse)", "memory_GB (provenance)", "psi4 output",
    ]:
        df[col] = pd.NA

    return df


def reorder_columns(df):
    """Reorder to target schema; add missing columns as NaN."""
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    # Keep only target cols (plus any extras not in schema, appended at end)
    extra = [c for c in df.columns if c not in TARGET_COLS]
    return df[TARGET_COLS + extra]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading timing var lookups...")
    ds5_lookup = load_timing_lookup_ds5()
    lookup_3acx = load_timing_lookup_3acx()
    lookup_pep90 = load_timing_lookup_pepconf90()
    lookup_pep2333 = load_timing_lookup_pepconf2333()
    print()

    for method in METHODS:
        print(f"=== Building {method} ===")
        frames = []

        # --- OLD DATASETS (batch_0_*) ---
        for prefix in OLD_PREFIXES:
            fname = f"{prefix}_{method}_completed.pkl"
            fpath = os.path.join(PHOENIX, fname)
            if not os.path.exists(fpath):
                continue
            df = process_old_batch_file(fpath, method, ds5_lookup)
            if df is not None and not df.empty:
                df["_source"] = f"{prefix}"
                frames.append(df)
                print(f"  {fname}: {len(df)} rows")
            else:
                print(f"  {fname}: empty, skipped")

        # --- NEW DATASET: 3acx ---
        fname_3acx = f"3acx_103+_20_{method}_completed.pkl"
        fpath_3acx = os.path.join(PHOENIX, fname_3acx)
        if os.path.exists(fpath_3acx):
            df = process_new_dataset_file(fpath_3acx, method, lookup_3acx)
            if df is not None and not df.empty:
                df["_source"] = "3acx_103+_20"
                frames.append(df)
                print(f"  {fname_3acx}: {len(df)} rows")
            else:
                print(f"  {fname_3acx}: empty, skipped")

        # --- NEW DATASET: pepconf_90 ---
        fname_pep90 = f"pepconf_90+_25_{method}_completed.pkl"
        fpath_pep90 = os.path.join(PHOENIX, fname_pep90)
        if os.path.exists(fpath_pep90):
            df = process_new_dataset_file(fpath_pep90, method, lookup_pep90)
            if df is not None and not df.empty:
                df["_source"] = "pepconf_90+_25"
                frames.append(df)
                print(f"  {fname_pep90}: {len(df)} rows")
            else:
                print(f"  {fname_pep90}: empty, skipped")

        # --- NEW DATASET: pepconf_2333 (FNO-CCSD only, phoenix empty) ---
        fname_pep2333 = f"pepconf_2333_20_{method}_completed.pkl"
        fpath_pep2333 = os.path.join(PHOENIX, fname_pep2333)
        if os.path.exists(fpath_pep2333):
            # Phoenix file exists but is empty — build rows from ice timing vars
            df = build_pepconf2333_rows(method, lookup_pep2333)
            if df is not None and not df.empty:
                df["_source"] = "pepconf_2333_20"
                frames.append(df)
                print(f"  pepconf_2333_20 (ice timing only): {len(df)} rows")

        if not frames:
            print(f"  No data found for {method}, skipping output.")
            print()
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = reorder_columns(combined)

        out_path = os.path.join(OUT_DIR, f"{method}.pkl")
        combined.to_pickle(out_path)
        print(f"  -> Wrote {out_path}: {len(combined)} total rows, {len(combined.columns)} cols")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
