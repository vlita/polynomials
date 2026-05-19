---
name: select-lot
description: Use this skill ONLY when the user explicitly says `select-LoT`. It parses a folder of dimer geometry files, builds an inference-ready dataframe, predicts interaction-energy errors and timing for all levels of theory (10 methods x 6 basis sets by default), filters by user walltime, and recommends the most accurate feasible level of theory. Do not trigger for general quantum chemistry questions or generic LoT selection unless `select-LoT` is stated.
---

# select-LoT

Build the workflow directly from `ie_and_time_predictor.py` and keep its logic intact unless the user explicitly asks for changes. The goal is to parse a folder of geometry files, build the inference dataframe, run interaction-energy error and timing estimators, filter by walltime, and save the dataframe. The user is a novice quantum chemist, so explain assumptions and report a clear recommendation.

## Inputs you must ask for
Ask these up front if not provided:
- Geometry folder path (folder containing dimer geometries in QCElemental/Psi4 string format).
- Walltime limit for a single calculation (ask for units; default to hours if they do not specify).
- Output path for the saved dataframe (do not assume a default; prompt until the user provides a path).

If the user provides a list of methods/basis sets or a subset of LoTs, use that; otherwise, use all 10 methods and 6 basis sets from the script.

## Default LoT grid (from script)
Methods:
- HF
- PBE-D3
- wB97X-D
- wB97X-V
- MP2
- B3LYP-D3
- B2PLYP-D3
- M05-2X
- FNO-CCSD
- FNO-CCSD(T)

Basis sets:
- cc-pVDZ
- cc-pVTZ
- cc-pVQZ
- aug-cc-pVQZ
- aug-cc-pVTZ
- aug-cc-pVDZ

## Workflow (mirror `ie_and_time_predictor.py`)
1) Parse geometry files with the same logic as `parse_geoms`.
   - Treat files as dimers and ensure two fragments are present.
   - Preserve the warnings about missing units or charge/multiplicity.
2) Build inference dataframe using `build_inference_table`.
   - Compute timing variables for dimer and monomers per basis set.
   - Expand for method x basis combinations; add `Level of Theory` strings.
3) Predict interaction-energy error with `predict_ie_errors_batch`.
4) Predict timing with `predict_timings_batch`.
5) Convert walltime to seconds and filter LoTs where
   `10 ** ESTIMATED CPU TIMES (log10(s)) <= walltime_seconds`.
6) Recommend a final LoT that yields the lowest predicted error for the largest fraction of systems that pass the walltime filter.
   - If multiple LoTs are close, prefer the one with broader feasibility.
   - If none pass, report that and ask for a larger walltime.
7) Save the full dataframe (including unfiltered rows) to the user-specified path.

## Output requirements
Always produce both:
- A saved dataframe at the requested path.
- A concise textual summary that includes:
  - Number of systems parsed
  - Walltime and units
  - Count of feasible LoTs
  - Recommended LoT and why
  - Any warnings encountered during parsing or estimation

## Implementation notes
- Use the existing environment dependencies (psi4, qcelemental, apnet_pt, pandas, numpy).
- If you need to copy or adapt the script, keep function names and column names unchanged.
- Do not change the estimator models or polynomial expressions.
- Ensure output is ASCII.
