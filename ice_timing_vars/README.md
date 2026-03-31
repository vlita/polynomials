# ice_timing_vars

Computes psi4 quantum chemistry sizing variables for all (dataset x method)
combinations and stores results as new dataframes alongside the originals.

## Directory layout

```
ice_timing_vars/
  compute_timing_vars.py   # main script — one invocation per (dataset x method)
  run_job.sh               # sbatch wrapper — submit this
  data/                    # place input .pkl files here (see Step 1)
  output/
    3acx/                  # results for 3acx_103+_20_in.pkl
    pepconf_90/            # results for pepconf_90+_25.pkl
    pepconf_2333/          # results for pepconf_2333_20.pkl
  logs/                    # sbatch stdout/stderr (created on submit)
```

## Output columns added to each dataframe

| Column       | Description                                    |
|--------------|------------------------------------------------|
| `method`     | method string used for this run                |
| `n_occupied` | number of occupied orbitals                    |
| `n_virtual`  | number of virtual orbitals                     |
| `np_total`   | number of DFT grid points                      |
| `nbf_aux`    | auxiliary basis functions (JKFIT or RIFIT)     |

Auxiliary basis selection follows `new_and_better_function` in filter_dimers.py:
- **JKFIT** (DF_BASIS_SCF): HF, B2PLYP-D3, B3LYP-D3, PBE-D3, wB97X-V, M05-2X, wB97X-D
- **RIFIT** (DF_BASIS_MP2): MP2, FNO-CCSD, and any other method

---

## Step 1 — One-time setup: copy input data and configure the sbatch script

Copy the three input pickle files into `data/`:

```bash
cp /path/to/projects/3acx_103+_20_in.pkl          ice_timing_vars/data/
cp /path/to/projects/pepconf/pepconf_90+_25.pkl    ice_timing_vars/data/
cp /path/to/projects/pepconf/pepconf_2333_20.pkl   ice_timing_vars/data/
```

Edit `run_job.sh` and fill in the two TODOs:
1. Replace `module load REPLACE_WITH_YOUR_MODULE` with the correct module(s)
2. Replace `p4_qca` with your actual conda environment name if different

---

## Step 2 — Submit jobs manually

Each sbatch command submits one (dataset x method) job. All 17 can be
submitted at the same time — they write to separate files with no conflicts.

cd into the `ice_timing_vars/` directory before submitting:

```bash
cd ice_timing_vars/
```

### 3acx — 8 jobs

```bash
sbatch --job-name=tv_3acx_HF         --output=logs/3acx_HF.out         --error=logs/3acx_HF.err         run_job.sh 3acx HF          8
sbatch --job-name=tv_3acx_B2PLYPD3   --output=logs/3acx_B2PLYP-D3.out  --error=logs/3acx_B2PLYP-D3.err  run_job.sh 3acx B2PLYP-D3   8
sbatch --job-name=tv_3acx_B3LYPD3    --output=logs/3acx_B3LYP-D3.out   --error=logs/3acx_B3LYP-D3.err   run_job.sh 3acx B3LYP-D3    8
sbatch --job-name=tv_3acx_PBED3      --output=logs/3acx_PBE-D3.out      --error=logs/3acx_PBE-D3.err     run_job.sh 3acx PBE-D3      8
sbatch --job-name=tv_3acx_wB97XV     --output=logs/3acx_wB97X-V.out     --error=logs/3acx_wB97X-V.err    run_job.sh 3acx wB97X-V     8
sbatch --job-name=tv_3acx_M052X      --output=logs/3acx_M05-2X.out      --error=logs/3acx_M05-2X.err     run_job.sh 3acx M05-2X      8
sbatch --job-name=tv_3acx_wB97XD     --output=logs/3acx_wB97X-D.out     --error=logs/3acx_wB97X-D.err    run_job.sh 3acx wB97X-D     8
sbatch --job-name=tv_3acx_MP2        --output=logs/3acx_MP2.out         --error=logs/3acx_MP2.err         run_job.sh 3acx MP2         8
```

### pepconf_90 — 8 jobs

```bash
sbatch --job-name=tv_pep90_HF        --output=logs/pepconf_90_HF.out        --error=logs/pepconf_90_HF.err        run_job.sh pepconf_90 HF        8
sbatch --job-name=tv_pep90_B2PLYPD3  --output=logs/pepconf_90_B2PLYP-D3.out --error=logs/pepconf_90_B2PLYP-D3.err run_job.sh pepconf_90 B2PLYP-D3 8
sbatch --job-name=tv_pep90_B3LYPD3   --output=logs/pepconf_90_B3LYP-D3.out  --error=logs/pepconf_90_B3LYP-D3.err  run_job.sh pepconf_90 B3LYP-D3  8
sbatch --job-name=tv_pep90_PBED3     --output=logs/pepconf_90_PBE-D3.out     --error=logs/pepconf_90_PBE-D3.err    run_job.sh pepconf_90 PBE-D3    8
sbatch --job-name=tv_pep90_wB97XV    --output=logs/pepconf_90_wB97X-V.out    --error=logs/pepconf_90_wB97X-V.err   run_job.sh pepconf_90 wB97X-V   8
sbatch --job-name=tv_pep90_M052X     --output=logs/pepconf_90_M05-2X.out     --error=logs/pepconf_90_M05-2X.err    run_job.sh pepconf_90 M05-2X    8
sbatch --job-name=tv_pep90_wB97XD    --output=logs/pepconf_90_wB97X-D.out    --error=logs/pepconf_90_wB97X-D.err   run_job.sh pepconf_90 wB97X-D   8
sbatch --job-name=tv_pep90_MP2       --output=logs/pepconf_90_MP2.out        --error=logs/pepconf_90_MP2.err       run_job.sh pepconf_90 MP2       8
```

### pepconf_2333 — 1 job

```bash
sbatch --job-name=tv_pep2333_FNOCCSD --output=logs/pepconf_2333_FNO-CCSD.out --error=logs/pepconf_2333_FNO-CCSD.err run_job.sh pepconf_2333 FNO-CCSD 8
```

---

## Monitoring

```bash
# Check queue status
squeue -u $USER

# Tail a specific log
tail -f logs/3acx_HF.out

# Check which output files have been created
ls -lh output/3acx/ output/pepconf_90/ output/pepconf_2333/
```

## Resuming interrupted jobs

If a job is killed (walltime, OOM, etc.) a checkpoint file will exist in the
output directory (e.g. `output/3acx/3acx_HF_checkpoint.pkl`). Simply resubmit
the same sbatch command — the script detects the checkpoint and resumes from
where it left off. No data is lost.

## Adjusting thread count

The third argument to `run_job.sh` sets `--threads` (default 8) which is
passed to `psi4.set_num_threads()`. Make sure `--cpus-per-task` in `run_job.sh`
matches this value. Psi4's wavefunction build and DFT grid construction
parallelise well with OpenMP.
