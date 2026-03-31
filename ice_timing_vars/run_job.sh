#!/bin/bash
# =============================================================================
# run_job.sh — Generic sbatch wrapper for compute_timing_vars.py
#
# Submit with:
#   sbatch --job-name=tv_<DATASET>_<METHOD> \
#          --output=logs/<DATASET>_<METHOD>.out \
#          --error=logs/<DATASET>_<METHOD>.err \
#          run_job.sh <DATASET> <METHOD> [THREADS]
#
# Examples:
#   sbatch --job-name=tv_3acx_HF --output=logs/3acx_HF.out --error=logs/3acx_HF.err run_job.sh 3acx HF 8
#   sbatch --job-name=tv_pepconf90_MP2 --output=logs/pepconf_90_MP2.out --error=logs/pepconf_90_MP2.err run_job.sh pepconf_90 MP2 8
#   sbatch --job-name=tv_pep2333_FNOCCSD --output=logs/pepconf_2333_FNO-CCSD.out --error=logs/pepconf_2333_FNO-CCSD.err run_job.sh pepconf_2333 FNO-CCSD 8
# =============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --constraint=gold6226
#SBATCH --mail-type=END                         # Mail preferences
#SBATCH --mail-user=vlita3@gatech.edu           # E-mail address for notifications

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
DATASET=${1:?"Usage: sbatch run_job.sh <DATASET> <METHOD> [THREADS]"}
METHOD=${2:?"Usage: sbatch run_job.sh <DATASET> <METHOD> [THREADS]"}
THREADS=${3:-8}

echo "========================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "Dataset     : $DATASET"
echo "Method      : $METHOD"
echo "Threads     : $THREADS"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load anaconda3
conda activate p4_qca

export SCRATCH=${TMPDIR}
export PSI_SCRATCH=${TMPDIR}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# cd to the directory containing this script so relative paths in
# compute_timing_vars.py resolve correctly
cd $SLURM_SUBMIT_DIR

python compute_timing_vars.py \
    --dataset "$DATASET" \
    --method  "$METHOD"  \
    --threads "$THREADS"

EXIT_CODE=$?

echo "========================================"
echo "Job finished: $(date)"
echo "Exit code   : $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE