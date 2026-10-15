#!/bin/bash
# =============================================================================
#  SLURM batch job: run the COMPLETE benchmark pipeline on a GPU node.
#
#  Runs run_pipeline.py with no seed argument, i.e. the full paper sweep:
#    5 seeds (run_1..run_5), each: forecASD -> 10 models (xgb, rf, sv, deepgbm,
#    sai, tab, ftt, cnn, gcn, deep_motifs) -> then average_runs + MCC metrics.
#
#  The torch models (deepgbm, sai, tab, ftt, cnn, gcn, deep_motifs) default to
#  --device auto, so they use the GPU automatically on this node. xgb/rf/sv and
#  forecASD are CPU (xgboost/sklearn) and use --cpus-per-task cores.
#
#  Submit FROM THE REPOSITORY ROOT:
#                 sbatch scripts/run_full_pipeline_job.sh
#  Watch:         squeue -u $USER        and     tail -f logs/full_pipeline.<jobid>.out
#  Cancel:        scancel <jobid>
#
#  Being an sbatch job, it keeps running on the compute node after you log out.
#  Edit the #SBATCH --partition / --nodelist / --gres lines below for your cluster.
# =============================================================================
#SBATCH --job-name=dm_full
#SBATCH --partition=GPU
#SBATCH --nodelist=totoro            # NVIDIA node (venv torch is a CUDA/cu130 build)
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8            # forecASD + xgb/rf/sv use CPU cores
#SBATCH --mem=64G                    # forecASD/feature matrices live in system RAM
#SBATCH --time=48:00:00             # full 5x11 sweep; lower to your partition's max
#SBATCH --output=logs/full_pipeline.%j.out   # combined stdout+stderr (all step logs)

set -uo pipefail

# Repository root = the directory you ran `sbatch` from (submit from the repo root).
PROJECT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT" || exit 1
mkdir -p logs

module load AI/cuda13.0

# Stream logs live (so `tail -f` shows progress) and match CPU threads to the
# allocation for the CPU-bound models.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DRIVER_LOG="logs/full_pipeline.${SLURM_JOB_ID}.driver.log"

echo "=================================================================="
echo " Job $SLURM_JOB_ID on $(hostname)  started $(date)"
echo "=================================================================="
nvidia-smi || echo "[WARN] nvidia-smi unavailable"

# --- Optional per-step overrides -------------------------------------------
# GPU VRAM is usually smaller than system RAM. The SAINT model (sai) was the
# step that OOM'd at ~59 GB on CPU; if it CUDA-OOMs on this GPU, cap it or pin
# it to CPU by uncommenting ONE of these lines:
# SAINT (sai) needs ~59 GB at the default batch of 128 — far more than totoro's
# 24 GB (RTX 3090) VRAM. Batch 32 drops that to ~15 GB so it fits the GPU.
EXTRA_ARGS=(--extra "sai:--batch-size 32")
# EXTRA_ARGS=(--extra "sai:--batch-size 16")     # smaller margin if 32 still CUDA-OOMs
# EXTRA_ARGS=(--extra "sai:--device cpu")        # or keep SAINT on CPU (raise --mem to 96G)
# ---------------------------------------------------------------------------

.venv/bin/python run_pipeline.py \
  --project-root "$PROJECT" \
  --log "$DRIVER_LOG" \
  "${EXTRA_ARGS[@]}"
STATUS=$?

echo "=================================================================="
echo " Job $SLURM_JOB_ID finished $(date) with exit code $STATUS"
echo "=================================================================="
exit $STATUS
