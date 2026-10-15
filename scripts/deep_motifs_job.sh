#!/bin/bash
#SBATCH --job-name=deepmotifs
#SBATCH --partition=GPU
#SBATCH --nodelist=totoro            # NVIDIA node (venv torch is a CUDA/cu130 build)
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8            # XGBoost prior uses CPU cores
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=run_sfari_gpu.%j.log

cd /home/njchu/Deep-MOTIFs
module load AI/cuda13.0

.venv/bin/python -m deep_motifs \
  --project-root /home/njchu/Deep-MOTIFs \
  --labels-dir /home/njchu/Deep-MOTIFs/ext_data/labels \
  --output-dir deep_motifs_sfari_gpu_outputs \
  --prior-model xgb \
  --fusion-mode rrf \
  --ppr-alpha 1.0 \
  --pu-class-prior 0.06 \
  --prior-uncertainty-delta 0.05 \
  --prior-weight-floor 0.10 \
  --prior-guided-calibration rank \
  --random-state 42 \
  --device auto
