#!/bin/bash

#SBATCH --job-name=EXPR-VAR-RYTVOS
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/EXPR-VAR-RYTVOS-%j.out
#SBATCH --error=logs/EXPR-VAR-RYTVOS-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

# Determinism env vars (also set inside QwenVOTRunner via _seed_all, but
# pinning here makes intent obvious).
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u run.py \
    --data_root  /home/wahba/git/data/ref-youtube-vos \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/expression_variance \
    --split      valid \
    --sample_rate 2 \
    --seed       0 \
    --min_expressions_per_group 2
