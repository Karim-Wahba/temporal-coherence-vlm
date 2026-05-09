#!/bin/bash

#SBATCH --job-name=POS-DOM-QWEN
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/pos_dominance-%j.out
#SBATCH --error=logs/pos_dominance-%j.err

mkdir -p logs

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

python -u run.py \
    --davis_root /home/wahba/git/data/davis/davis/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/pos_dominance \
    --split      valid \
    --sample_rate 8 \
    --expressions_per_seq 16
