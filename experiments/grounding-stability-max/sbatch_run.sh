#!/bin/bash

#SBATCH --job-name=GSM-QWEN-e16-max-pred
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/GSM-QWEN-e16-max-pred-%j.out
#SBATCH --error=logs/GSM-QWEN-e16-max-pred-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

python -u run.py \
    --davis_root /home/wahba/git/data/davis/davis/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/grounding_stability_max_best_pred \
    --split      valid \
    --sample_rate 8 \
    --image_mode \
    --expressions_per_seq 16 \
    --token_selection best_pred
