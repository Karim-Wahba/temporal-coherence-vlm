#!/bin/bash

#SBATCH --job-name=GS-QWEN-e16
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/GS-QWEN-e16-%j.out
#SBATCH --error=logs/GS-QWEN-e16-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

python -u  run.py \
    --davis_root /home/wahba/git/data/davis/davis/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/grounding_stability_final_16_expressions \
    --split      valid \
    --sample_rate 8 \
    --image_mode \
    --expressions_per_seq 16
