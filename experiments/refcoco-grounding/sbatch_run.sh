#!/bin/bash

#SBATCH --job-name=RC-QWEN
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/RC-QWEN-%j.out
#SBATCH --error=logs/RC-QWEN-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

python -u run.py \
    --refcoco_root /home/wahba/git/data/refcoco \
    --dataset      refcoco \
    --split        val \
    --model_id     Qwen/Qwen3-VL-8B-Instruct \
    --save_dir     results/refcoco_val \
    --sents_per_ref 1 \
    --vis_every    50
