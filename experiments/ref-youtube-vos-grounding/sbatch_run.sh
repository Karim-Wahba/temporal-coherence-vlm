#!/bin/bash

#SBATCH --job-name=RYTVOS-GS
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/RYTVOS-GS-%j.out
#SBATCH --error=logs/RYTVOS-GS-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

python -u run.py \
    --data_root   /home/wahba/git/data/ref-youtube-vos \
    --model_id    Qwen/Qwen3-VL-8B-Instruct \
    --save_dir    results/rytvos_valid \
    --split       valid \
    --sample_rate 2 \
    --expressions_per_seq 4
