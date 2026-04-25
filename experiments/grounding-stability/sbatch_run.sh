#!/bin/bash

#SBATCH --job-name=GS-QWEN
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=/mnt/lustre/home/geiger/gwb913/git/temporal-coherence-vlm/experiments/grounding-stability/logs/QWEN-%j.out
#SBATCH --error=/mnt/lustre/home/geiger/gwb913/git/temporal-coherence-vlm/experiments/grounding-stability/logs/QWEN-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source /etc/profile.d/conda.sh
conda activate vlm_probe

cd /mnt/lustre/home/geiger/gwb913/git/temporal-coherence-vlm/experiments/grounding-stability

srun /home/geiger/gwb913/.conda/envs/vlm_probe/bin/python3 run.py \
    --davis_root /mnt/lustre/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/grounding_stability_final \
    --split      valid \
    --sample_rate 8 \
    --image_mode \
    --expressions_per_seq 4
