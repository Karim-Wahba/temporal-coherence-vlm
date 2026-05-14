#!/bin/bash

#SBATCH --job-name=OPRO-META-full
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/OPRO-META-full%j.out
#SBATCH --error=logs/OPRO-META-full%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source /etc/profile.d/conda.sh
conda activate vlm_probe

mkdir -p logs

OUT_DIR=results/opro_meta_full

# No --max_clips → iterate over every (seq, obj) pair in Ref-DAVIS valid split.
# --skip_tam halves per-iteration wall time (no MassGT/MassPred, IoU only).
# Drop --skip_tam if you need the attention-mass metrics for analysis.
/home/geiger/gwb913/.conda/envs/vlm_probe/bin/python -u main.py \
    --config configs/default.yaml \
    --out_dir $OUT_DIR \
    --skip_tam

/home/geiger/gwb913/.conda/envs/vlm_probe/bin/python -u plot_results.py \
    --results $OUT_DIR/clip_results.json \
    --out_dir $OUT_DIR/figures
