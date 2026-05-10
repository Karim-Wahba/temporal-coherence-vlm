#!/bin/bash

#SBATCH --job-name=GSL-dry
#SBATCH --partition=L40Sday
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/GSL-dry-%j.out
#SBATCH --error=logs/GSL-dry-%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vlm_probe

SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/$(basename "${BASH_SOURCE[0]}")"
RESUBMITTED=0
on_usr1() {
    if [ "$RESUBMITTED" -eq 0 ]; then
        echo "[$(date)] USR1 received (≤5min to time limit) — resubmitting $SCRIPT"
        sbatch "$SCRIPT"
        RESUBMITTED=1
    fi
}
trap on_usr1 USR1

# Dry run: 1 sequence × 1 expression, all 20 variants, full per-seq grids.
# Use --image_mode to mirror your most recent grounding-stability-max run
# (drop the flag to use video mode / 3D-RoPE).
# Resume support: experiment.py reads existing results.json and skips done items.
python -u run.py \
    --davis_root /home/wahba/git/data/davis/davis/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/grounding-layers-e16 \
    --split      valid \
    --sample_rate 8 \
    --image_mode --expressions_per_seq 16 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    wait "$PID"
done
