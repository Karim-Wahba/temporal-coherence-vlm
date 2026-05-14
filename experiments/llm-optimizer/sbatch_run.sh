#!/bin/bash

#SBATCH --job-name=LLM-OPT-QWEN-headroom-analysis
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/LLM-OPT-QWEN-headroom-analysis%j.out
#SBATCH --error=logs/LLM-OPT-QWEN-headroom-analysis%j.err

scontrol show job $SLURM_JOB_ID
nvidia-smi

source /etc/profile.d/conda.sh
conda activate vlm_probe

mkdir -p logs

OUT_DIR=results/llm_optimizer_qwenanalysis

/home/geiger/gwb913/.conda/envs/vlm_probe/bin/python -u poc.py \
    --results_json  ../expression-variance/results/expression_variance/results.json \
    --grouped_json  ../expression-variance/results/expression_variance/grouped_stats.json \
    --model_id      Qwen/Qwen3-VL-8B-Instruct \
    --top_n_groups  5 \
    --n_candidates  3 \
    --n_iterations  2 \
    --select_by     headroom \
    --eval          real \
    --davis_root    /home/geiger/gwb913/git/davis/DAVIS2017/unsupervised \
    --out_dir       $OUT_DIR

/home/geiger/gwb913/.conda/envs/vlm_probe/bin/python -u plot.py \
    --results $OUT_DIR/optimization_results.json \
    --out_dir $OUT_DIR/figures
