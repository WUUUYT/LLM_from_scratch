#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source .venv/bin/activate

python generate.py \
    --checkpoint checkpoints/lr_1e-3/ckpt_0005000.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda
