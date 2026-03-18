#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source .venv/bin/activate

python generate.py \
    --checkpoint checkpoints/lr_1e-2/ckpt_final_10000.pt \
    --vocab_path outputs/tinystories_vocab.pkl \
    --merges_path outputs/tinystories_merges.pkl \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda \
    --vocab_size 10000 \
