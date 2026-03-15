#!/bin/bash
#SBATCH --job-name=owt_leaderboard
#SBATCH --output=logs/owt_leaderboard_%j.out
#SBATCH --error=logs/owt_leaderboard_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

nvidia-smi
export PYTHONUNBUFFERED=1

WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
cd $WORK_DIR
source .venv/bin/activate

TRAIN_PATH="dataset/owt_train_ids.uint16"
VAL_PATH="dataset/owt_valid_ids.uint16"

python training_together.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --vocab_size 32000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 4 \
    --d_ff 1344 \
    --batch_size 128 \
    --max_iters 10000 \
    --max_lr 1e-3 \
    --min_lr 1e-4 \
    --warmup_iters 1000 \
    --val_every 100 \
    --val_iters 25 \
    --log_every 10 \
    --checkpoint_every 1000 \
    --device cuda \
    --wandb \
    --wandb_project cs336_leaderboard \
    --checkpoint_dir checkpoints/leaderboard
