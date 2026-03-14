#!/bin/bash
#SBATCH --job-name=owt_leaderboard
#SBATCH --output=logs/owt_leaderboard_%j.out
#SBATCH --error=logs/owt_leaderboard_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

nvidia-smi

WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
cd $WORK_DIR
source .venv/bin/activate

TRAIN_PATH="$WORK_DIR/data/owt_train_ids.uint16"
VAL_PATH="$WORK_DIR/data/owt_valid_ids.uint16"

python training_together.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --vocab_size 32000 \
    --context_length 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --batch_size 64 \
    --max_iters 80000 \
    --max_lr 8e-4 \
    --min_lr 8e-5 \
    --warmup_iters 2000 \
    --val_every 500 \
    --val_iters 20 \
    --checkpoint_every 1000 \
    --weight_tying \
    --device cuda \
    --wandb \
    --wandb_project cs336_leaderboard \
    --checkpoint_dir checkpoints/leaderboard
