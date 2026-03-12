#!/bin/bash
#SBATCH --job-name=lr_sweep
#SBATCH --output=logs/lr_sweep_%j.out
#SBATCH --error=logs/lr_sweep_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

nvidia-smi

source .venv/bin/activate

TRAIN_PATH="data/tinystories_train_ids.uint16"
VAL_PATH="data/tinystories_test_ids.uint16"

for LR in 1e-2 3e-2; do
    MIN_LR=$(python -c "print($LR / 10)")
    echo "=============================="
    echo "Running with lr=$LR"
    echo "=============================="
    python training_together.py \
        --train_path $TRAIN_PATH \
        --val_path $VAL_PATH \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --num_heads 16 \
        --num_layers 4 \
        --d_ff 1344 \
        --batch_size 32 \
        --max_iters 5000 \
        --min_lr $MIN_LR \
        --warmup_iters 200 \
        --val_every 100 \
        --checkpoint_every 1000 \
        --device cuda \
        --wandb \
        --wandb_project cs336_lr_sweep \
        --max_lr $LR \
        --checkpoint_dir checkpoints/lr_$LR
done
