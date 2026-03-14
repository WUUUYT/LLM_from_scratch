#!/bin/bash
#SBATCH --job-name=batch_sweep
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

nvidia-smi

source .venv/bin/activate

TRAIN_PATH="data/tinystories_train_ids.uint16"
VAL_PATH="data/tinystories_test_ids.uint16"
BEST_LR=1e-3
MIN_LR=3e-5
TOTAL_TOKENS=327680000

# batch_size 从 1 到 GPU 显存上限
# max_iters = TOTAL_TOKENS / (batch_size × 256)
# batch_size = 1 8 32 64 128 256 512
for BATCH_SIZE in 32 64 128 256 512; do
    MAX_ITERS=$(python -c "print(int($TOTAL_TOKENS / ($BATCH_SIZE * 256)))")
    echo "DEBUG: MAX_ITERS='$MAX_ITERS', exit_code=$?"
    WARMUP_ITERS=$(python -c "print(int($MAX_ITERS * 0.04))")
    echo "=============================="
    echo "Running with batch_size=$BATCH_SIZE, max_iters=$MAX_ITERS"
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
        --batch_size $BATCH_SIZE \
        --max_iters $MAX_ITERS \
        --max_lr $BEST_LR \
        --min_lr $MIN_LR \
        --warmup_iters $WARMUP_ITERS \
        --val_every 100 \
        --checkpoint_every 1000 \
        --device cuda \
        --wandb \
        --wandb_project cs336_batch_sweep \
        --checkpoint_dir checkpoints/batch_$BATCH_SIZE
done
