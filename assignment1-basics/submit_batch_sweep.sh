#!/bin/bash
#SBATCH --job-name=batch_sweep
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00

nvidia-smi


WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
cd $WORK_DIR
source .venv/bin/activate

TRAIN_PATH="dataset/tinystories_train_ids.uint16"
VAL_PATH="dataset/tinystories_valid_ids.uint16"
BASE_LR=1e-3
BASE_BATCH=128
TOTAL_TOKENS=327680000

for BATCH_SIZE in 16 32 64 128 256 512 1024; do
    MAX_ITERS=$(( TOTAL_TOKENS / (BATCH_SIZE * 256) ))
    WARMUP_ITERS=$(( MAX_ITERS * 4 / 100 ))

    VAL_EVERY=$(( MAX_ITERS / 50 ))
    if [ $VAL_EVERY -lt 1 ]; then VAL_EVERY=1; fi

    CURRENT_LR=$(awk -v base=$BASE_LR -v b=$BATCH_SIZE -v base_b=$BASE_BATCH 'BEGIN { print base * sqrt(b / base_b) }')
    MIN_LR=$(awk -v clr=$CURRENT_LR 'BEGIN { print clr / 10 }')

    echo "======================================"
    echo "🚀 BATCH_SIZE=$BATCH_SIZE | ITERS=$MAX_ITERS | LR=$CURRENT_LR | VAL_EVERY=$VAL_EVERY"
    echo "======================================"

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
        --max_lr $CURRENT_LR \
        --min_lr $MIN_LR \
        --warmup_iters $WARMUP_ITERS \
        --weight_decay 0.1 \
        --checkpoint_dir checkpoints/bs_$BATCH_SIZE \
        --checkpoint_every $(( MAX_ITERS / 5 )) \
        --val_every $VAL_EVERY \
        --val_iters 20 \
        --log_every 10 \
        --wandb \
        --wandb_project cs336_batch_sweep \
        --wandb_name "bs_${BATCH_SIZE}" \
        --device cuda
done
