#!/bin/bash

TRAIN_PATH="data/tinystories_train_ids.uint16"
VAL_PATH="data/tinystories_test_ids.uint16"

for LR in 1e-5 1e-4 3e-4 1e-3 3e-3; do
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
        --min_lr 3e-5 \
        --warmup_iters 200 \
        --val_every 100 \
        --checkpoint_every 1000 \
        --device mps \
        --wandb \
        --wandb_project cs336_lr_sweep \
        --max_lr $LR \
        --checkpoint_dir checkpoints/lr_$LR
done
