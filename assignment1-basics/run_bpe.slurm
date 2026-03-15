#!/bin/bash
#SBATCH --job-name=train_bpe
#SBATCH --output=logs/train_bpe_%j.out
#SBATCH --error=logs/train_bpe_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
cd $WORK_DIR
source .venv/bin/activate

# 进入项目目录
cd cs336_basics

echo "Start training BPE..."
python train_bpe_tinystories.py
