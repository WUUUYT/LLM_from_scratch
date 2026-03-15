#!/bin/bash
#SBATCH --job-name=encode_dataset
#SBATCH --output=logs/encode_%j.out
#SBATCH --error=logs/encode_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
cd $WORK_DIR
source .venv/bin/activate

python cs336_basics/tokenizer_experiments.py
