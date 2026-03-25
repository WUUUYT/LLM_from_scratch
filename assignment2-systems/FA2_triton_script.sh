#!/bin/bash
#SBATCH --job-name=lr_sweep
#SBATCH --output=logs/FA2_triton_%j.out
#SBATCH --error=logs/FA2_triton_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

nvidia-smi

WORK_DIR=/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment2-systems
cd $WORK_DIR
source .venv/bin/activate

pytest -k test_flash_forward_pass_triton
