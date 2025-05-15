#!/bin/bash
#SBATCH --job-name=run
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --output=logs/%x_%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate virtual environment
source ~/SSE/llm_env/bin/activate

# Run the inference script with the model name and the mode argument
python run.py "$1" "$2"
