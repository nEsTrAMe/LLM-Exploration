#!/bin/bash
#SBATCH --job-name=benachmark_llm_feedback
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:4

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate virtual environment
source ~/SSE/llm_env/bin/activate

# Run the inference script
python run_llm_feedback.py