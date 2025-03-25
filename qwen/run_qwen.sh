#!/bin/bash
#SBATCH --job-name=qwen_inference
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=epyc2
#SBATCH --output=qwen_output.log


# Load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate virtual environment
source ~/SSE/llm_env/bin/activate

# Run the inference script
python run_qwen_interface.py
