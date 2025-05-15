#!/bin/bash

models=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

# Submit jobs for each model with parameters 1, 2, and 3
for model in "${models[@]}"; do
  for param in 1 2 3; do
    echo "Submitting job for model: $model with parameter: $param"
    sbatch run.sh "$model" "$param"
  done
done