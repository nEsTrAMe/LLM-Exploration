import os
import re
import json
from transformers import pipeline
from evaluate import load
from datasets import load_dataset

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configuration
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

PIPELINE_ARGS = {
    "task": "text-generation",
    "model": MODEL_NAME,
    "torch_dtype": "auto",
    "device_map": "auto",
    "token": "to be replaced",
    "return_full_text": False,
    "temperature": 0.7,
}

# Load dataset and evaluation metric
dataset = load_dataset("openai_humaneval")['test']
code_eval_metric = load("code_eval")

def fix_tests(example):
    example['test'] += f"\ncheck({example['entry_point']})"
    return example

def get_prompt(example):
    return [
        {"role": "system", "content": "You are an AI that completes Python code. Think step by step before coding."},
        {"role": "user", "content": f"{example['prompt']}\n# Let's think step by step and then write the function."},
    ]

def generate_code(example, num_samples=5):
    messages = get_prompt(example)
    candidates = []
    for _ in range(num_samples):
        outputs = pipe(messages, max_new_tokens=512)
        generated_text = outputs[0]["generated_text"].strip()
        match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        code = match.group(1).strip() if match else generated_text
        candidates.append(code)
    return candidates

# Fix dataset
dataset = dataset.map(fix_tests)

# Init model pipeline
pipe = pipeline(**PIPELINE_ARGS)

# Logging and result storage
logs = {}
all_pass_k = []

for example in dataset.select(range(50)):
    task_id = example["task_id"]
    candidates = generate_code(example)
    k_pass, results = code_eval_metric.compute(
        references=[example["test"]],
        predictions=[candidates],
        k=[1, 3, 5],
        num_workers=1,
        timeout=10.0
    )
    all_pass_k.append(k_pass)

    result_only = [entry[1]['result'] for value in results.values() for entry in value]

    logs[task_id] = {
        "prompt": example["prompt"],
        "test": example["test"],
        "candidates": candidates,
        "results": result_only,
        "pass@k": k_pass
    }

# Compute average pass@k
avg_pass_k = {}
num_tasks = len(all_pass_k)
for k in [1, 3, 5]:
    avg_pass_k[f"pass@{k}"] = sum(item[f"pass@{k}"] for item in all_pass_k) / num_tasks

logs["average_pass@k"] = avg_pass_k

# Save results
with open("humaneval_reasoning_prompt_results.json", "w") as f:
    json.dump(logs, f, indent=2)

print("\nEvaluation complete. Logs saved to 'humaneval_reasoning_prompt_results.json'")