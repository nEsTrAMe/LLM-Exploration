import os
import re
import json
import sys
import tempfile
import subprocess
from transformers import pipeline
from datasets import load_dataset

# Get model
MODEL_NAME = sys.argv[1]
clean_model_name = MODEL_NAME.replace("/", "-")

# Get mode
MODE = sys.argv[2]

if MODE == "2":
    mode = "reasoning"
elif MODE == "3":
    mode = "feedback"
else:
    mode = "default"

PIPELINE_ARGS = {
    "task": "text-generation",
    "model": MODEL_NAME,
    "torch_dtype": "auto",
    "device_map": "auto",
    "token": "to be replaced",
    "return_full_text": False,
    "temperature": 0.7,
}

NUM_SAMPLES = 5
K_VALUES = [1, 3, 5]
MAX_RETRIES = 2

# Load dataset
dataset = load_dataset("openai_humaneval")['test']

def fix_tests(example):
    example['test'] += f"""

def test_solution():
    check({example['entry_point']})
"""
    return example

def get_prompt(example, feedback, code, error_msg):

    if MODE == "2":

        messages = [
            {"role": "user", "content": "You are an AI that completes Python code!"},
            {"role": "user", "content": "Reason step-by-step but keep the reasoning as short as possible. Put your final answer within python``` answer ```"},
            {"role": "user", "content": example["prompt"]}
        ]
        return messages

    messages = [
        {"role": "system", "content": "You are an AI that completes Python code!"},
        {"role": "user", "content": example["prompt"]},
    ]
    if feedback:
        previous_attempt = f"""
        This was a previous attempt:

        ```python
        {code}
        ```
        """
        error_message = f"""
        The result of the pytest is the following:

        {error_msg}

        Check the error message to see why the test failed and provide an improved solution of the above task based on the feedbackof the test.
        """

        messages.append({"role": "user", "content": previous_attempt })
        messages.append({"role": "user", "content": error_message})
    return messages

def generate_code(example, num_samples=1, feedback=False, code=None, error_msg=None):
    messages = get_prompt(example, feedback, code, error_msg)
    candidates = []
    for _ in range(num_samples):
        outputs = pipe(messages, max_new_tokens=4096)
        generated_text = outputs[0]["generated_text"].strip()
        match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        code = match.group(1).strip() if match else generated_text
        candidates.append(code)
    return candidates

def run_pytest_on_code(code: str, test_code: str) -> (str, str):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        full_code = f"{code}\n\n{test_code}"
        tmp.write(full_code)
        tmp.flush()
        try:
            result = subprocess.run(
                ["pytest", tmp.name, "-q", "--tb=short", "--disable-warnings"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "failed", "Timeout occurred during test execution."
        if result.returncode == 0:
            return "passed", ""
        else:
            return "failed", result.stdout

# Fix dataset
dataset = dataset.map(fix_tests)

# Init model pipeline
pipe = pipeline(**PIPELINE_ARGS)

# Logging and result storage
logs = {}
all_pass_k = []

# Logging and result storage
logs = {}
all_pass_k = []

for example in dataset.select(range(50)):
    task_id = example["task_id"]

    # Initial code generation
    candidates = generate_code(example, num_samples=NUM_SAMPLES)

    final_candidates = []
    final_results = []
    improvement_flags = []

    for code in candidates:
        result, error_msg = run_pytest_on_code(code, example["test"])
        retry_count = 0
        improved = False

        if MODE == "3":
            # Retry if it failed
            while result != "passed" and retry_count < MAX_RETRIES:
                code = generate_code(example, 1, True, code, error_msg)[0]
                result, error_msg = run_pytest_on_code(code, example["test"])
                retry_count += 1
                if result == "passed":
                    improved = True


        final_candidates.append(code)
        final_results.append(result)
        improvement_flags.append(improved)

    # Compute pass@k
    pass_k = {}
    for k in K_VALUES:
        top_k = final_results[:k]
        pass_k[f"pass@{k}"] = 1 if "passed" in top_k else 0

    all_pass_k.append(pass_k)

    logs[task_id] = {
        "prompt": example["prompt"],
        "test": example["test"],
        "candidates": final_candidates,
        "results": final_results,
        **pass_k,
        "improvements": improvement_flags,
    }

# Compute average pass@k across all tasks
avg_pass_k = {}
for k in K_VALUES:
    avg_pass_k[f"pass@{k}"] = sum(d[f"pass@{k}"] for d in all_pass_k) / len(all_pass_k)

logs["average_pass@k"] = avg_pass_k
logs["total_feedback_improvements"] = sum(improvement_flags for task_id, task in logs.items() if isinstance(task, dict) for improvement_flags in task.get("improvements", []))

# Save results
output_file = f"json/humaneval_{mode}_{clean_model_name}.json"
with open(output_file, "w") as f:
    json.dump(logs, f, indent=2)

print(f"\nEvaluation complete. Logs saved to '{output_file}'")