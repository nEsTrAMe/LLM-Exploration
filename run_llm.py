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
}

# Load dataset and evaluation metric
data_set = load_dataset("openai_humaneval")['test']
code_eval_metric = load("code_eval")

# Log structure
logs = {}

def fix_tests(example):
    """fix testing"""
    example['test'] += f"\ncheck({example['entry_point']})"
    return example

def get_prompt(example, strategy="default", feedback=None, test=None):
    """function which provides the different prompt strategies"""
    messages = [
        {"role": "system", "content": "You are an AI that completes Python code!"},
    ]
    prompt = example['prompt']

    # Should the test cases be provided in the feedback?
    test_cases = example['test']

    if strategy == "default":
        messages.append({"role": "user", "content": prompt})

    elif strategy == "reasoning":
        messages.append({"role": "user", "content": "Think through the problem carefully and explain your reasoning before writing code.\n\n"})
        messages.append({"role": "user", "content": prompt})

    elif strategy == "feedback":
        messages.append({"role": "user", "content": f"The following problem was provided to you:\n{prompt}"})

        for i, (f_code, f_test) in enumerate(zip(feedback, test)):
            messages.append({"role": "user", "content": f"Attempt {i+1}:\nYour solution:\n{f_code}\nYour solution failed the test with the message:\n{f_test}."})

        messages.append({"role": "user", "content": "Provide an improved solution."})

    return messages

def generate_code(example, strategy="default", num_samples=3, feedback=None, test=None):
    """generate code compleations"""
    messages = get_prompt(example, strategy=strategy, feedback=feedback, test=test)

    candidates = []
    for _ in range(num_samples):
        outputs = pipe(messages, max_new_tokens=512)
        generated_text = outputs[0]["generated_text"].strip()

        match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        code = match.group(1).strip() if match else generated_text
        candidates.append(code)
    return candidates

def evaluate_code(reference, prediction, k_values=[1, 3]):
    """Evaluate a single prediction with the code_eval metric."""
    k_pass, results = code_eval_metric.compute(
        references=[reference],
        predictions=[prediction],
        k=k_values,
        num_workers=1,
        timeout=10.0,
    )
    return k_pass, results

# Fix testing
data_set = data_set.map(fix_tests)

# Initialize pipeline
pipe = pipeline(**PIPELINE_ARGS)

# Loop through multiple examples
results_summary = []

for idx, example in enumerate(data_set.select(range(5))):

    task_log = {
        "PROMPT": example["prompt"],
        "TEST": example["test"],
        "HISTORY": []
    }

    # Default strategy
    default = generate_code(example, strategy="default")
    default_k, default_result = evaluate_code(example['test'], default)
    task_log["HISTORY"].append({
        "strategy": "default",
        "candidates": default,
        "k_pass": default_k,
        "feedback": default_result
    })

    # Reasoning strategy
    reasoning = generate_code(example, strategy="reasoning")
    reasoning_k, reasoning_result = evaluate_code(example['test'], reasoning)
    task_log["HISTORY"].append({
        "strategy": "reasoning",
        "candidates": reasoning,
        "k_pass": reasoning_k,
        "feedback": reasoning_result
    })

    # Feedback loop using all candidates and results
    attempt_codes = []
    attempt_feedback = []

    current_solutions = generate_code(example, strategy="default")
    attempt_codes.extend(current_solutions)

    k_pass, results = evaluate_code(example['test'], current_solutions)
    feedback_batch = [res[1]['result'] for res in results[0]]
    attempt_feedback.extend(feedback_batch)

    task_log["HISTORY"].append({
        "strategy": "feedback-attempt-1",
        "candidates": current_solutions,
        "k_pass": k_pass,
        "feedback": results
    })

    # attempts 2 and 3
    for attempt in range(2, 4):

        if 'passed' in feedback_batch:
            break

        improved_solutions = generate_code(
            example,
            strategy="feedback",
            feedback=attempt_codes,
            test=attempt_feedback
        )

        k_pass, results = evaluate_code(example['test'], improved_solutions)
        feedback_batch = [res[1]['result'] for res in results[0]]
        attempt_codes.extend(improved_solutions)
        attempt_feedback.extend(feedback_batch)

        task_log["HISTORY"].append({
            "strategy": f"feedback-attempt-{attempt}",
            "candidates": improved_solutions,
            "k_pass": k_pass,
            "feedback": results
        })

    # Save the log
    logs[f"{example['task_id']}"] = task_log

# Save to JSON
with open("humaneval_run_logs.json", "w") as f:
    json.dump(logs, f, indent=2)

print("\nLogs written to 'humaneval_run_logs.json'")