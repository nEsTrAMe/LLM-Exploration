
# TODO:log in json

import os
import re
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
    test_cases = example['test']

    if strategy == "default":
        messages.append({"role": "user", "content": prompt})

    elif strategy == "reasoning":
        messages.append({"role": "user", "content": "Think through the problem carefully and explain your reasoning before writing code.\n\n"})
        messages.append({"role": "user", "content": prompt})

    elif strategy == "feedback":
        messages.append({"role": "user", "content": f"The following problem was provided to you:\n{prompt}"})

        for i, (f_code, f_test) in enumerate(zip(feedback, test)):
            messages.append({"role": "user", "content": f"Attempt {i+1}:\nYour solution:\n{f_code}\nFeedback:\n{f_test}"})

        messages.append({"role": "user", "content": "Provide an improved solution."})
    return messages

def generate_code(example, strategy="default", num_samples=1, feedback=None, test=None):
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

def evaluate_code(reference, prediction, k_values=[1]):
    """Evaluate a single prediction with the code_eval metric."""
    k_pass, results = code_eval_metric.compute(
        references=[reference],
        predictions=[prediction],
        k=k_values,
        num_workers=1,
        timeout=10.0,
    )
    return k_pass, results

def run_feedback_loop(example, initial_strategy="default", max_attempts=3):
    # Initialize history
    attempt_codes = []
    attempt_feedbacks = []

    # First attempt
    current_solution = generate_code(example, strategy=initial_strategy)
    attempt_codes.append(current_solution)
    k_pass, results = evaluate_code(example['test'], current_solution)

    # Try improving until max_attempts or pass
    for attempt in range(max_attempts):
        feedback_result = results[0][0][1]['result']
        attempt_feedbacks.append(feedback_result)

        if feedback_result == 'passed':
            break

        improved_solution = generate_code(
            example,
            strategy="feedback",
            feedback=attempt_codes,
            test=attempt_feedbacks
        )

        k_pass, results = evaluate_code(example['test'], improved_solution)
        attempt_codes.append(improved_solution)

    return attempt_codes[-1], k_pass

# Fix testing
data_set = data_set.map(fix_tests)

# Initialize pipeline
pipe = pipeline(**PIPELINE_ARGS)

# Loop through multiple examples
results_summary = []

for idx, example in enumerate(data_set.select(range(10))):
    print(f"\n--- Example {idx} ---")

    # Default strategy
    default = generate_code(example, strategy="default")
    default_k, _ = evaluate_code(example['test'], default)
    print(f"[Default] k_pass: {default_k}")


    # Reasoning strategy
    reasoning = generate_code(example, strategy="reasoning")
    reasoning_k, _ = evaluate_code(example['test'], reasoning)
    print(f"[Reasoning] k_pass: {reasoning_k}")

    # Feedback loop
    final_solution, feedback_k = run_feedback_loop(example)
    print(f"[Feedback] k_pass: {feedback_k}")

    results_summary.append({
        "example_id": idx,
        "default_k": default_k,
        "reasoning_k": reasoning_k,
        "feedback_k": feedback_k
    })

# Print final summary
print("\n=== Summary of Results ===")
for result in results_summary:
    print(result)
