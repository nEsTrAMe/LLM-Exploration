import os
import re
from transformers import pipeline
from evaluate import load
from datasets import load_dataset

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
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
    prompt = example["prompt"]
    if strategy == "default":
        messages.append({"role": "user", "content": prompt})

    elif strategy == "reasoning":
        messages.append({"role": "user", "content": "Think through the problem carefully and explain your reasoning before writing code.\n\n" })
        messages.append({"role": "user", "content": prompt})

    elif strategy == "feedback":
        messages.append({"role": "user", "content": f"The following problem was provided to you in the previous run:\n{prompt}"})
        messages.append({"role": "user", "content": f"You provided the following solution for the problem:\n{feedback}"})
        messages.append({"role": "user", "content": f"The tests provided the following feedback:\n{test}"})
        messages.append({"role": "user", "content": "Provide an improved solution to pass all the tests"})

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

# Fix testing
data_set = data_set.map(fix_tests)

# Initialize pipeline
pipe = pipeline(**PIPELINE_ARGS)

# testing
example = data_set[70]

# 1. Default
default = generate_code(example, strategy="default")
k_pass1, results = evaluate_code(example['test'], default)
print(k_pass1)

# 2. Reasoning
reasoning = generate_code(example, strategy="reasoning")
k_pass, results = evaluate_code(example['test'], reasoning)
print(k_pass)

# 3. Feedback
feedback = generate_code(example, strategy="feedback", feedback=reasoning, test=results[0][0][1]['result'])
k_pass2, results2 = evaluate_code(example['test'], feedback)
print(k_pass2)
print(results2)