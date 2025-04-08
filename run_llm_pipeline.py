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
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
PIPELINE_ARGS = {
    "task": "text-generation",
    "model": MODEL_NAME,
    "torch_dtype": "auto",
    "device_map": "auto",
    "token": "to be replaced",
    "return_full_text": False,
}

# Load dataset and evaluation metric
data_set= load_dataset("openai_humaneval")['test']
code_eval_metric = load("code_eval")

# Initialize pipeline
pipe = pipeline(**PIPELINE_ARGS)

# Define prompting strategies
PROMPT_VARIANTS = {
    "PROMPT#1": lambda p: p,
    "PROMPT#2": lambda p: f"{p}\n\n# Think step-by-step before writing the code.",
}

def generate_code(prompt, num_samples=5):
    """Generate code completions for a given prompt, extracting code blocks."""
    candidates = []
    for _ in range(num_samples):
        messages = [
            {"role": "system", "content": "You are an AI that completes Python code!"},
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(messages, max_new_tokens=256)
        generated_text = outputs[0]["generated_text"].strip()

        # Extract the first Python code block if present
        match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # Fall back to entire text if no code block is found
            code = generated_text

        candidates.append(code)
    return candidates

def evaluate_results(results_by_prompt, test_cases, k_values=[1, 5]):
    """Evaluate generated code and print pass@k scores."""
    prompt_performance = {}
    for prompt_key, predictions in results_by_prompt.items():
        pass_at_k, _ = code_eval_metric.compute(
            references=test_cases,
            predictions=predictions,
            k=k_values,
            num_workers=4,
            timeout=10.0,
        )
        prompt_performance[prompt_key] = pass_at_k
        print(f"\nResults for {prompt_key}:")
        for k in k_values:
            print(f"Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")
    return prompt_performance

# Process dataset
test_cases = []
results_by_prompt = {key: [] for key in PROMPT_VARIANTS.keys()}

# uncomement for testing
i = 0
for problem in data_set:
    # unvomment for testing
    if i == 2:
        break
    test_cases.append(problem["test"])

    for prompt_key, prompt_fn in PROMPT_VARIANTS.items():
        prompt = prompt_fn(problem["prompt"])
        results_by_prompt[prompt_key].append(generate_code(prompt))

    # uncomment for testing
    i += 1

# Evaluate PROMPT#1 and PROMPT#2
prompt_performance = evaluate_results(results_by_prompt, test_cases)

# Define and evaluate PROMPT#3
PROMPT_VARIANTS["PROMPT#3"] = lambda p, l, s: f"{p}\n\n# Here is the solution of a previous run:\n {l}. The pass@1 score of this solution was {s}. Improve the solution."
results_by_prompt["PROMPT#3"] = []

# uncomment for testing
i = 0
for problem in data_set:
    # uncomment for testing
    if i == 2:
        break

    best_prompt = "PROMPT#2" if prompt_performance["PROMPT#2"]["pass@1"] >= prompt_performance["PROMPT#1"]["pass@1"] else "PROMPT#1"
    prompt = PROMPT_VARIANTS["PROMPT#3"](problem["prompt"], results_by_prompt[best_prompt][0][0], prompt_performance[best_prompt]["pass@1"])
    results_by_prompt["PROMPT#3"].append(generate_code(prompt))

    # uncomment for testing
    i += 1

# Evaluate PROMPT#3
evaluate_results({"PROMPT#3": results_by_prompt["PROMPT#3"]}, test_cases)