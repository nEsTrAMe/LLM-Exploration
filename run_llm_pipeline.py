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


# overwrite datset, see discord

# Initialize pipeline
pipe = pipeline(**PIPELINE_ARGS)

prompts = ["PROMPT#1", "PROMPT#2", "PROMPT#3"]

def generate_code(messages, num_samples=5):
    """Generate code compleations"""
    candidates = []
    for _ in range(num_samples):
        outputs = pipe(messages, max_new_tokens=512)
        generated_text = outputs[0]["generated_text"].strip()

        # Extract the first Python code block if present
        match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = generated_text
        candidates.append(code)
    return candidates

def evaluate_results(results_by_prompt, test_cases, k_values=[1, 3, 5]):
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

def evaluate_single(reference, prediction):
    """Evaluate a single prediction with the code_eval metric."""
    result, _ = code_eval_metric.compute(
        references=[reference],
        predictions=[prediction],
        k=[1],
        num_workers=1,
        timeout=10.0,
    )
    return result["pass@1"]

# Main logic
test_cases = []
results_by_prompt = {key: [] for key in prompts}

# Define the base prompt (shared starting point)
base_message = [
    {"role": "system", "content": "You are an AI that completes Python code!"},
]

i = 0
for problem in data_set:
    if i == 10:  # Testing limiter
        break

    test_cases.append(problem["test"])

    # PROMPT#1
    simple_request = base_message.copy()
    simple_request.append({"role": "user", "content": problem["prompt"]})

    # Generate and collect code for the single prompt
    results_1 = generate_code(simple_request)
    results_by_prompt["PROMPT#1"].append(results_1)

    # PROMPT#2
    reasoning_request = base_message.copy()
    reasoning_request.append({"role": "system", "content": "You always reason and think step-by-step before writing the code"})
    reasoning_request.append({"role": "user", "content": problem["prompt"]})

    # Generate and collect code for the single prompt
    results_2 = generate_code(reasoning_request)
    results_by_prompt["PROMPT#2"].append(results_2)

    score_2 = evaluate_single(problem["test"], results_2)

    # PROMPT#3
    feedback_request = base_message.copy()
    feedback_request.append({"role": "user", "content": problem["prompt"]})
    feedback_request.append({"role": "user", "content": f"In the previous attempt you provided the following solution:\n{results_2[0]}"})
    feedback_request.append({"role": "user", "content": f"The provided solution had a pass@1 score of {score_2} using the following test cases:\n {problem['test']}"})
    feedback_request.append({"role": "user", "content": "Improve the solution if the score is not 1.0"})

    # Generate and collect code for the single prompt
    results_3 = generate_code(feedback_request)
    results_by_prompt["PROMPT#3"].append(results_3)

    i += 1


prompt_performance = evaluate_results(results_by_prompt, test_cases)