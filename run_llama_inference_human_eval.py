import torch
from transformers import pipeline
from evaluate import load
from datasets import load_dataset

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Load evaluation dataset
human_eval = load_dataset("openai_humaneval")['test']

# Load code evaluation metric
code_eval_metric = load("code_eval")

# Set the number of candidates per problem
num_samples_per_problem = 1 # change amount after testing

# Lists to store test cases and predictions
test_cases = []
candidates = []

i = 0   #testing

for problem in human_eval:

    if i > 1:   #testing
        break   #testing

    prompt = problem['prompt']
    # print(prompt)   #testing

    test_code = problem['test']

    # Store the test cases
    test_cases.append(test_code)

    # Generate multiple candidate solutions for each problem
    problem_candidates = []

    i+=1    #testing

    # create inner loop for samples per problem
    for _ in range(num_samples_per_problem):
            pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token='to be replaced',
                return_full_text=False
                )
            messages = [
                {"role": "system", "content": "You are an AI which compleates python code! You only output python code and no explanations! Do not declare the code as python!"},
                {"role": "user", "content": f"{prompt}"},
                ]
            outputs = pipe(
                messages,
                max_new_tokens=256,
                )
            generated_text = outputs[0]["generated_text"]
            code_output = generated_text.split("```")[-2] if "```" in generated_text else generated_text  # Extract code if it's inside triple backticks

            problem_candidates.append(code_output.strip())
    candidates.append(problem_candidates)

# Compute pass@k
k_values = [1, 1]
print("Evaluating generated code...")
pass_at_k, results = code_eval_metric.compute(
    references=test_cases,
    predictions=candidates,
    k=k_values,
    num_workers=4,  # Adjust based on your system
    timeout=10.0,   # Adjust the timeout as needed
)

# Print the results
for k in k_values:
    print(f"Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")