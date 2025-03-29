import os
from transformers import pipeline
from evaluate import load
from datasets import load_dataset

# allow the model evaluation to run
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# specify model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Load evaluation dataset
data_set= load_dataset("openai_humaneval")['test']

# Load code evaluation metric
code_eval_metric = load("code_eval")

# list to store test cases and predictions
test_cases = []
results_by_prompt = {}

pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",
        token='to be replaced',
        return_full_text=False
        )

# Define prompting strategies
prompt_variants = {
    "PROMPT#1": lambda p: p,
    "PROMPT#2": lambda p: f"{p}\n\n# Think step-by-step before writing the code.",
}


# uncomment for testing
i = 0

# iterate over problems in data set
for problem in data_set:

    # uncomment for testing
    if i == 3:
        break

    prompt_base = problem['prompt']
    # uncomment for testing
    # print(prompt)

    test_code = problem['test']
    # uncomment for testing
    # print(test_code)

    # Store the test cases
    test_cases.append(test_code)

    # Store results per prompt type
    for prompt_key, prompt_fn in prompt_variants.items():

        # create complete prompt
        prompt = prompt_fn(prompt_base)

        # uncomment for testing
        # print(prompt)

        # List to store all the candidates for each problem
        problem_candidates = []

        # create multiple predictions for each problem
        for _ in range(5):
                messages = [
                    {"role": "system", "content": "You are an AI which compleates python code! You only output python code and no explanations!"},
                    {"role": "user", "content": f"{prompt}"},
                    ]
                outputs = pipe(
                    messages,
                    max_new_tokens=256,
                    )

                # extract the generated text
                generated_text = outputs[0]["generated_text"]

                # Format the code
                if generated_text.startswith("```python"):
                    generated_text = generated_text[9:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]

                # Strip any unnecessary whitespace
                formatted_output = generated_text.strip()

                problem_candidates.append(formatted_output)

        # store code predictions
        if prompt_key not in results_by_prompt:
            results_by_prompt[prompt_key] = []
        results_by_prompt[prompt_key].append(problem_candidates)

    # uncomment for testing
    i += 1

# uncomment for testing
# print(f"!!--{code_predictions}--!!")

# Compute pass@k
k_values = [1, 3, 5]

for prompt_key, predictions in results_by_prompt.items():
    pass_at_k, results = code_eval_metric.compute(
        references=test_cases,
        predictions=predictions,
        k=k_values,
        num_workers=4,  # Adjust based on your system
        timeout=10.0,   # Adjust the timeout as needed
    )

    print(f"\nResults for {prompt_key}:")

    # Print the results
    for k in k_values:
        print(f"Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")