import torch
from transformers import pipeline
from datasets import load_dataset

model_id = "meta-llama/Llama-3.2-1B-Instruct"
dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")

# Filter only Python code examples
def is_python_code(example):
    return example["lang"].lower() == "python"

python_dataset = dataset.filter(is_python_code)

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token='to be replaced with access token'
)

def benchmark_model():
    results = []
    for example in python_dataset:
        code_snippet = example["rejected"]
        messages = [
            {"role": "system", "content": "You are a security expert analyzing Python code for vulnerabilities."},
            {"role": "user", "content": f"Analyze the following Python code for security vulnerabilities:\n{code_snippet}"},
        ]
        
        output = pipe(
            messages,
            max_new_tokens=256,
        )
        
        results.append({
            "code": code_snippet,
            "analysis": output[0]["generated_text"]
        })
    
    return results

benchmark_results = benchmark_model()

for result in benchmark_results[:5]:  # Print first 5 results for inspection
    print("Code Snippet:")
    print(result["code"])
    print("Analysis:")
    print(result["analysis"])
    print("=" * 80)

