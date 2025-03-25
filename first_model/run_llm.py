from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    token="to be replaced"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Write a quicksort algorithm in Python."
messages = [
    {"role": "system", "content": "You are an AI assistant. You provide helpful and precise responses."},
    {"role": "user", "content": prompt}
]

# Format input for Llama-3.2
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
