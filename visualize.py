import matplotlib.pyplot as plt
import pandas as pd

# Model evaluation data
data = {
    "Model": [
        "Llama-3.2-1B", "Llama-3.1-8B", "Qwen2.5-1.5B", "Qwen2.5-7B", 
        "DS-Llama-8B", "DS-Qwen-1.5B", "DS-Qwen-7B", "DS-Qwen-14B"
    ],
    "Default": [0.36, 0.68, 0.59, 0.80, 0.76, 0.40, 0.78, 0.84],
    "Reasoning": [0.23, 0.61, 0.49, 0.79, 0.75, 0.41, 0.76, 0.80],
    "Feedback": [0.37, 0.77, 0.68, 0.91, 0.90, 0.52, 0.86, None]  # None = missing value
}

df = pd.DataFrame(data)
df.set_index("Model", inplace=True)

# Plot
plt.figure(figsize=(12, 6))
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

plt.title("pass@1 performance by prompt strategy")
plt.ylabel("pass@1 Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.ylim(0.2, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Prompt Type")
plt.tight_layout()
plt.show()