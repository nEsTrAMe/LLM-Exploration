# 🧠 Exploration of Self-Reflective LLMs for Code

Large Language Models (LLMs) have rapidly advanced in recent years, producing increasingly high-quality outputs across a wide range of tasks. A key factor behind this progress is the integration of reasoning capabilities, encouraging models not just to predict, but to think. Recent models like [o1](https://openai.com/o1/) and [DeepSeek-R1](https://api-docs.deepseek.com/news/news250120) exemplify this shift, utilizing techniques such as self-reflection, chain-of-thought prompting, and reinforcement learning to enhance logical reasoning. This has shown particular promise in complex domains like code repair and generation, where deep understanding and step-by-step problem solving are essential.

## 🎯 Objectives
This project investigates the effectiveness of self-reflective and reasoning-driven large language models (LLMs) on coding tasks, with a focus on code repair and vulnerability mitigation. By leveraging state-of-the-art models and techniques, we aim to evaluate whether deeper reasoning capabilities lead to more reliable and intelligent code generation.

## 💭 Reinforcement Learning and Emergent Reasoning
A notable development in this domain is the introduction of DeepSeek-R1, a reasoning model that leverages reinforcement learning to enhance large language models reasoning abilities. The initial version, DeepSeek-R1-Zero, was trained exclusively using reinforcement learning without supervised fine-tuning. This approach led to the emergence of sophisticated reasoning behaviors, enabling the model to achieve significant improvements on benchmarks like the AIME 2024 mathematics test, where its pass rate increased from 15.6% to 71.0%. [[1][1]]

## ⏩ Advancements in Prompting Strategies
Another significant area of progress involves the development of advanced prompting strategies to enhance LLM reasoning. Techniques such as Chain-of-Thought reasoning have been explored to improve the logical deduction and multi-step reasoning capabilities of large language models by guiding them to generate intermediate reasoning steps before arriving at a final answer. It involves providing models with exemplars that demonstrate step-by-step problem-solving processes, enabling them to tackle complex tasks more effectively. Empirical studies have shown that Chain-of-Thought prompting significantly improves performance across various reasoning benchmarks, including arithmetic, commonsense, and symbolic reasoning tasks. [[2][2]] [[3][3]]

## 👷 Evaluation methodology
This project evaluates models using the [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) dataset, which contains 164 hand-written programming problems. Each problem includes a function signature, a natural language docstring, an incomplete function body, and a set of unit tests.

We use three evaluation strategies for each model:
1. **Default**: The model is prompted with the problem and asked to directly complete the code.
2. **Reasoning**: The model is instructed to reason step-by-step before writing the solution.
3. **Feedback**: The model first produces a default solution, then improves it iteratively based on test feedback.

## 🛠️ Environment setup
The University of Bern provides a [HPC cluster](https://hpc-unibe-ch.github.io/) for student and research projects. UBELIX uses Slurm as job scheduler and resource manager.

In order to make everything as reproducible as possible, a python virtaul environment is created in which all dependencies later used to run the project are installed. We will use [transformers](https://huggingface.co/docs/transformers/index) to inference with the models.

```Bash
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

python -m venv llm_env
source ~/SSE/llm_env/bin/activate

pip install torch
pip install transformers
pip install accelerate
pip install evaluate
pip install pytest
```

## 🔎 Code Evaluation
This project uses pass@k as a primary evaluation metric for generated code snippets. The pass@k metric measures the probability that at least one out of k generated code samples passes all test cases. It provides a practical way to assess the reliability and usefulness of code generation systems, especially when multiple attempts can be made. This metric reflects real-world scenarios where users may review or rerun several generated outputs to find a correct solution. 

To support an effective feedback loop, pytest was used to create a test suite. This allows for clear identification of failed assertions via tracebacks, which can then be passed back to the model for analysis. Please consult [run.py](/run.py) for the exact implementation.

## ⚙️ Models
In this project, two distict types of large language models where evaluated. Fistly, Instruct models which are optimized to follow user instructions accurately and generate coherent, well-structured outputs in response to prompts. Secondly, Reasoning models, which are designed to perform complex multi-step reasoning, excelling in tasks that require logic, inference, or problem-solving capabilities.

**Instruct**
- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

**Reasoning**
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

## 🏃 Run project
To run a specific part of the evaluation pipeline for a single model, use the following command:
```Bash
sbatch run.sh "model name" MODE
```

Example: Run the reasoning task (mode 2) for the Qwen2.5-Coder-1.5B-Instruct model:
```Bash
sbatch run.sh "Qwen/Qwen2.5-Coder-1.5B-Instruct" 2
```

To run the full evaluation pipeline across multiple models, use the [submit_all.sh](/submit_all.sh) script:
```Bash
bash submit_all.sh
```

> [!IMPORTANT]  
> Adjust the Hardware requirements depending on the model used.

## 📋 Evaluation
The following tables show the average pass@k values for different models, categorized by prompt type. The prompts used, the generated candidates, and the detailed test results from the test run can be found [here](/results/).

### Default prompt

<center>

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Qwen2.5-Coder-1.5B-Instruct  | 0.xx   | 0.xx   | 0.xx   |
| Qwen2.5-Coder-7B-Instruct    | 0.xx   | 0.xx   | 0.xx   |
| Llama-3.2-1B-Instruct        | 0.xx   | 0.xx   | 0.72   |
| DeepSeek-R1-Distill-Llama-8B | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-14B | 0.xx   | 0.xx   | 0.xx   |

</center>

### Reasoning prompt

<center>

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Qwen2.5-Coder-1.5B-Instruct  | 0.xx   | 0.xx   | 0.xx   |
| Qwen2.5-Coder-7B-Instruct    | 0.xx   | 0.xx   | 0.xx   |
| Llama-3.2-1B-Instruct        | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Llama-8B | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-14B | 0.xx   | 0.xx   | 0.xx   |

</center>

### Feedback Loop

<center>

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Qwen2.5-Coder-1.5B-Instruct  | 0.xx   | 0.xx   | 0.xx   |
| Qwen2.5-Coder-7B-Instruct    | 0.xx   | 0.xx   | 0.xx   |
| Llama-3.2-1B-Instruct        | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Llama-8B | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.xx   | 0.xx   | 0.xx   |
| DeepSeek-R1-Distill-Qwen-14B | 0.xx   | 0.xx   | 0.xx   |

</center>

TODO: Summarize and Evaluate Results!

## Sources
1. https://arxiv.org/abs/2501.12948
2. https://www.arxiv.org/abs/2502.03671
3. https://arxiv.org/abs/2201.11903

[1]: https://arxiv.org/abs/2501.12948
[2]: https://www.arxiv.org/abs/2502.03671
[3]: https://arxiv.org/abs/2201.11903
