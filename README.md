# 🔭 Exploration of Self-Reflective LLMs for Code

Large Language Models (LLMs) have rapidly advanced in recent years, producing increasingly high-quality outputs across a wide range of tasks. A key factor behind this progress is the integration of reasoning capabilities, encouraging models not just to predict, but to think. Recent models like [o1](https://openai.com/o1/) and [DeepSeek-R1](https://api-docs.deepseek.com/news/news250120) exemplify this shift, utilizing techniques such as self-reflection, chain-of-thought prompting, and reinforcement learning to enhance logical reasoning. This has shown promise in complex domains like code repair and generation, where deep understanding and step-by-step problem solving are essential.

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

To make everything as reproducible as possible, a python virtual environment is created in which all dependencies later used to run the project are installed. We will use [transformers](https://huggingface.co/docs/transformers/index) to inference with the models.

```Bash
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

python -m venv llm_env
source ~/SSE/llm_env/bin/activate

pip install torch
pip install transformers
pip install accelerate
pip install pytest
```

## 🔎 Code Evaluation
This project uses pass@k as a primary evaluation metric for generated code snippets. The pass@k metric measures the probability that at least one out of k generated code samples passes all test cases. It provides a practical way to assess the reliability and usefulness of code generation systems, especially when multiple attempts can be made. This metric reflects real-world scenarios where users may review or rerun several generated outputs to find a correct solution. 

To support an effective feedback loop, pytest was used to create a test suite. This allows for clear identification of failed assertions via tracebacks, which can then be passed back to the model for analysis. Please consult [run.py](/run.py) for the exact implementation.

## ⚙️ Models
In this project, two distinct types of large language models where evaluated. Firstly, Instruct models which are optimized to follow user instructions accurately and generate coherent, well-structured outputs in response to prompts. Secondly, Reasoning models, which are designed to perform complex multi-step reasoning, excelling in tasks that require logic, inference, or problem-solving capabilities.

**Instruct**
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

**Reasoning**
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

## 🏃 Run project

> [!IMPORTANT]  
> Adjust the Hardware requirements depending on the model used.

Create folders for logs and for the json output:
```Bash
mkdir logs
mkdir json
```

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

## 📊 Model Evaluation
The following tables show the average pass@k values for different models, categorized by prompt type. The prompts used, the generated candidates, and the detailed test results from the test run can be found [here](/results/).

### Default prompt

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Llama-3.2-1B-Instruct        | 0.36   | 0.49   | 0.58   |
| Llama-3.1-8B-Instruct        | 0.68   | 0.78   | 0.80   |
| Qwen2.5-Coder-1.5B-Instruct  | 0.59   | 0.75   | 0.78   |
| Qwen2.5-Coder-7B-Instruct    | 0.80   | 0.86   | 0.88   |
| DeepSeek-R1-Distill-Llama-8B | 0.76   | 0.84   | 0.87   |
| DeepSeek-R1-Distill-Qwen-1.5B| 0.40   | 0.63   | 0.66   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.78   | 0.86   | 0.87   |
| DeepSeek-R1-Distill-Qwen-14B | 0.84   | 0.88   | 0.90   |

### Reasoning prompt

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Llama-3.2-1B-Instruct        | 0.23   | 0.37   | 0.46   |
| Llama-3.1-8B-Instruct        | 0.61   | 0.72   | 0.78   |
| Qwen2.5-Coder-1.5B-Instruct  | 0.49   | 0.74   | 0.77   |
| Qwen2.5-Coder-7B-Instruct    | 0.79   | 0.87   | 0.89   |
| DeepSeek-R1-Distill-Llama-8B | 0.75   | 0.85   | 0.88   |
| DeepSeek-R1-Distill-Qwen-1.5B| 0.41   | 0.65   | 0.70   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.76   | 0.84   | 0.87   |
| DeepSeek-R1-Distill-Qwen-14B | 0.80   | 0.90   | 0.91   |

### Feedback Loop

| Model                        | pass@1 | pass@3 | pass@5 |
| ---------------------------- | ------ | ------ | ------ |
| Llama-3.2-1B-Instruct        | 0.37   | 0.55   | 0.65   |
| Llama-3.1-8B-Instruct        | 0.77   | 0.87   | 0.90   |
| Qwen2.5-Coder-1.5B-Instruct  | 0.68   | 0.82   | 0.86   |
| Qwen2.5-Coder-7B-Instruct    | 0.91   | 0.94   | 0.94   |
| DeepSeek-R1-Distill-Llama-8B | 0.90   | 0.95   | 0.96   |
| DeepSeek-R1-Distill-Qwen-1.5B| 0.52   | 0.76   | 0.78   |
| DeepSeek-R1-Distill-Qwen-7B  | 0.86   | 0.93   | 0.94   |
| DeepSeek-R1-Distill-Qwen-14B |   -    |   -    |   -    |

> [!NOTE]  
> Due to job time constrains, the feedback loop for the `DeepSeek-R1-Distill-Qwen-14B` model could not be completed.

## 📈 Model Performance
The evaluation clearly demonstrates that larger models consistently outperform their smaller counterparts across all prompting strategies. Notably:

- `DeepSeek-R1-Distill-Qwen-14B` achieved the highest scores in both the Default and Reasoning prompts, showcasing the effectiveness of reinforcement learning in enhancing reasoning capabilities.

- Feedback-based prompting outperformed both Default and Reasoning prompts across most models. For example, using feedback, Qwen2.5-Coder-7B-Instruct corrected 69 initially incorrect code suggestions, increasing its pass@1 from 0.80 (Default) to 0.91 (Feedback). This demonstrates the power of iterative self-improvement in enhancing code correctness.

- Smaller models, such as Llama-3.2-1B, showed limited performance improvements with Reasoning prompts, suggesting a lower ceiling for emergent reasoning behaviors without adequate model capacity.

This indicates that reasoning and feedback strategies are particularly effective when applied to models already capable of complex reasoning, and that both model architecture and scale play key roles in performance.

## 🧠 Key Insights
**Feedback loops are powerful**: Incorporating test-driven feedback significantly enhances model performance. This mirrors how human developers iterate over code, suggesting that LLMs benefit from similar interactive refinement cycles.

**Model size matters**: Larger models exhibit stronger emergent reasoning behaviors and perform better across all prompt types, especially in tasks requiring logical deduction and multi-step planning.

**Specialized training boosts reasoning**: Reasoning-optimized models like `DeepSeek-R1-*` consistently outperform instruct-tuned models of similar size, highlighting the impact of targeted reinforcement learning strategies in complex coding tasks.

## Sources
1. https://arxiv.org/abs/2501.12948
2. https://www.arxiv.org/abs/2502.03671
3. https://arxiv.org/abs/2201.11903

[1]: https://arxiv.org/abs/2501.12948
[2]: https://www.arxiv.org/abs/2502.03671
[3]: https://arxiv.org/abs/2201.11903
