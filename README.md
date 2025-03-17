# Seminar Software Engineering

In recent years, large language models have achieved remarkable advancements, particularly in enhancing their reasoning capabilities. These improvements stem from innovative training methodologies, including reinforcement learning and self-reflection techniques, leading to superior performance in tasks requiring logical reasoning and code generation.

## Reinforcement Learning and Emergent Reasoning
A notable development in this domain is the introduction of DeepSeek-R1, a reasoning model that leverages reinforcement learning to enhance large language models reasoning abilities. The initial version, DeepSeek-R1-Zero, was trained exclusively using reinforcement learning without supervised fine-tuning. This approach led to the emergence of sophisticated reasoning behaviors, enabling the model to achieve significant improvements on benchmarks like the AIME 2024 mathematics test, where its pass rate increased from 15.6% to 71.0%. [[1][1]]

Building upon this foundation, DeepSeek-R1 incorporated a multi-stage training process that combined supervised learning with reinforcement learning. This strategy addressed earlier challenges such as poor readability and language mixing, resulting in a model that matches the performance of OpenAI's o1 across various reasoning tasks. [[1][1]]

## Advancements in Prompting Strategies
Another significant area of progress involves the development of advanced prompting strategies to enhance LLM reasoning. Techniques such as Chain-of-Thought reasoning have been explored to improve the logical deduction and multi-step reasoning capabilities of large language models by guiding them to generate intermediate reasoning steps before arriving at a final answer. It involves providing models with exemplars that demonstrate step-by-step problem-solving processes, enabling them to tackle complex tasks more effectively. Empirical studies have shown that Chain-of-Thought prompting significantly improves performance across various reasoning benchmarks, including arithmetic, commonsense, and symbolic reasoning tasks. [[2][2]] [[3][3]]

## Sources
[1]: https://arxiv.org/abs/2501.12948
[2]: https://www.arxiv.org/abs/2502.03671
[3]: https://arxiv.org/abs/2201.11903
