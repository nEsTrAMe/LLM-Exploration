# Seminar Software Engineering

In recent years, large language models have achieved remarkable advancements, particularly in enhancing their reasoning capabilities. These improvements stem from innovative training methodologies, including reinforcement learning and self-reflection techniques, leading to superior performance in tasks requiring logical reasoning and code generation.

TODO: What will be done in this project, what are the main goals of it

### Reinforcement Learning and Emergent Reasoning
A notable development in this domain is the introduction of DeepSeek-R1, a reasoning model that leverages reinforcement learning to enhance large language models reasoning abilities. The initial version, DeepSeek-R1-Zero, was trained exclusively using reinforcement learning without supervised fine-tuning. This approach led to the emergence of sophisticated reasoning behaviors, enabling the model to achieve significant improvements on benchmarks like the AIME 2024 mathematics test, where its pass rate increased from 15.6% to 71.0%. [[1][1]]

Building upon this foundation, DeepSeek-R1 incorporated a multi-stage training process that combined supervised learning with reinforcement learning. This strategy addressed earlier challenges such as poor readability and language mixing, resulting in a model that matches the performance of OpenAI's o1 across various reasoning tasks. [[1][1]]

### Advancements in Prompting Strategies
Another significant area of progress involves the development of advanced prompting strategies to enhance LLM reasoning. Techniques such as Chain-of-Thought reasoning have been explored to improve the logical deduction and multi-step reasoning capabilities of large language models by guiding them to generate intermediate reasoning steps before arriving at a final answer. It involves providing models with exemplars that demonstrate step-by-step problem-solving processes, enabling them to tackle complex tasks more effectively. Empirical studies have shown that Chain-of-Thought prompting significantly improves performance across various reasoning benchmarks, including arithmetic, commonsense, and symbolic reasoning tasks. [[2][2]] [[3][3]]

## Benchmarking large language models
To see how the large language models perform in vulnerability repair and if the performance can be improved with different prompting strategies, we have to the chosen models. To run them, [Ubelix](https://hpc-unibe-ch.github.io/), a high performance computer, which is available for research and student projects at the university of Bern, is used.

#### First steps
To get familiar with Batch jobs, a simple Slurm job was submitted to and run:
```Bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

# Put your code below this line
module load Workspace_Home
echo "Hello, UBELIX from node $(hostname)" > hello.txt
```
A detailed guide about Batch Jobs can be found [here](https://help.jasmin.ac.uk/docs/batch-computing/).

#### Running first model
As a fist model, [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) was chosen as it is a model with a rather low amount of parameters.

`module avail` was used to check which modules are already installed on Ubelix and which have to be installed. Fortunately, all the modules which are used to run the model are already installed and ready to use.

To get a stable and reproductive environment a Python virtual environment is used:

> [!NOTE]  
> In order to keep the overview, a default python virtaul environment was created, in which the dependencies of all used models are installed 

```Bash
# load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# create virtual environment
python -m venv llm_env
source llm_env/bin/activate
```

Install the necessary dependencies:
```Bash
pip install torch
pip install transformers
pip install accelerate
pip install evaluate
```

To run the model, the batch job [run_llama.sh](first_model/run_llama.sh) and a corresponding [python scipt](first_model/run_llama_inference.py) was created.

After running the job (`sbatch run_llama.sh`), a slurm log file is created with the output of the job. E.g. 
```
{'role': 'assistant', 'content': "Arrr, me hearty! Yer lookin' fer me, eh? Well, I be Captain Cutlass, the scurviest and most cunning pirate to ever sail the seven seas! Me trusty parrot, Polly, be me loyal sidekick and me most trusted advisor. We've had us many a grand adventure together, plunderin' treasure and battlin' scurvy dogs. So, what be bringin' ye to these waters?"}
```

#### Running a Qwen2.5-Coder model
As a second model, [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) was used. This model has significantly improvements in code generation, code reasoning and code fixing and does have more parameters than the previously used. Therefore it is suitable for this project.


To run the model, the batch job [run_qwen.sh](first_model/run_qwen.sh) and a corresponding [python scipt](qwen/run_qwen_inference.py) was created.

After running the job (`sbatch run_qwen.sh`), a slurm log file is created with the output of the job. E.g. 

```
Sure! Quick Sort is a popular and efficient sorting algorithm that uses a divide-and-conquer approach to sort elements. Here's a Python implementation of the Quick Sort algorithm:

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

# Example usage:
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print(sorted_arr)


### Explanation:
1. **Base Case**: If the array has 0 or 1 elements, it is already sorted, so we return it as is.
2. **Pivot Selection**: We choose the pivot element from the array. In this example, we select the middle element.
3. **Partitioning**: We create three sub-arrays:
   - `left`: All elements less than the pivot.
   - `middle`: All elements equal to the pivot.
   - `right`: All elements greater than the pivot.
4. **Recursive Sorting**: We recursively apply the `quick_sort` function to the `left` and `right` sub-arrays.
5. **Concatenation**: Finally, we concatenate the sorted `left` sub-array, the `middle` sub-array, and the sorted `right` sub-array to get the final sorted array.

This implementation is simple and easy to understand, but it may not be the most efficient in terms of space complexity due to the use of additional lists. For an in-place version, you can modify the algorithm to avoid creating new lists.

```

#### Working with hugging face datasets
Install the necessary dependencies:
```Bash
pip install datasets
```
After the installation, the hugging face datasets python library can be used to load to the data. Due to performance, the meta-llama/Llama-3.2-1B-Instruct is used to get familiar with huggingface datasets. 
In [run_llama_interface_datasets.py](fist_model/run_llama_interface_datasets.py) the [Code_Vulnerability_Security_DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) is loaded. This is a dataset created to train models for secure coding. In this case we will use the rejected answers as input and ask the model to detect and fix the vulnerability. Furthermore the chosen anser can be used as the expected solution.



## Sources
[1]: https://arxiv.org/abs/2501.12948
[2]: https://www.arxiv.org/abs/2502.03671
[3]: https://arxiv.org/abs/2201.11903
