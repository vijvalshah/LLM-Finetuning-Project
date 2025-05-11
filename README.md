# LLM Finetuning for Automated Code Review

## Project Overview

This project focuses on enhancing automated code review capabilities through the finetuning of Language Learning Models (LLMs). My approach involves two key tasks: generating meaningful review comments and producing improved code refinements. I've utilized a dataset from Microsoft's CodeReviewer research for this work, using a sample of 5000 entries to efficiently demonstrate results.

## Methodology

### Approach 1: Fine-tuning Llama 3 8B with QLoRA

To effectively adapt a powerful model to code review tasks while keeping resource requirements manageable, I implemented Parameter Efficient Fine-tuning on Llama 3 (8B) using the QLoRA approach.

#### Data Preparation

I transformed the datasets into instruction-following format, with specific structures for each task:

- **For Review Comment Generation:**
  ```
  instruction: <prompt>
  input: <code change>
  output: <review comment>
  ```

- **For Code Refinement Generation:**
  ```
  instruction: <prompt>
  input: <review comment, original code>
  output: <improved code>
  ```

Dataset files are available in the [Review](/Review/) and [Refinement](/Refinement/) directories.

#### Training and Inference

I used the [unsloth](https://github.com/unslothai/unsloth) framework for efficient training on consumer-grade hardware with 16GB VRAM. The detailed training configuration and hyperparameters can be found in the [fine-tuning notebook](/Fine-tuning/llama-3-train-test.ipynb).

![Training Approach](/Fine-tuning/Finetune_Prompt.jpeg)

#### Results Evaluation

Using BLEU-4 and BERTScore metrics (plus Exact Match for refinement tasks), my fine-tuned model demonstrated these results:

**Review Comment Generation:**

| Model | BLEU-4 | BERTScore |
|-------|--------|-----------|
| CodeReviewer (223M) | 4.28 | 0.8348 |
| My Llama 3 (8B) | 5.27 | 0.8476 |

**Code Refinement Generation:**

| Model | BLEU-4 | EM | BERTScore |
|-------|--------|-------|-----------|
| CodeReviewer (223M) | 83.61 | 0.308 | 0.9776 |
| My Llama 3 (8B) | 80.47 | 0.237 | 0.9745 |

My instruction-tuned Llama 3 model outperformed the baseline CodeReviewer in review comment generation while showing competitive performance on code refinement.

### Approach 2: Metadata-Enhanced Prompting with GPT-3.5

I also explored a complementary approach using few-shot prompting with GPT-3.5 Turbo Instruct.

#### Enhanced Prompting Strategy

I enhanced the prompts with static metadata including:
- Function call graphs (generated using tree-sitter)
- Code summaries (generated with CodeT5)

This enriched context allows the model to better understand code structure and purpose.

**Prompt Formats:**
- For review comments, I included code changes, function call graphs, and code summaries
- For refinements, I added review comments to the above elements

![Review Workflow](/Prompting/code-review%20pipeline.png)
![Refinement Workflow](/Prompting/code-refinement%20pipeline.png)

#### Retrieval-Based Few-Shot Examples

For each test sample, I selected the most relevant examples from the training set using BM25 information retrieval, then experimented with 3-shot and 5-shot prompting at different temperature settings.

The prompting experiments are implemented in [this script](/Prompting/prompt_experiment_script.py), with automation handled by [run_experiment.sh](/Prompting/run_experiment.sh).

#### Comparative Results

**Review Comment Generation:**

| Model | BLEU-4 | BERTScore |
|-------|--------|-----------|
| CodeReviewer (223M) | 4.28 | 0.8348 |
| My Llama 3 (8B) | 5.27 | 0.8476 |
| My GPT-3.5 Approach | 8.27 | 0.8515 |

**Code Refinement Generation:**

| Model | BLEU-4 | EM | BERTScore |
|-------|--------|-------|-----------|
| CodeReviewer (223M) | 83.61 | 0.308 | 0.9776 |
| My Llama 3 (8B) | 80.47 | 0.237 | 0.9745 |
| My GPT-3.5 Approach | 79.46 | 0.107 | 0.9704 |

## Conclusions

My metadata-enhanced prompting approach with GPT-3.5 significantly improved review comment generation performance over both baseline and fine-tuned models. However, for code refinement, my fine-tuned Llama 3 model proved more effective than prompting GPT-3.5, though neither surpassed the baseline for this specific task.

These findings suggest different optimal approaches depending on the specific code review activity. Future work could explore more sophisticated retrieval-augmented generation techniques to further enhance performance.
