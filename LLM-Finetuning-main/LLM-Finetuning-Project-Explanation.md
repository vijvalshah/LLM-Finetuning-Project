# LLM Fine-tuning Project for Code Review Automation

## Overview
This project focuses on fine-tuning and prompting language models (LLMs) to automate code review activities. The project addresses two specific tasks:

1. **Review Comment Generation**: Given a code diff/change, generate a natural language review comment.
2. **Code Refinement Generation**: Given a code snippet and a review comment, generate improved/fixed code.

The project explores two approaches:
- **Part A**: QLoRA fine-tuning of open-source Llama 3 8B model
- **Part B**: Static metadata augmented few-shot prompting of GPT-3.5 Turbo

## Data Overview

The dataset is based on Microsoft's CodeReviewer paper (ESEC/FSE 2022) and includes:
- Code diffs (patches showing changes)
- Review comments
- Code summaries (added through processing)
- Function call graphs (added through processing)

## Input-Output Examples

### Review Comment Generation Task

**Raw Data Example:**
```json
{
  "patch": "@@ -34,7 +34,7 @@\n         # Validate input\n         if not check(root):\n             return None\n-        results = {}\n+        results = collections.defaultdict(list)\n         \n         # Build adjacency list\n         for node in root.children:",
  "msg": "Using a defaultdict here is a good improvement as it makes the code more concise by avoiding key existence checks later on.",
  "summary": "This function processes a tree structure and builds an adjacency list for further operations.",
  "callgraph": "[parse_tree->[check,build_adjacency_list,process_result],build_adjacency_list->[traverse]]"
}
```

**Processed for Fine-tuning (Instruction Format):**
```json
{
  "instruction": "You are a powerful code reviewer model. Your job is to suggest review comment in natural language. You are given a question, and context regarding a diff hunk or code change in programming language. You must output appropriate, contextual review comment for that code change.",
  "input": "<del>        results = {}</del><add>        results = collections.defaultdict(list)</add>",
  "output": "Using a defaultdict here is a good improvement as it makes the code more concise by avoiding key existence checks later on."
}
```

**Processed for Few-shot Prompting:**
```
[Code To Be Reviewed]: 
<del>        results = {}</del>
<add>        results = collections.defaultdict(list)</add>

[Summary]: This function processes a tree structure and builds an adjacency list for further operations.

[Callgraph]: [parse_tree->[check,build_adjacency_list,process_result],build_adjacency_list->[traverse]]

[Review Comment]: <s> ? </s>
```

### Code Refinement Generation Task

**Raw Data Example:**
```json
{
  "old": "@@ -34,7 +34,7 @@\n         # Validate input\n         if not check(root):\n             return None\n-        results = {}\n         \n         # Build adjacency list\n         for node in root.children:",
  "new": "@@ -34,7 +34,7 @@\n         # Validate input\n         if not check(root):\n             return None\n+        results = collections.defaultdict(list)\n         \n         # Build adjacency list\n         for node in root.children:",
  "comment": "Consider using a defaultdict to make the code more concise by avoiding explicit key existence checks.",
  "summary": "This function processes a tree structure and builds an adjacency list for further operations.",
  "callgraph": "[parse_tree->[check,build_adjacency_list,process_result],build_adjacency_list->[traverse]]"
}
```

**Processed for Fine-tuning (Instruction Format):**
```json
{
  "instruction": "You are a powerful code reviewer model. Your job is to suggest refined or fixed code based on the natural language review comment. You are given a question, and context regarding an old diff hunk or code change in programming language. You are also given a review comment based on that old code. You must output accurate refined, fixed new code snippet for that old code change and corresponding review comment in the same programming language as the old code.",
  "input": "Review Comment: Consider using a defaultdict to make the code more concise by avoiding explicit key existence checks.\nOld Code: # Validate input\n if not check(root):\n     return None\n results = {}\n \n # Build adjacency list\n for node in root.children:",
  "output": "New Code: # Validate input\n if not check(root):\n     return None\n results = collections.defaultdict(list)\n \n # Build adjacency list\n for node in root.children:"
}
```

**Processed for Few-shot Prompting:**
```
[Before Refinement]: 
# Validate input
if not check(root):
    return None
results = {}

# Build adjacency list
for node in root.children:

[Summary]: This function processes a tree structure and builds an adjacency list for further operations.

[Callgraph]: [parse_tree->[check,build_adjacency_list,process_result],build_adjacency_list->[traverse]]

[Review Comment]: Consider using a defaultdict to make the code more concise by avoiding explicit key existence checks.

[After Refinement]: <s> ? </s>
```

## Project Structure

```
LLM-Finetuning-main/
├── Fine-tuning/                  # Fine-tuning related code
│   ├── dataset-preprocess.ipynb  # Preprocesses data into instruction format
│   ├── llama-3-train-test.ipynb  # Fine-tuning and testing code for Llama 3
│   └── Finetune_Prompt.jpeg      # Visualization of prompt structure
├── Metric/                       # Evaluation metrics code
│   ├── evaluate.py               # Main evaluation script
│   ├── smooth_bleu.py            # BLEU score implementation
│   └── stopwords.txt             # Stopwords for evaluation
├── Prompting/                    # Few-shot prompting code
│   ├── prompt_experiment_script.py  # Prompting implementation
│   ├── run_experiment.sh         # Script to run prompting experiments
│   ├── code-review pipeline.png  # Review comment generation diagram
│   └── code-refinement pipeline.png  # Code refinement diagram
├── Refinement/                   # Processed data for refinement task
│   ├── ref-test-5000-tuned.jsonl    # Test data in instruction format
│   ├── ref-test-5000-merged.jsonl   # Test data with metadata
│   ├── ref-5000-pred.txt            # Model predictions
│   ├── ref-5000-output.txt          # Raw model outputs
│   └── ref-5000-gold.txt            # Gold/reference answers
└── Review/                       # Processed data for review task
    ├── msg-test-5000-tuned.jsonl    # Test data in instruction format  
    ├── msg-test-5000-merged.jsonl   # Test data with metadata
    ├── msg-5000-pred.txt            # Model predictions
    ├── msg-5000-output.txt          # Raw model outputs
    └── msg-5000-gold.txt            # Gold/reference answers
```

## Part A: Fine-tuning Llama 3 8B with QLoRA

### Data Preprocessing
The dataset is converted into an instruction-following format suitable for supervised fine-tuning:

**Review Comment Generation Task:**
```
instruction: <prompt instructing the model to generate review comments>
input: <diff hunk/code change>
output: <review comment>
```

**Code Refinement Generation Task:**
```
instruction: <prompt instructing the model to refine code>
input: <review comment, old diff hunk/code change>
output: <new diff hunk/code change>
```

The implementation for this preprocessing is in `Fine-tuning/dataset-preprocess.ipynb`.

### Fine-tuning Process

The project uses Parameter-Efficient Fine-tuning (PEFT) with QLoRA (Quantized Low-Rank Adaptation):
1. The base model is Llama 3 8B
2. 4-bit quantization is used to reduce memory requirements
3. LoRA adapters are applied to attention and MLP layers
4. Fine-tuning is done using the Unsloth framework for efficiency

Key hyperparameters:
- LoRA rank: 32
- LoRA alpha: 16
- Learning rate: 2e-4
- Batch size: 8
- Training steps: ~500-1000

The implementation is in `Fine-tuning/llama-3-train-test.ipynb`.

### Inference

After fine-tuning, the model is used to generate:
1. Review comments for code changes
2. Refined code based on review comments

## Part B: Few-shot Prompting with GPT-3.5 Turbo

### Data Augmentation

The prompting approach enriches the context by adding:
1. **Function Call Graph**: Shows function relationships and dependencies
2. **Code Summary**: Provides a natural language summary of code purpose

### BM25 Retrieval for Few-shot Examples

The process uses BM25 information retrieval to find the most relevant examples:
1. Tokenize the test query (code to be reviewed)
2. Calculate BM25 scores against the training examples
3. Select top-k (3 or 5) most relevant examples as few-shot demonstrations
4. Format these examples in a prompt template

### Prompt Structure

**Review Comment Generation Task:**
```
[Code To Be Reviewed]: <diff hunk/code change>
[Summary]: <code summary>
[Callgraph]: <function call graph>
[Review Comment]: <s> ? </s>
```

**Code Refinement Generation Task:**
```
[Before Refinement]: <old diff hunk/code change>
[Summary]: <code summary>  
[Callgraph]: <function call graph>
[Review Comment]: <review comment>
[After Refinement]: <s> ? </s>
```

The implementation is in `Prompting/prompt_experiment_script.py`.

## Evaluation Metrics

The project uses the following metrics to evaluate performance:

1. **BLEU-4 Score**: Measures n-gram overlap between prediction and reference
   - Implementation in `Metric/smooth_bleu.py`
   - Can be run with or without stopword removal

2. **BERTScore**: Measures semantic similarity using BERT embeddings
   - Not directly shown in code but mentioned in the README

3. **Exact Match (EM)**: For refinement task only - percentage of predictions that exactly match references

## Results

**Review Comment Generation Task:**
| Model | BLEU-4 | BERTScore |
|-------|--------|-----------|
| CodeReviewer (223M) | 4.28 | 0.8348 |
| Llama 3 (8B) Fine-tuned | 5.27 | 0.8476 |
| GPT-3.5 Turbo (175B) Prompted | 8.27 | 0.8515 |

**Code Refinement Generation Task:**
| Model | BLEU-4 | EM | BERTScore |
|-------|--------|-------------|-----------|
| CodeReviewer (223M) | 83.61 | 0.308 | 0.9776 |
| Llama 3 (8B) Fine-tuned | 80.47 | 0.237 | 0.9745 |
| GPT-3.5 Turbo (175B) Prompted | 79.46 | 0.107 | 0.9704 |

## How to Use This Project

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU with at least 16GB VRAM for fine-tuning
- OpenAI API key for prompting experiments

### Fine-tuning Process
1. Prepare data in instruction format using `dataset-preprocess.ipynb`
2. Run fine-tuning with `llama-3-train-test.ipynb`
3. Evaluate results with scripts in the `Metric` directory

### Prompting Process
1. Set up OpenAI API credentials
2. Configure experiment parameters in `run_experiment.sh`
3. Run experiments with `prompt_experiment_script.py`
4. Evaluate results with the same metric scripts

## Key Takeaways

1. **Fine-tuning vs. Prompting**:
   - Fine-tuning excels in structured code refinement tasks
   - Prompting with larger models performs better for review comment generation

2. **Data Enrichment**:
   - Adding code summaries and call graphs improves model performance
   - BM25 retrieval helps select relevant examples for few-shot learning

3. **Model Size Considerations**:
   - Smaller models (8B) can be effectively fine-tuned with QLoRA
   - Larger models (175B) can achieve good results with just prompting

4. **Task Characteristics**:
   - Review comment generation benefits more from semantic understanding (larger models)
   - Code refinement benefits from specific training on programming patterns 

## Technical Deep Dive

### Understanding QLoRA and PEFT

QLoRA (Quantized Low-Rank Adaptation) is a technique that makes fine-tuning large language models more efficient:

1. **Quantization**: The base model weights are quantized to 4-bit precision, dramatically reducing memory requirements. This allows fine-tuning on consumer GPUs with limited VRAM.

2. **Low-Rank Adaptation**: Instead of updating all model weights during fine-tuning:
   - Small adapter matrices (LoRA) are added to specific layers (attention and MLP)
   - Only these adapter weights are updated during training
   - The base model remains frozen, saving memory and computation

3. **Key Parameters Explained**:
   - `r`: LoRA rank - controls the capacity of adapters (higher = more capacity but more parameters)
   - `alpha`: LoRA alpha - scales the LoRA updates, affects training dynamics
   - `target_modules`: Which layers receive LoRA adapters
   - `load_in_4bit`: Enables 4-bit quantization of the base model

4. **Unsloth Optimization**: The Unsloth framework provides several optimizations:
   - Faster inference and training for Llama models
   - Optimized kernel implementations
   - Memory-efficient gradient checkpointing
   - Easy API for setting up QLoRA

### BM25 Retrieval Algorithm

BM25 (Best Matching 25) is a bag-of-words retrieval algorithm used to find relevant examples for few-shot prompting:

1. **How BM25 Works**:
   - Calculate term frequency (TF) for each word in the document
   - Apply inverse document frequency (IDF) to prioritize rare words
   - Use saturation function to prevent common words from dominating
   - Combine these factors to score document relevance to a query

2. **Implementation in this Project**:
   ```python
   from rank_bm25 import BM25Okapi
   
   # Tokenize corpus (training examples)
   tokenized_corpus = [doc.split(" ") for doc in train_code_old]
   bm25 = BM25Okapi(tokenized_corpus)
   
   # For each test example, find most relevant training examples
   tokenized_query = test_code.split(" ")
   scores = bm25.get_scores(tokenized_query)
   top_k_indices = scores.argsort()[-num_examples:][::-1]
   ```

3. **Advantages for Few-Shot Prompting**:
   - Finds semantically similar code examples
   - Works well for code by matching variable names, function calls, and patterns
   - Fast and efficient without requiring embeddings or neural networks

### Obtaining and Preparing the Dataset

The original CodeReviewer dataset can be obtained from the Microsoft research team:

1. **Dataset Source**:
   - GitHub: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
   - Paper: https://arxiv.org/pdf/2203.09095.pdf

2. **Preprocessing Steps**:
   - **For Review Comment Generation**:
     1. Extract code diffs (patches)
     2. Process diff lines to identify adds/deletes/unchanged code
     3. Format with special tokens: `<add>`, `<del>`, `<keep>`
     4. Pair with corresponding review comments

   - **For Code Refinement**:
     1. Extract old code snippets
     2. Extract new (refined) code snippets
     3. Pair with review comments
     4. Format as instruction tuning data

3. **Metadata Generation**:
   - **Function Call Graph**:
     - Generated using Tree-sitter parser
     - Represented as a string showing function call relationships
     - Example: `[funcA->[funcB,funcC], funcB->[funcD]]`
   
   - **Code Summary**:
     - Generated using CodeT5 model
     - Provides natural language explanation of code functionality

### Downloading and Using the Models

1. **Llama 3 8B Model**:
   - Access through Hugging Face: `meta-llama/Meta-Llama-3-8B`
   - Requires acceptance of Meta's usage terms
   - Use with unsloth for 4-bit quantization:
     ```python
     from unsloth import FastLanguageModel
     model, tokenizer = FastLanguageModel.from_pretrained(
         model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
         max_seq_length = 2048,
         load_in_4bit = True,
     )
     ```

2. **GPT-3.5 Turbo**:
   - Requires OpenAI API key: https://platform.openai.com
   - Environment variable setup:
     ```
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - API usage in Python:
     ```python
     import openai
     openai.api_key = "your-api-key-here"
     response = openai.Completion.create(
         engine="gpt-3.5-turbo-instruct",
         prompt=context,
         temperature=0.5,
         max_tokens=250,
     )
     ```

### Running Evaluation

1. **BLEU Score Calculation**:
   ```python
   from smooth_bleu import bleu_fromstr
   
   # Calculate BLEU between predictions and references
   bleu = bleu_fromstr(predictions, references, rmstop=True)
   print(f"BLEU-4 Score: {bleu}")
   ```

2. **BERTScore Calculation**:
   ```python
   from bert_score import score
   
   # Calculate BERTScore
   P, R, F1 = score(predictions, references, lang="en", verbose=True)
   print(f"BERTScore F1: {F1.mean().item()}")
   ```

3. **Exact Match Calculation**:
   ```python
   # Calculate exact match percentage
   exact_matches = sum(1 for p, g in zip(predictions, golds) if p.strip() == g.strip())
   em_score = exact_matches / len(predictions)
   print(f"Exact Match: {em_score}")
   ```

## Additional Learning Resources

### Key Papers

1. **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
   - Explains 4-bit quantization and low-rank adaptation techniques

2. **CodeReviewer**: [CodeReviewer: Pre-Training for Automating Code Review Activities](https://arxiv.org/abs/2203.09095)
   - Original paper describing the dataset and baseline model

3. **Retrieval-Augmented Generation**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
   - Background on using retrieval to enhance language model outputs

### Tutorials and Guides

1. **Hugging Face PEFT Guide**: [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/index)
   - Complete documentation for implementing PEFT techniques

2. **Unsloth Documentation**: [Unsloth GitHub](https://github.com/unslothai/unsloth)
   - Efficient fine-tuning framework for Llama models

3. **OpenAI API Documentation**: [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
   - Guide for using GPT models through the API

### Related GitHub Repositories

1. **CodeBERT**: [microsoft/CodeBERT](https://github.com/microsoft/CodeBERT)
   - Contains the CodeReviewer implementation and dataset

2. **Tree-sitter**: [tree-sitter/tree-sitter](https://github.com/tree-sitter/tree-sitter)
   - Tool used for generating function call graphs

3. **CodeT5**: [salesforce/CodeT5](https://github.com/salesforce/CodeT5)
   - Model used for generating code summaries

## Common Issues and Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size and sequence length
   - Use gradient accumulation steps
   - Ensure proper 4-bit quantization setup

2. **Poor Performance**:
   - Check for data quality and preprocessing issues
   - Experiment with different learning rates and training steps
   - Try different prompt templates for few-shot learning

3. **OpenAI API Rate Limits**:
   - Implement exponential backoff for retries
   - Add sleep between API calls
   - Consider using higher tier API access

4. **Evaluation Discrepancies**:
   - Ensure consistent tokenization between training and evaluation
   - Check for stopword handling in BLEU calculation
   - Normalize whitespace and code formatting before comparison 