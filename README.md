# minllm
A minimum reproduction of LLM research.

## Requirements

This package built using PyTorch and written in Python 3. To setup an environment to run all of the lessons, we suggest using conda or venv:

```
> python3 -m venv minllm
> source minllm/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run from the root of the repository, and you should set your python path to include the repository root:

```
> export PYTHONPATH=$(pwd)
```

If you have issues with PyTorch and different CUDA versions on your instance, make sure to install the correct version of PyTorch for the CUDA version on your machine. For example, if you have CUDA 11.8 installed, you can install PyTorch using:

```
> pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Papers

| Date  | Name  | Paper | Config | Instructions
| ----: | ---- | ----- | ------ | -----
| June 2018 | GPT | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | |
| October 2018 | BERT | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | |
| February 2019 | GPT-2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [config](https://github.com/swookey-thinky/minllm/blob/main/configs/tinyshakespeare/gpt2.yaml) |
| October 2019 | BART | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) | |
| May 2020 | GPT-3 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | |
| May 2020 | RAG | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | |
| December 2021 | Gopher | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) | |
| January 2022 | LaMDA | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) | |
| March 2022 | Chinchilla | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | |
| March 2022 | Instruct-GPT | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | |
| April 2022 | PaLM | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | |
| April 2022 | Flamingo | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | |
| February 2023 | LLaMA | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | |
| March 2023 | GPT-4 | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | |
| July 2023 | LLaMA 2 | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | |
| August 2023 | Code LLaMA | [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950) | |
| February 2024 | OLMo | [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) | |
| July 2024 | LLaMA 3 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | |
| September 2024 | Molmo | [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://molmo.allenai.org/paper.pdf) | |

## Datasets

| Name | Source | Used By | Description
| :--- | :----- | :------ | :----------

## Benchmarks

| Name | Source | Description
| :--- | :----- | -----------

