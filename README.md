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
| :---- | :---- | ----- | ------ | -----
| February 2019 | GPT-2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [config](https://github.com/swookey-thinky/minllm/blob/main/configs/tinyshakespeare/gpt2.yaml) |

## Datasets

| Name | Source | Used By | Description
| :--- | :----- | :------ | :----------

## Benchmarks

| Name | Source | Description
| :--- | :----- | -----------

