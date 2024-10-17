"""StoryCloze dataset, for Q&A evaluation.

This is not a dataset for training on.
"""

import json
import os
import numpy as np
import requests
import tarfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import load_dataset
from minllm.tokenizer.bpe.base import Tokenizer
from minllm.tokenizer import special_tokens


class StoryCloze(Dataset):
    def __init__(self, split: str = "validation"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._dataset_hf = load_dataset(
            "LSDSem/story_cloze",
            name="2016",
            split=split,
            trust_remote_code=True,
            data_dir="minllm/datasets/data",
        )
        self._data_length = len(self._dataset_hf)

    def __len__(self):
        return self._data_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._dataset_hf[idx]


class StoryClozeTokenized(Dataset):
    def __init__(
        self,
        root_dir,
        tokenizer: Tokenizer,
        context_length: int = 1024,
        split: str = "validation",
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        archive_base = "StoryCloze"
        self._token_file_name = os.path.join(
            root_dir, f"{archive_base}/tokens_{split}_{context_length}.bin"
        )
        self._label_file_name = os.path.join(
            root_dir, f"{archive_base}/labels_{split}_{context_length}.bin"
        )
        self._data_type = np.uint16
        self._context_length = context_length

        string_dataset = StoryCloze(split=split)
        self._data_length = len(string_dataset)

        if not os.path.isfile(self._token_file_name):
            os.makedirs(os.path.dirname(self._token_file_name), exist_ok=True)

            num_answers = 2
            all_examples = []
            all_labels = []

            for example_idx in tqdm(range(len(string_dataset)), desc="Tokenizing"):
                example = string_dataset[example_idx]

                context = (
                    example["input_sentence_1"]
                    + example["input_sentence_2"]
                    + example["input_sentence_3"]
                    + example["input_sentence_4"]
                )
                label = example["answer_right_ending"] - 1
                answers = [example["sentence_quiz1"], example["sentence_quiz2"]]
                assert len(answers) == num_answers

                context_tokens = tokenizer.encode(context)
                answers_tokens = [tokenizer.encode(a) for a in answers]

                # Pack the tokens into a tensor of length (num_answers, context_length)
                tokenized_example = np.zeros(
                    shape=(num_answers, context_length), dtype=self._data_type
                )
                for idx, answer_tokens in enumerate(answers_tokens):
                    example_length = (
                        1  # start token
                        + len(context_tokens)
                        + 1  # delimiter
                        + len(answer_tokens)
                        + 1  # classify
                    )

                    if example_length > context_length:
                        # Trim the start of the context tokens so that we can
                        # fit in the context length
                        diff = example_length - context_length
                        context_tokens = context_tokens[diff:]
                        example_length = (
                            1  # start token
                            + len(context_tokens)
                            + 1  # delimiter
                            + len(answer_tokens)
                            + 1  # classify
                        )

                    assert (
                        example_length <= context_length
                    ), f"{example_length} > {context_length}"

                    start_token = [tokenizer.special_tokens[special_tokens.GPT_START]]
                    assert len(start_token) == 1

                    delimiter_token = [
                        tokenizer.special_tokens[special_tokens.GPT_DELIMITER]
                    ]
                    assert len(delimiter_token) == 1

                    classify_token = [
                        tokenizer.special_tokens[special_tokens.GPT_CLASSIFY]
                    ]
                    assert len(classify_token) == 1

                    tokenized_example[idx, :example_length] = np.asarray(
                        start_token
                        + context_tokens
                        + delimiter_token
                        + answer_tokens
                        + classify_token
                    )
                all_examples.append(tokenized_example)
                all_labels.append(label)

            output_array = np.memmap(
                self._token_file_name,
                dtype=self._data_type,
                mode="w+",
                shape=(self._data_length, num_answers, context_length),
            )

            chunk_size = 1024
            for i in tqdm(
                range(self._data_length // chunk_size), desc="Writing output"
            ):
                chunk = np.array(
                    all_examples[i * chunk_size : i * chunk_size + chunk_size],
                    dtype=self._data_type,
                )
                output_array[i * chunk_size : i * chunk_size + len(chunk)] = chunk
            output_array.flush()

            # Write the labels array
            labels_array = np.memmap(
                self._label_file_name,
                dtype=self._data_type,
                mode="w+",
                shape=(self._data_length,),
            )
            for idx, label in enumerate(all_labels):
                labels_array[idx] = label
            labels_array.flush

        file_size = os.stat(self._token_file_name).st_size
        self._num_answers = file_size // (
            self._data_length * self._data_type().itemsize * self._context_length
        )

    def __len__(self):
        # The last item we can get
        return self._data_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = np.memmap(
            self._token_file_name,
            dtype=self._data_type,
            shape=(self._data_length, self._num_answers, self._context_length),
            mode="r",
        )
        labels = np.memmap(self._label_file_name, dtype=self._data_type, mode="r")
        x = torch.from_numpy(data[idx].astype(np.int64))
        y = torch.tensor(labels[idx], dtype=torch.int64)
        return x, y
