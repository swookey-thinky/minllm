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


class BooksCorpus(Dataset):
    def __init__(self):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._dataset_hf = load_dataset(
            "bookcorpus/bookcorpus", split="train", trust_remote_code=True
        )
        self._data_length = len(self._dataset_hf)

    def __len__(self):
        return self._data_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._dataset_hf[idx]["text"]


class BooksCorpusTokenized(Dataset):
    def __init__(self, root_dir, tokenizer: Tokenizer, context_length: int = 1024):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._context_length = context_length

        archive_base = "BooksCorpus"
        self._token_file_name = os.path.join(
            root_dir, f"{archive_base}/tokens_{context_length}.bin"
        )
        self._data_type = np.uint16

        if not os.path.isfile(self._token_file_name):
            os.makedirs(os.path.dirname(self._token_file_name), exist_ok=True)

            string_dataset = BooksCorpus()
            all_tokens = [
                token
                for row in [
                    tokenizer.encode(string_dataset[idx])
                    for idx in tqdm(range(len(string_dataset)), desc="Tokenizing")
                ]
                for token in row
            ]
            self._data_length = len(all_tokens)
            output_array = np.memmap(
                self._token_file_name,
                dtype=self._data_type,
                mode="w+",
                shape=(self._data_length,),
            )

            chunk_size = 1024
            for i in tqdm(
                range(self._data_length // chunk_size), desc="Writing output"
            ):
                chunk = np.array(
                    all_tokens[i * chunk_size : i * chunk_size + chunk_size],
                    dtype=self._data_type,
                )
                output_array[i * chunk_size : i * chunk_size + len(chunk)] = chunk
            output_array.flush()

        data = np.memmap(self._token_file_name, dtype=self._data_type, mode="r")
        self._data_length = data.size

    def __len__(self):
        # The last item we can get
        return self._data_length - self._context_length - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = np.memmap(self._token_file_name, dtype=self._data_type, mode="r")
        x = torch.from_numpy(data[idx : idx + self._context_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + 1 + self._context_length].astype(np.int64)
        )
        return x, y
