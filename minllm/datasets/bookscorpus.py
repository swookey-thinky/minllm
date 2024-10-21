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
from minllm.datasets.utils import download_file_from_google_drive


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
    def __init__(
        self,
        root_dir,
        tokenizer: Tokenizer,
        context_length: int = 1024,
        download: bool = True,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._context_length = context_length

        archive_base = "BooksCorpus"
        self._token_file_name_root = os.path.join(root_dir, f"{archive_base}")
        self._token_file_template = "tokens_{shard_idx}_{context_length}.npy"

        first_shard_name = os.path.join(
            self._token_file_name_root,
            self._token_file_template.format(
                shard_idx=0, context_length=context_length
            ),
        )

        if not os.path.isfile(first_shard_name):
            os.makedirs(os.path.dirname(first_shard_name), exist_ok=True)

            if download and context_length == 512:
                download_file_from_google_drive(
                    id="1Vo4YUB9vIZioa-Za0pNgOlP2vGcPVtrW",
                    destination=first_shard_name,
                )
            else:
                raise NotImplementedError()

        # Only mmap the data once
        shard_file_name = os.path.join(
            self._token_file_name_root,
            self._token_file_template.format(
                shard_idx=0, context_length=self._context_length
            ),
        )
        self._shard_data = np.load(shard_file_name)
        self._data_length = self._shard_data.shape[0]

    def __len__(self):
        # The last item we can get
        last_index = self._data_length - self._context_length - 1
        return last_index + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (self._data_length - self._context_length - 1):
            raise IndexError()

        x = torch.from_numpy(
            self._shard_data[idx : idx + self._context_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self._shard_data[idx + 1 : idx + 1 + self._context_length].astype(np.int64)
        )
        assert x.shape[0] == self._context_length, f"{x.shape} {idx}"
        assert y.shape[0] == self._context_length, f"{y.shape} {idx}"
        return x, y
