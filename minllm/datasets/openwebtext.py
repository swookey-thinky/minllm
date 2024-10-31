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


class OpenWebTextTokenized(Dataset):
    def __init__(
        self,
        root_dir,
        tokenizer: Tokenizer,
        context_length: int = 1024,
        download: bool = True,
        split: str = "train",
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in ["train", "validation"]
        self._context_length = context_length

        archive_base = "OpenWebText"
        self._token_file_name_root = os.path.join(root_dir, f"{archive_base}")
        self._token_train_file_name = "tokens_train.bin"
        self._token_val_file_name = "tokens_val.bin"

        self._train_file_name = os.path.join(
            self._token_file_name_root, self._token_train_file_name
        )

        self._val_file_name = os.path.join(
            self._token_file_name_root, self._token_val_file_name
        )

        if not os.path.isfile(self._train_file_name):
            if download:
                os.makedirs(os.path.dirname(self._train_file_name), exist_ok=True)
                assert tokenizer.name() == "tiktoken_gpt2"
                download_file_from_google_drive(
                    id="1TYD2qEsdRVxA0SSzOP3JHltzyVJMNzQw",
                    destination=self._train_file_name,
                )
                download_file_from_google_drive(
                    id="1NHCc8_nVv-WlNWZjBXwv6w1wx0oGyLzp",
                    destination=self._val_file_name,
                )
            else:
                raise NotImplementedError()

        self._data_file_name = (
            self._train_file_name if split == "train" else self._val_file_name
        )
        data = np.memmap(self._data_file_name, dtype=np.uint16, mode="r")
        self._data_length = len(data)

    def __len__(self):
        # The last item we can get
        last_index = self._data_length - self._context_length - 1
        return last_index + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (self._data_length - self._context_length - 1):
            raise IndexError()

        data = np.memmap(self._data_file_name, dtype=np.uint16, mode="r")
        x = torch.from_numpy(data[idx : idx + self._context_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + 1 + self._context_length].astype(np.int64)
        )
        assert x.shape[0] == self._context_length, f"{x.shape} {idx}"
        assert y.shape[0] == self._context_length, f"{y.shape} {idx}"
        return x, y
