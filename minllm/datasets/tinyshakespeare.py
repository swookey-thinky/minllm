import json
import os
import numpy as np
import requests
import tarfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from minllm.datasets.utils import download_file_from_google_drive


class TinyShakespeare(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        archive_base = "TinyShakespeare"
        archive_file_name = os.path.join(root_dir, f"{archive_base}/input.txt")

        if not os.path.isfile(archive_file_name):
            os.makedirs(os.path.dirname(archive_file_name), exist_ok=True)
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(archive_file_name, "w", encoding="utf-8") as f:
                f.write(requests.get(data_url).text)

        with open(archive_file_name, "r", encoding="utf-8") as f:
            data = f.read()
        self._data_length = len(data)
        self._data = data

    def __len__(self):
        return self._data_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._data[idx]


class TinyShakespeareTokenized(Dataset):
    def __init__(self, root_dir, context_length: int = 1024):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self._context_length = context_length

        archive_base = "TinyShakespeare"
        token_file_name = os.path.join(root_dir, f"{archive_base}/tokens.bin")
        archive_file_name = os.path.join(root_dir, f"{archive_base}/input.txt")

        if not os.path.isfile(token_file_name):
            os.makedirs(os.path.dirname(archive_file_name), exist_ok=True)
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(archive_file_name, "w", encoding="utf-8") as f:
                f.write(requests.get(data_url).text)

            with open(archive_file_name, "r", encoding="utf-8") as f:
                data = f.read()

            # Now tokenize
            import tiktoken

            token_encoder = tiktoken.get_encoding("gpt2")
            tokens = token_encoder.encode_ordinary(data)
            tokens = np.array(tokens, dtype=np.uint16)
            tokens.tofile(token_file_name)

        self._data = np.fromfile(token_file_name, dtype=np.uint16)
        self._data_length = self._data.shape[0]

    def __len__(self):
        # The last item we can get
        return self._data_length - self._context_length - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.from_numpy(
            self._data[idx : idx + self._context_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self._data[idx + 1 : idx + 1 + self._context_length].astype(np.int64)
        )
        return x, y
