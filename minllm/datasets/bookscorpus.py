from datasets import load_dataset
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        split: str = "train",
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
        self._train_file_name = os.path.join(
            self._token_file_name_root, f"tokens_{tokenizer.name}_train.bin"
        )
        self._val_file_name = os.path.join(
            self._token_file_name_root, f"tokens_{tokenizer.name}_val.bin"
        )

        if not os.path.isfile(self._train_file_name):
            os.makedirs(os.path.dirname(self._train_file_name), exist_ok=True)

            train_download_ids = {
                "spacy": "1Vo4YUB9vIZioa-Za0pNgOlP2vGcPVtrW",
                "gpt2": "",
            }
            val_download_ids = {
                "spacy": "1Vo4YUB9vIZioa-Za0pNgOlP2vGcPVtrW",
                "gpt2": "",
            }

            if download and tokenizer.name() in train_download_ids:
                download_file_from_google_drive(
                    id=train_download_ids[tokenizer.name()],
                    destination=self._train_file_name,
                )
                download_file_from_google_drive(
                    id=val_download_ids[tokenizer.name()],
                    destination=self._val_file_name,
                )
            else:
                self._prepare(tokenizer=tokenizer)

        if split == "train":
            self._shard_data = np.load(self._train_file_name, allow_pickle=True)
            self._data_length = self._shard_data.shape[0]
        else:
            self._shard_data = np.load(self._val_file_name, allow_pickle=True)
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

    def _prepare(self, tokenizer: Tokenizer):
        num_proc = 8
        num_proc_load_dataset = num_proc

        dataset = load_dataset(
            "bookcorpus/bookcorpus",
            trust_remote_code=True,
            num_proc=num_proc_load_dataset,
        )
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

        # Tokenize the dataset
        def process(example):
            ids = tokenizer.encode(example["text"], append_eot=True)
            out = {"ids": ids, "len": len(ids)}
            return out

        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="Tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(
                self._token_file_name_root, f"tokens_{tokenizer.name}_{split}.bin"
            )
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
