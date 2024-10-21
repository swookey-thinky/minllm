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
        self._token_file_template = "tokens_{shard_idx}_{context_length}.bin"
        self._data_type = np.uint16

        # Output shard size is 1 gigabyte
        self._output_shard_size = 1024**3

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
                    id="1kOgcs-gmUmqFPbCGnvVMkJXgLmMFivLD",
                    destination=first_shard_name,
                )
            else:
                # The raw corpus should have 74004228 examples
                string_dataset = BooksCorpus()

                # 1m entries per input shard
                self._input_shard_size = 1048576
                self._input_num_shards = (
                    len(string_dataset) // self._input_shard_size
                ) + (1 if len(string_dataset) % self._input_shard_size != 0 else 0)
                self._create_data_files(
                    string_dataset=string_dataset,
                    context_length=context_length,
                    tokenizer=tokenizer,
                )

        # Calculate the total size based on the size of all of the shards
        self._num_shards = 0
        self._data_length = 0
        while True:
            shard_name = os.path.join(
                self._token_file_name_root,
                self._token_file_template.format(
                    shard_idx=self._num_shards, context_length=context_length
                ),
            )

            if os.path.isfile(shard_name):
                self._data_length += (
                    os.stat(shard_name).st_size // self._data_type().itemsize
                )
                self._num_shards += 1
            else:
                break
        assert self._num_shards == 1

        # Only mmap the data once
        shard_file_name = os.path.join(
            self._token_file_name_root,
            self._token_file_template.format(
                shard_idx=0, context_length=self._context_length
            ),
        )
        self._shard_data = np.memmap(shard_file_name, dtype=self._data_type, mode="r")

    def __len__(self):
        # The last item we can get
        last_index = self._data_length - self._context_length - 1
        return last_index + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (self._data_length - self._context_length - 1):
            raise IndexError()

        x = torch.from_numpy(self._shard_data[idx : idx + self._context_length])
        y = torch.from_numpy(self._shard_data[idx + 1 : idx + 1 + self._context_length])
        assert x.shape[0] == self._context_length, f"{x.shape} {idx}"
        assert y.shape[0] == self._context_length, f"{y.shape} {idx}"
        return x, y

    def _sharded_getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (self._data_length - self._context_length - 1):
            raise IndexError()

        # Find the output shard this belongs to
        shard_idx = idx // self._output_shard_size
        assert shard_idx < self._num_shards

        # Does it span a shard?
        shard_offset = idx - shard_idx * self._output_shard_size
        assert shard_offset < self._output_shard_size

        if shard_offset + self._context_length + 1 >= self._output_shard_size:
            # The data spans two shards
            shard1_file_name = os.path.join(
                self._token_file_name_root,
                self._token_file_template.format(
                    shard_idx=shard_idx, context_length=self._context_length
                ),
            )
            shard2_file_name = os.path.join(
                self._token_file_name_root,
                self._token_file_template.format(
                    shard_idx=shard_idx + 1, context_length=self._context_length
                ),
            )
            shard1_data = np.memmap(shard1_file_name, dtype=self._data_type, mode="r")
            shard2_data = np.memmap(shard2_file_name, dtype=self._data_type, mode="r")

            shard2_length = (
                shard_offset + self._context_length + 1 - self._output_shard_size
            )
            data = np.concatenate(
                [
                    shard1_data[
                        shard_offset : shard_offset + self._context_length
                    ].astype(np.int64),
                    shard2_data[: shard2_length + 1].astype(np.int64),
                ]
            )

            x = torch.from_numpy(data[0 : self._context_length].astype(np.int64))
            y = torch.from_numpy(data[1 : 1 + self._context_length].astype(np.int64))
        else:
            # Data is fully contained in the shard
            shard_file_name = os.path.join(
                self._token_file_name_root,
                self._token_file_template.format(
                    shard_idx=shard_idx, context_length=self._context_length
                ),
            )
            data = np.memmap(shard_file_name, dtype=self._data_type, mode="r")
            x = torch.from_numpy(
                data[shard_offset : shard_offset + self._context_length].astype(
                    np.int64
                )
            )
            y = torch.from_numpy(
                data[shard_offset + 1 : shard_offset + 1 + self._context_length].astype(
                    np.int64
                )
            )
        assert x.shape[0] == self._context_length, f"{x.shape} {idx}"
        assert y.shape[0] == self._context_length, f"{y.shape} {idx}"
        return x, y

    def _write_output_shard(self, all_tokens, output_shard_idx: int):
        output_tokens = all_tokens[: self._output_shard_size]
        all_tokens = all_tokens[self._output_shard_size :]
        output_shard_size = len(output_tokens)

        shard_file_name = os.path.join(
            self._token_file_name_root,
            self._token_file_template.format(
                shard_idx=output_shard_idx, context_length=self._context_length
            ),
        )
        output_array = np.memmap(
            shard_file_name,
            dtype=self._data_type,
            mode="w+",
            shape=(output_shard_size,),
        )

        chunk_size = 1024
        for i in tqdm(
            range(
                (len(output_tokens) // chunk_size) + 1
                if len(output_tokens) % chunk_size != 0
                else 0
            ),
            desc=f"Writing Output Shard {output_shard_idx}",
            leave=False,
        ):
            chunk = np.array(
                output_tokens[i * chunk_size : i * chunk_size + chunk_size],
                dtype=self._data_type,
            )

            output_array[i * chunk_size : i * chunk_size + len(chunk)] = chunk
        output_array.flush()

        return all_tokens

    def _create_data_files(self, string_dataset, context_length, tokenizer):
        all_tokens = []
        output_shard_idx = 0
        for shard_idx in tqdm(range(self._input_num_shards), desc="Processing Shards"):
            shard_tokens = [
                token
                for row in [
                    tokenizer.encode(string_dataset[idx])
                    for idx in tqdm(
                        range(
                            shard_idx * self._input_shard_size,
                            min(
                                shard_idx * self._input_shard_size
                                + self._input_shard_size,
                                len(string_dataset),
                            ),
                        ),
                        desc="Tokenizing",
                        leave=False,
                    )
                ]
                for token in row
            ]
            all_tokens.extend(shard_tokens)

            # After each input shard, write the output shards, as much as we can
            while len(all_tokens) > self._output_shard_size:
                all_tokens = self._write_output_shard(
                    all_tokens, output_shard_idx=output_shard_idx
                )
                output_shard_idx += 1

        # Write any remaining data
        while len(all_tokens) > 0:
            all_tokens = self._write_output_shard(
                all_tokens, output_shard_idx=output_shard_idx
            )
            output_shard_idx += 1
