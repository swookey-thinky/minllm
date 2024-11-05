from datasets import load_dataset
import glob
from huggingface_hub import hf_hub_download
import json
import os
import numpy as np
import requests
import tarfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from minllm.tokenizer.bpe.base import Tokenizer
from minllm.datasets.utils import download_file_from_google_drive


class FineWeb10BTokenized(Dataset):
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

        archive_base = "FineWeb10B"
        self._num_train_files = 103
        self._num_val_files = 1
        self._token_file_name_root = os.path.join(root_dir, f"{archive_base}")
        self._token_train_file_template = "fineweb_train_{idx:06d}.bin"
        self._token_val_file_template = "fineweb_val_{idx:06d}.bin"

        # Do we have all of the downloaded files?
        train_pattern = os.path.join(self._token_file_name_root, "fineweb_train_*.bin")
        val_pattern = os.path.join(self._token_file_name_root, "fineweb_val_*.bin")
        train_files = sorted(glob.glob(train_pattern))
        val_files = sorted(glob.glob(val_pattern))

        if (
            len(train_files) != self._num_train_files
            or len(val_files) != self._num_val_files
        ):
            if download:
                os.makedirs(self._token_file_name_root, exist_ok=True)
                assert tokenizer.name() == "tiktoken_gpt2"

                def get(fname):
                    if not os.path.exists(
                        os.path.join(self._token_file_name_root, fname)
                    ):
                        hf_hub_download(
                            repo_id="kjj0/fineweb10B-gpt2",
                            filename=fname,
                            repo_type="dataset",
                            local_dir=self._token_file_name_root,
                        )

                # Get the validation file
                get("fineweb_val_%06d.bin" % 0)

                for i in range(1, self._num_train_files + 1):
                    get("fineweb_train_%06d.bin" % i)
            else:
                raise NotImplementedError()

        # Each shard has 100000000 tokens, except for the last shard which has
        # 55324043 tokens.
        # val has 100000000 tokens
        num_tokens_last_shard = 55324043
        self._num_tokens_per_shard = 100000000
        total_val_tokens = 100000000
        total_train_tokens = (
            self._num_tokens_per_shard * (self._num_train_files - 1)
            + num_tokens_last_shard
        )
        self._data_length = total_train_tokens if split == "train" else total_val_tokens
        self._split = split

    def __len__(self):
        # The last item we can get
        last_index = self._data_length - self._context_length - 1
        return last_index + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (self._data_length - self._context_length - 1):
            raise IndexError()

        if self._split == "train":
            # Find the shard that this idx belongs too
            train_shard_idx = (idx // self._num_tokens_per_shard) + 1
            train_idx = idx % self._num_tokens_per_shard

            # Does all of the data fit in the shard, or does it span two shards?
            if (train_idx + self._context_length + 1) < self._num_tokens_per_shard:
                # The data fully fits in the shard
                data = np.memmap(
                    os.path.join(
                        self._token_file_name_root,
                        self._token_train_file_template.format(idx=train_shard_idx),
                    ),
                    dtype=np.uint16,
                    mode="r",
                )
                x = torch.from_numpy(
                    data[train_idx : train_idx + self._context_length].astype(np.int64)
                )
                y = torch.from_numpy(
                    data[train_idx + 1 : train_idx + 1 + self._context_length].astype(
                        np.int64
                    )
                )
            else:
                # The data spans two shards
                shard1_data = np.memmap(
                    os.path.join(
                        self._token_file_name_root,
                        self._token_train_file_template.format(idx=train_shard_idx),
                    ),
                    dtype=np.uint16,
                    mode="r",
                )
                shard2_data = np.memmap(
                    os.path.join(
                        self._token_file_name_root,
                        self._token_train_file_template.format(idx=train_shard_idx + 1),
                    ),
                    dtype=np.uint16,
                    mode="r",
                )

                shard1_idx_x = train_idx
                shard1_idx_length = self._num_tokens_per_shard - train_idx
                shard2_idx_x = 0
                shard2_idx_length = self._context_length - shard1_idx_length

                x1 = torch.from_numpy(shard1_data[shard1_idx_x:].astype(np.int64))
                x2 = torch.from_numpy(
                    shard2_data[shard2_idx_x : shard2_idx_x + shard2_idx_length].astype(
                        np.int64
                    )
                )
                x = torch.cat([x1, x2], dim=0)

                shard1_idx_y = train_idx + 1
                if shard1_idx_y >= self._num_tokens_per_shard:
                    # The targets should be fully in shard 2
                    y = torch.from_numpy(
                        shard2_data[0 : self._context_length].astype(np.int64)
                    )
                else:
                    # The targets span both shards
                    shard1_idx_length = self._num_tokens_per_shard - shard1_idx_y
                    shard2_idx_y = 0
                    shard2_idx_length = self._context_length - shard1_idx_length

                    y1 = torch.from_numpy(shard1_data[shard1_idx_y:].astype(np.int64))
                    y2 = torch.from_numpy(
                        shard2_data[
                            shard2_idx_y : shard2_idx_y + shard2_idx_length
                        ].astype(np.int64)
                    )
                    y = torch.cat([y1, y2], dim=0)
        else:
            # If we are val, there is only one shard
            data = np.memmap(
                os.path.join(
                    self._token_file_name_root,
                    self._token_val_file_template.format(idx=0),
                ),
                dtype=np.uint16,
                mode="r",
            )
            x = torch.from_numpy(
                data[idx : idx + self._context_length].astype(np.int64)
            )
            y = torch.from_numpy(
                data[idx + 1 : idx + 1 + self._context_length].astype(np.int64)
            )
        assert x.shape[0] == self._context_length, f"{x.shape} {idx}"
        assert y.shape[0] == self._context_length, f"{y.shape} {idx}"
        return x, y
