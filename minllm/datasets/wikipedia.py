import json
import os
import numpy as np
import tarfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from minllm.datasets.utils import download_file_from_google_drive


class Wikipedia2020English(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        assert split in ["train", "0"]

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        archive_base = "Wikipedia_2020.en"
        archive_file_name = os.path.join(
            root_dir, f"{archive_base}/Wikipedia_2020.en.tar.gz"
        )

        num_json_files = 10
        json_base_spec = f"{archive_base}/{archive_base}" + "/wikipedia-en-{idx}.json"
        archive_files = [json_base_spec.format(idx=i) for i in range(num_json_files)]

        def data_exists(list_of_files) -> bool:
            for path in list_of_files:
                if not os.path.isfile(os.path.join(root_dir, path)):
                    return False
            return True

        assert data_exists(archive_files)

        if not data_exists(archive_files):
            _ARCHIVE_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
            _ARCHIVE_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"

            if not os.path.isfile(archive_file_name):
                download(archive_file_name, _ARCHIVE_URL, _ARCHIVE_FILE_ID)

                # Extract the archive
                tar = tarfile.open(archive_file_name, "r:gz")
                tar.extractall()
                tar.close()
            assert data_exists(archive_files)

        # Load in the individual json
        self._data = []
        if split == "train":
            for json_path in tqdm(archive_files, "Loading data"):
                full_json_path = os.path.join(root_dir, json_path)
                with open(full_json_path) as fp:
                    json_data = json.load(fp)
                self._data.extend(json_data)
        else:
            full_json_path = os.path.join(root_dir, json_base_spec.format(idx=split))
            with open(full_json_path) as fp:
                json_data = json.load(fp)
            self._data.extend(json_data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._data[idx]


class Wikipedia2020EnglishTokenized(Dataset):
    def __init__(self, root_dir, context_length: int = 1024, tokenizer="gpt-2"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir

        assert tokenizer in ["gpt-2"]

        archive_base = "Wikipedia_2020.en"
        self._data_file_name = os.path.join(
            root_dir, f"{archive_base}/Wikipedia_2020.en.gpt2.bin"
        )
        self._context_length = context_length
        self._data_type = np.uint16

        if not os.path.isfile(self._data_file_name):
            # Create the tokenized dataset from the raw string dataset
            import tiktoken

            token_encoder = tiktoken.get_encoding("gpt2")
            string_dataset = Wikipedia2020English(root_dir=root_dir, split="0")
            all_tokens = [
                token
                for row in [
                    token_encoder.encode_ordinary(string_dataset[idx])
                    + [token_encoder.eot_token]
                    for idx in tqdm(
                        range(len(string_dataset)), desc="Tokenizing dataset"
                    )
                ]
                for token in row
            ]
            self._data_length = len(all_tokens)
            output_array = np.memmap(
                self._data_file_name,
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

        data = np.memmap(self._data_file_name, dtype=self._data_type, mode="r")
        self._data_length = data.size

    def __len__(self):
        # The last item we can get
        return self._data_length - self._context_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = np.memmap(self._data_file_name, dtype=self._data_type, mode="r")
        return torch.from_numpy(data[idx : idx + self._context_length].astype(np.int64))
