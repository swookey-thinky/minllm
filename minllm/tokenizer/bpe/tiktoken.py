"""Tiktoken based tokenizer."""

import os
import tiktoken
from typing import List

from minllm.tokenizer.bpe.utils import render_token


class TiktokenTokenizer:
    """Base class for Tokenizers"""

    def __init__(self, name: str = "gpt2"):
        super().__init__()

        self._encoding = tiktoken.get_encoding(name)
        self._encoding_name = name

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(
        self, text: str, with_special: bool = False, append_eot: bool = False
    ) -> List[int]:
        # encode_ordinary ignores any special tokens
        if with_special:
            ids = self._encoding.encode(text, allowed_special={"<|endoftext|>"})
        else:
            ids = self._encoding.encode_ordinary(text)

        if append_eot:
            ids.append(self._encoding.eot_token)
        return ids

    def decode(self, ids: List[int]):
        # Tokenizer can decode a list of integers into a string
        return self._encoding.decode(ids)

    def name(self) -> str:
        return f"tiktoken_{self._encoding_name}"

    def save(self, file_prefix: str, output_path: str):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        raise NotImplementedError()

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        raise NotImplementedError()
