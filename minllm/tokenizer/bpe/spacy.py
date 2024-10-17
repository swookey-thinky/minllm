"""Implementation of the spaCy tokenizer for en-us.

As used in the GPT-1 paper:
"Improving Language Understanding by Generative Pre-Training"
(https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

Based on the original implementation at:
https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py

In order to use, must manually download the en-core-web-sm dataset:

> python -m spacy download en_core_web_sm
"""

import re
import ftfy
import json
import spacy
from typing import List, Optional

from minllm.tokenizer.bpe.base import Tokenizer


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(
        """(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )
    text = re.sub("\s*\n\s*", " \n ", text)
    text = re.sub("[^\S\n]+", " ", text)
    return text.strip()


class _TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "ner", "textcat"]
        )
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path).read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def encode(self, texts: List[str]):
        texts_tokens = []
        for text in texts:
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                text_tokens.extend(
                    [
                        self.encoder.get(t, 0)
                        for t in self.bpe(token.text.lower()).split(" ")
                    ]
                )
            texts_tokens.append(text_tokens)
        return texts_tokens


class SpacyTextTokenizer(Tokenizer):
    def __init__(self, special_tokens: Optional[List[str]] = None):
        super().__init__()

        # Encoder implementation
        self._text_encoder = _TextEncoder(
            encoder_path="minllm/tokenizer/bpe/data/spacy/encoder_bpe_40000.json",
            bpe_path="minllm/tokenizer/bpe/data/spacy/vocab_40000.bpe",
        )

        self._special_tokens = []
        if special_tokens is not None:
            for special_token in special_tokens:
                self._text_encoder.encoder[special_token] = len(
                    self._text_encoder.encoder
                )
            self._special_tokens = special_tokens
        self.special_tokens = self._text_encoder.encoder

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        # Tokenizer can encode a string into a list of integers
        return self._text_encoder.encode([text])[0]

    def decode(self, ids: List[int]):
        # Tokenizer can decode a list of integers into a string
        text = "".join(self._text_encoder.decoder[idx] for idx in ids)
        return text

    def save(self, file_prefix: str, output_path: str):
        raise NotImplementedError

    def load(self, model_file):
        raise NotImplementedError
