"""Harness for LM Eval model.

A model wrapper to evaluate models using Eleuther AI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
"""

from lm_eval.api.model import LM
from lm_eval import utils
import torch
from tqdm import tqdm
from typing import List, Tuple

from minllm.tokenizer.bpe.base import Tokenizer
from minllm.utils import instantiate_from_config, DotConfig


class LMEvalLanguageModel(LM):
    def __init__(self, checkpoint_path: str, **kwargs) -> None:
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = DotConfig(checkpoint["config"])

        self._config = config
        self._model = instantiate_from_config(config.model, use_config_struct=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])

        if "tokenizer" in config:
            self._tokenizer: Tokenizer = instantiate_from_config(
                config.tokenizer.to_dict()
            )
        else:
            print("Tokenizer not found in config, using tiktoken GPT-2")
            from minllm.tokenizer.bpe.tiktoken import TiktokenTokenizer

            self._tokenizer: Tokenizer = TiktokenTokenizer(name="gpt2")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            if context == "":
                # BOS or EOS as context
                raise NotImplementedError()
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self._tokenizer.encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            input_tokens = context_enc + continuation_enc
            input_len = len(input_tokens)
            continuation_len = len(continuation_enc)
            context_len = len(context_enc)

            input_tokens = torch.tensor(
                input_tokens[-(self._config.model.params.context_length + 1) :][:-1],
                dtype=torch.long,
            )
            logits, _ = self._model(input_tokens[None, :])
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            # logits are shape (1, seq_len, vocab_size). Now select only
            # the context tokens
            logits = logits[:, -continuation_len:, :]
            assert logits.shape[1] == continuation_len

            greedy_tokens = logits.argmax(dim=-1)

            continuation_tokens = torch.tensor(
                (continuation_enc)[-self._config.model.params.context_length :],
                dtype=torch.long,
            )[None, :]
            max_equal = (greedy_tokens == continuation_tokens).all()

            logits = torch.gather(logits, 2, continuation_tokens.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq]
            answer = (float(logits.sum()), bool(max_equal))
            res.append(answer)
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()

        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            assert request.arguments[0].strip() != ""

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self._tokenizer.encode(context + continuation)
        context_enc = self._tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc
