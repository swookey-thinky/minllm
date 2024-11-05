"""
Full definition of a GPT-1 Language Model, all of it in this single file.
References:
1) the official GPT TensorFlow implementation released by OpenAI:
https://github.com/openai/finetune-transformer-lm/blob/master/train.py#L162
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from minllm.layers.attention import CausalSelfAttention
from minllm.layers.base import LayerNorm, MLP


class TransformerBlock(nn.Module):
    """Transformer block from GPT."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self._config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.embedding_dim),
                wpe=nn.Embedding(config.context_length, config.embedding_dim),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_layers)]
                ),
            )
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying, from https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # idx is the batched tensor of tokens, of shape (B, L)
        device = idx.device
        _, t = idx.size()
        assert (
            t <= self._config.context_length
        ), f"Cannot forward sequence of length {t}, block size is only {self._config.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Token embeddings of shape (b, t, embedding_dim)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (t, embedding_dim)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        # Note these weights are tied to the embedding weights above.
        logits = self.lm_head(x)

        # Calculate the loss in the model so that multi-gpu memory
        # usage is not gated back in the main process.
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, x, loss
        return logits, x

    def logits(self, h: torch.Tensor):
        return self.lm_head(h)
