import torch.nn as nn
from torch.nn import functional as F

from minllm.layers.attention import CausalSelfAttention
from minllm.layers.base import LayerNorm, MLP


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
