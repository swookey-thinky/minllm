import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_attention_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_attention_heads = config.num_attention_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (embedding_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)
        k = k.view(
            B, T, self.num_attention_heads, C // self.num_attention_heads
        ).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(
            B, T, self.num_attention_heads, C // self.num_attention_heads
        ).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(
            B, T, self.num_attention_heads, C // self.num_attention_heads
        ).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
