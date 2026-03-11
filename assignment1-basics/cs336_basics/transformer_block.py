import torch
import torch.nn as nn

from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        dropout=0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), token_positions)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x
