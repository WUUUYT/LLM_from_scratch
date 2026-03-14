import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: bool,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_qkv = Linear(d_model, 3 * d_model)
        self.W_o = Linear(d_model, d_model)

        if rope:
            self.rope = RotaryPositionEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len
            )
        else:
            self.rope = None

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        *batch_dims, seq_len, _ = x.shape

        Q, K, V = (
            self.W_qkv(x)
            .view(*batch_dims, seq_len, 3, self.num_heads, self.d_k)
            .permute(-3, *range(len(batch_dims)), -2, -4, -1)
            .unbind(dim=0)
        )
        # Each: (*batch_dims, num_heads, seq_len, d_k)
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )

        out = scaled_dot_product_attention(Q, K, V, ~causal_mask)
        out = (
            out.transpose(-2, -3).contiguous().view(*batch_dims, seq_len, self.d_model)
        )

        return self.W_o(out)
