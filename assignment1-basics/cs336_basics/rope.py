import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.d_k = d_k

        pow_indices = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / theta ** (pow_indices / d_k)
        t = torch.arange(max_seq_len).float()

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        token_positions: index of token passed in.
        """
        token_positions = token_positions.to(x.device)
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)

        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, d_k//2)
        sin = sin.unsqueeze(1)  # (batch, 1, seq_len, d_k//2)

        x_paired = x.view(*x.shape[:-1], -1, 2)
        x1 = x_paired[..., 0]
        x2 = x_paired[..., 1]  # (..., d_k / 2)

        out1 = cos * x1 - sin * x2
        out2 = sin * x1 + cos * x2

        return torch.stack([out1, out2], dim=-1).flatten(-2)
