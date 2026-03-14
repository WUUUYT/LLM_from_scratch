import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        dropout: float = 0.0,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, theta, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if weight_tying:
            self.lm_head.weight = self.embedding.weight
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        token_positions = (
            torch.arange(seq_len, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.norm(x)
        return self.lm_head(x)
