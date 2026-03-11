import math

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff else int(((d_model * 8 / 3 + 63) // 64) * 64)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.W1 = nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty((self.d_model, self.d_ff), **factory_kwargs))
        self.W3 = nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.d_model + self.d_ff))
        for p in [self.W1, self.W2, self.W3]:
            nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_linear = torch.einsum("...m,fm->...f", x, self.W1)
        swish_gate = gate_linear * torch.sigmoid(gate_linear)

        linear = torch.einsum("...m,fm->...f", x, self.W3)

        intermediate = linear * swish_gate

        return torch.einsum("... f, m f -> ... m", intermediate, self.W2)
