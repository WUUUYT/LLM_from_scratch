from math import sqrt

import torch


def softmax(x: torch.Tensor, dim: int):
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x_safe = x - max_vals
    exp_x = torch.exp(x_safe)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = torch.einsum("...nd, ...md -> ...nm", Q, K) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = torch.einsum("...nm, ...md -> ...nd", attn_weights, V)
    return output
