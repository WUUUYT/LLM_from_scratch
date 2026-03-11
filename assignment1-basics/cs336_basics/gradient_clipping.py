from math import sqrt

import torch


def gradient_clipping(params, M, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    grads_norm = sqrt(sum(grad.norm().item() ** 2 for grad in grads))
    if grads_norm > M:
        scale = M / (grads_norm + eps)
        for g in grads:
            g.mul_(scale)
