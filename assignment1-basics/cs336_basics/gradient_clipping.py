import torch


def gradient_clipping(params, M, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return

    grads_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in grads]))
    if grads_norm > M:
        scale = M / (grads_norm + eps)
        for g in grads:
            g.mul_(scale)
