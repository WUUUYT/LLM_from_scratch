"""
ddp_overlap.py — Distributed Data Parallel wrapper with overlapped gradient communication.

Usage:
    model = MyModel().to(device)
    ddp_model = DDP(model)

    for _ in range(train_steps):
        logits = ddp_model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
"""

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that:
      1. Broadcasts rank 0's parameters to all ranks on init.
      2. Registers a post-accumulate-grad hook on every parameter so that
         each gradient is asynchronously all-reduced as soon as it is ready
         during the backward pass (overlapping communication with computation).
      3. Exposes finish_gradient_synchronization() to wait for all pending
         all-reduce handles before optimizer.step().
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles = []  # async all_reduce handles, cleared each step

        # ── Step 1: broadcast rank 0's parameters to all ranks ───────────────
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # ── Step 2: register hooks so each gradient is reduced as soon as ready
        world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(world_size))

    def _make_hook(self, world_size: int):
        """
        Returns a hook function that:
          - Launches an async all_reduce (SUM) on param.grad
          - Divides by world_size to get the mean
          - Stores the handle for later wait()
        """

        def hook(param: torch.Tensor) -> None:
            # Divide first so the all_reduce result is already the mean
            param.grad /= world_size
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return hook

    def forward(self, *inputs, **kwargs):
        """Delegate forward to the wrapped module."""
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all pending async all_reduce calls to be queued on the GPU.
        Must be called after loss.backward() and before optimizer.step().
        """
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
