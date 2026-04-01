"""
ddp_bucketed.py — DDP with bucketed gradient all-reduce and backward overlap.

Parameters are grouped into buckets of at most bucket_size_mb MB.
Buckets are assigned in reverse parameter order (matching backward order).
Each bucket is all-reduced asynchronously as soon as all its gradients are ready.

Usage:
    model = MyModel().to(device)
    ddp_model = DDPBucketed(model, bucket_size_mb=25.0)

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


class DDPBucketed(nn.Module):
    """
    Distributed Data Parallel with bucketed gradient communication.

    - Broadcasts rank 0 parameters on init.
    - Groups parameters into buckets of at most bucket_size_mb MB (in reverse order).
    - Registers a post-accumulate-grad hook on each parameter; when all params
      in a bucket are ready, flattens and async all-reduces that bucket.
    - finish_gradient_synchronization() waits for all handles, unflattens
      results back into .grad, and resets state for the next step.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # ── Broadcast rank 0 parameters to all ranks ──────────────────────────
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # ── Build buckets in reverse parameter order ──────────────────────────
        # Reverse order approximates the order gradients become ready in backward
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        params_reversed = list(reversed(list(self.module.parameters())))

        self._buckets: list[list[nn.Parameter]] = []
        current_bucket: list[nn.Parameter] = []
        current_bytes = 0

        for param in params_reversed:
            if not param.requires_grad:
                continue
            param_bytes = param.numel() * param.element_size()
            # Start a new bucket if adding this param would exceed the size limit
            # (always add at least one param per bucket)
            if current_bytes + param_bytes > bucket_size_bytes and current_bucket:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_bytes = 0
            current_bucket.append(param)
            current_bytes += param_bytes

        if current_bucket:
            self._buckets.append(current_bucket)

        n_buckets = len(self._buckets)

        # ── Per-bucket state (reset each step) ───────────────────────────────
        # pending[i]  : number of params in bucket i whose grad is not yet ready
        # handles[i]  : async all_reduce handle for bucket i (None until fired)
        # flat[i]     : the flattened gradient tensor sent to all_reduce
        self._pending: list[int] = [0] * n_buckets
        self._handles: list = [None] * n_buckets
        self._flat: list[torch.Tensor] = [None] * n_buckets

        # Map each parameter → its bucket index (for O(1) lookup in hooks)
        self._param_to_bucket: dict = {}
        for bucket_idx, bucket in enumerate(self._buckets):
            for param in bucket:
                self._param_to_bucket[param] = bucket_idx

        # ── Register post-accumulate-grad hooks ───────────────────────────────
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

        # Initialize pending counts
        self._reset_pending()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_pending(self):
        """Reset pending counters to the number of requires_grad params per bucket."""
        for i, bucket in enumerate(self._buckets):
            self._pending[i] = sum(1 for p in bucket if p.requires_grad)

    def _grad_hook(self, param: torch.Tensor) -> None:
        """
        Called by autograd immediately after param.grad is accumulated.
        Decrements the pending counter for this param's bucket.
        If the bucket is complete, flattens grads and launches async all_reduce.
        """
        bucket_idx = self._param_to_bucket[param]
        self._pending[bucket_idx] -= 1

        if self._pending[bucket_idx] == 0:
            self._launch_allreduce(bucket_idx)

    def _launch_allreduce(self, bucket_idx: int) -> None:
        """Flatten bucket grads, divide by world_size, async all_reduce."""
        bucket = self._buckets[bucket_idx]
        grads = [p.grad for p in bucket if p.grad is not None]

        if not grads:
            return

        # Flatten all grads in this bucket into one contiguous tensor
        flat = torch._utils._flatten_dense_tensors(grads)

        # Divide before SUM so result is the mean
        flat /= self.world_size

        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)

        self._handles[bucket_idx] = handle
        self._flat[bucket_idx] = flat

    # ── Public interface ──────────────────────────────────────────────────────

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all bucket all_reduces to be queued, then unflatten results
        back into each parameter's .grad. Resets state for the next step.
        """
        for bucket_idx, (handle, flat) in enumerate(zip(self._handles, self._flat)):
            if handle is None:
                continue

            # Wait for this bucket's all_reduce to be queued on GPU
            handle.wait()

            # Unflatten back into each parameter's .grad
            bucket = self._buckets[bucket_idx]
            grads = [p.grad for p in bucket if p.grad is not None]
            params_grad = [p for p in bucket if p.grad is not None]

            for param, new_grad in zip(
                params_grad,
                torch._utils._unflatten_dense_tensors(flat, grads),
            ):
                param.grad.copy_(new_grad)

        # Reset state for the next training step
        self._handles = [None] * len(self._buckets)
        self._flat = [None] * len(self._buckets)
        self._reset_pending()
