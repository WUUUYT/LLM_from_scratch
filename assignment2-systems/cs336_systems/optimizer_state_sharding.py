"""
sharded_optimizer.py — Optimizer state sharding across ranks.

Each rank owns and updates only 1/world_size of the parameters.
After optimizer.step(), each rank broadcasts its updated shard to all others.

Usage:
    optimizer = ShardedOptimizer(
        model.parameters(),
        optimizer_cls=torch.optim.AdamW,
        lr=1e-4,
        weight_decay=0.1,
    )

    for _ in range(train_steps):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Wraps any PyTorch optimizer with optimizer state sharding.

    - Parameters are partitioned across ranks (round-robin by default).
    - Each rank only maintains optimizer state for its own shard.
    - After step(), each rank broadcasts its updated params to all other ranks.
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ):
        # ── Standard Optimizer init (calls add_param_group for each group) ───
        # We must call super().__init__ with ALL params so that Optimizer
        # bookkeeping (param_groups, state) is set up correctly.
        # add_param_group will partition each group across ranks.
        self._optimizer_cls = optimizer_cls
        self._kwargs = kwargs
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        # _all_params stores every parameter (for broadcast after step)
        self._all_params: list[torch.Tensor] = []

        # The inner optimizer only gets this rank's shard of parameters
        self._inner_optimizer: Optimizer | None = None

        # Call super().__init__ — this triggers add_param_group for each group
        # We pass a dummy empty list first; add_param_group will handle the real params
        defaults = dict(**kwargs)
        super().__init__(params, defaults)

    # ── Parameter partitioning ────────────────────────────────────────────────

    def _assign_shard(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Given a list of parameters, return the subset owned by this rank.
        We use round-robin assignment: rank r owns params[r], params[r+W], params[r+2W], ...
        """
        return [p for i, p in enumerate(params) if i % self._world_size == self._rank]

    # ── Optimizer interface ───────────────────────────────────────────────────

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Called by super().__init__ for each parameter group, and may be called
        later to add new groups (e.g., for layer unfreezing).

        Partitions the group's parameters across ranks, builds the inner
        optimizer if it doesn't exist yet, or adds the shard to it.
        """
        # Normalize params to a list of tensors
        params = list(param_group.get("params", []))

        # Track all params (needed for broadcast after step)
        self._all_params.extend(params)

        # Determine this rank's shard
        shard_params = self._assign_shard(params)

        # Build a new group dict with only shard params
        shard_group = {k: v for k, v in param_group.items() if k != "params"}
        shard_group["params"] = shard_params

        # Add to inner optimizer (create it on first call)
        if self._inner_optimizer is None:
            self._inner_optimizer = self._optimizer_cls(
                [shard_group] if shard_params else [{"params": []}],
                **self._kwargs,
            )
        else:
            if shard_params:
                self._inner_optimizer.add_param_group(shard_group)

        # Also call super to keep Optimizer.param_groups in sync
        # (required for zero_grad() and other base-class methods to work)
        super().add_param_group(param_group)

    def step(self, closure: Callable = None, **kwargs: Any) -> Any:
        """
        1. Run the inner optimizer step on this rank's shard.
        2. Broadcast each parameter from its owner rank to all other ranks.
        """
        loss = None
        if self._inner_optimizer is not None:
            loss = self._inner_optimizer.step(closure, **kwargs)

        # Broadcast updated params: param i is owned by rank (i % world_size)
        for i, param in enumerate(self._all_params):
            src_rank = i % self._world_size
            dist.broadcast(param.data, src=src_rank)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients via the inner optimizer (only covers shard params)."""
        if self._inner_optimizer is not None:
            self._inner_optimizer.zero_grad(set_to_none=set_to_none)
        # Also zero gradients not in our shard (they were computed but won't
        # be updated by us; zeroing prevents stale grads in next backward)
        for i, param in enumerate(self._all_params):
            if i % self._world_size != self._rank:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

    @property
    def state(self):
        """Expose inner optimizer state for compatibility."""
        if self._inner_optimizer is not None:
            return self._inner_optimizer.state
        return {}

    def state_dict(self):
        if self._inner_optimizer is not None:
            return self._inner_optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._inner_optimizer is not None:
            self._inner_optimizer.load_state_dict(state_dict)
