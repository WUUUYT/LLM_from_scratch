"""
naive_ddp.py — Naïve Distributed Data Parallel implementation.

Two functions are implemented:
  - get_ddp_individual_parameters(model):
      Broadcast rank 0's parameters to all ranks so every GPU starts identically.

  - ddp_individual_parameters_on_after_backward(model, optimizer):
      After backward(), all-reduce each parameter's gradient individually,
      so every GPU has the average gradient across all ranks.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def get_ddp_individual_parameters_(model: nn.Module) -> nn.Module:
    """
    Prepare a model for naïve DDP by broadcasting rank 0's parameters to all ranks.

    Args:
        model: a locally-initialized nn.Module (possibly with different weights per rank)

    Returns:
        the same model, with parameters overwritten to match rank 0
    """
    for param in model.parameters():
        # Broadcast each parameter tensor from rank 0 to all other ranks.
        # After this, every rank holds an identical copy of rank 0's parameters.
        dist.broadcast(param.data, src=0)

    return model


def ddp_individual_parameters_on_after_backward_(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Call this after loss.backward() and before optimizer.step().

    All-reduces each parameter's gradient individually across all ranks,
    replacing the local gradient with the average across all ranks.
    This is equivalent to computing gradients on the full (unsharded) batch.

    Args:
        model:     the DDP-wrapped model
        optimizer: unused here, but part of the interface for future extensions
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is None:
            # Skip parameters that did not receive a gradient
            # (e.g., frozen parameters or those not in the computation graph)
            continue

        # Sum gradients across all ranks, then divide by world_size to get the mean.
        # This is equivalent to computing the gradient on the full batch.
        with torch.no_grad():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
