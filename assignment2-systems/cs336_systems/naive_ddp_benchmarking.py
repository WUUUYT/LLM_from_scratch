"""
benchmark_naive_ddp.py — Benchmark naive DDP training for the XL transformer model.

Measures:
  - Total time per training step
  - Time spent on gradient all-reduce communication
  - Proportion of time spent communicating

Setup: single-node, 2 GPUs, XL model (d_model=1600, d_ff=6400, num_layers=48, num_heads=25)

Usage:
    uv run python benchmark_naive_ddp.py
    uv run python benchmark_naive_ddp.py --markdown
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# ── XL model config (Table 1) ─────────────────────────────────────────────────
XL_CONFIG = dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
VOCAB_SIZE = 10_000
BATCH_SIZE = 4  # global batch size
CONTEXT_LENGTH = 128

WARMUP_STEPS = 5
TIMED_STEPS = 10


# ── Naive DDP helpers (from naive_ddp.py) ─────────────────────────────────────
def broadcast_parameters(model: nn.Module) -> None:
    """Broadcast rank 0's parameters to all ranks."""
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def allreduce_gradients(model: nn.Module) -> float:
    """
    All-reduce each parameter's gradient individually.
    Returns the time spent on communication in seconds.
    """
    world_size = dist.get_world_size()
    comm_start = time.perf_counter()

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    torch.cuda.synchronize()
    return time.perf_counter() - comm_start


# ── Worker ────────────────────────────────────────────────────────────────────
def benchmark_worker(rank, world_size, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ── Build model ───────────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=XL_CONFIG["d_model"],
        d_ff=XL_CONFIG["d_ff"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        rope_theta=10000,
    ).to(device)

    # Broadcast rank 0's parameters to all ranks
    broadcast_parameters(model)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Each rank gets local_bs = BATCH_SIZE // world_size examples
    local_bs = BATCH_SIZE // world_size
    batch = torch.randint(0, VOCAB_SIZE, (local_bs, CONTEXT_LENGTH), device=device)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        logits = model(batch)
        loss = logits.mean()
        loss.backward()
        allreduce_gradients(model)
        optimizer.step()
        torch.cuda.synchronize()

    # ── Timed steps ───────────────────────────────────────────────────────────
    total_times = []
    comm_times = []

    for _ in range(TIMED_STEPS):
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        optimizer.zero_grad()

        # Forward + backward
        logits = model(batch)
        loss = logits.mean()
        loss.backward()

        # All-reduce gradients (timed separately)
        comm_time = allreduce_gradients(model)

        optimizer.step()
        torch.cuda.synchronize()

        step_end = time.perf_counter()
        total_times.append(step_end - step_start)
        comm_times.append(comm_time)

    # ── Gather results from all ranks ─────────────────────────────────────────
    local_result = {
        "rank": rank,
        "total_times_ms": [t * 1000 for t in total_times],
        "comm_times_ms": [t * 1000 for t in comm_times],
    }
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_result)

    if rank == 0:
        # Average across ranks and steps
        all_total = []
        all_comm = []
        for r in all_results:
            all_total.extend(r["total_times_ms"])
            all_comm.extend(r["comm_times_ms"])

        mean_total = sum(all_total) / len(all_total)
        mean_comm = sum(all_comm) / len(all_comm)
        comm_pct = 100.0 * mean_comm / mean_total

        result_queue.put(
            {
                "world_size": world_size,
                "mean_total_ms": round(mean_total, 2),
                "mean_comm_ms": round(mean_comm, 2),
                "comm_pct": round(comm_pct, 2),
            }
        )

    dist.destroy_process_group()


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("This benchmark requires at least 2 CUDA GPUs.")
        return

    world_size = 2
    result_queue = mp.Queue()

    print(f"Benchmarking naive DDP — XL model, {world_size} GPUs")
    print(f"Batch size: {BATCH_SIZE} (local: {BATCH_SIZE // world_size} per rank)")
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Warmup: {WARMUP_STEPS}  |  Timed steps: {TIMED_STEPS}\n")

    mp.spawn(
        fn=benchmark_worker,
        args=(world_size, result_queue),
        nprocs=world_size,
        join=True,
    )

    result = result_queue.get()

    if args.markdown:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "world_size": result["world_size"],
                    "total_step_ms": result["mean_total_ms"],
                    "comm_ms": result["mean_comm_ms"],
                    "comm_%": result["comm_pct"],
                }
            ]
        )
        print("## Naive DDP Benchmark — XL Model (1 node × 2 GPUs)\n")
        print(df.to_markdown(index=False))
    else:
        print(f"World size:          {result['world_size']}")
        print(f"Total step time:     {result['mean_total_ms']} ms")
        print(f"Comm time (all-red): {result['mean_comm_ms']} ms")
        print(f"Comm fraction:       {result['comm_pct']} %")


if __name__ == "__main__":
    main()
