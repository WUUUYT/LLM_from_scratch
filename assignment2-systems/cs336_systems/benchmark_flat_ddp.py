"""
benchmark_flat_ddp.py — Compare naive (per-parameter) vs flat (batched) all-reduce DDP.

Both implementations are benchmarked under the same conditions:
  - 1 node x 2 GPUs
  - XL model (d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
  - batch_size=4, context_length=128

Usage:
    uv run python benchmark_flat_ddp.py
    uv run python benchmark_flat_ddp.py --markdown
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
from ddp_overlap import DDP
from ddp_overlap_bucketed import DDPBucketed

# ── Config ────────────────────────────────────────────────────────────────────
XL_CONFIG = dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 128
WARMUP_STEPS = 5
TIMED_STEPS = 10


# ── Shared helpers ────────────────────────────────────────────────────────────
def broadcast_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


# ── Strategy 1: Naive — one all_reduce per parameter ─────────────────────────
def allreduce_individual(model: nn.Module) -> float:
    """All-reduce each parameter gradient individually. Returns comm time (s)."""
    world_size = dist.get_world_size()
    t0 = time.perf_counter()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ── Strategy 2: Flat — one all_reduce for all gradients concatenated ──────────
def allreduce_flat(model: nn.Module) -> float:
    """
    Flatten all gradients into a single tensor, all-reduce once, then
    unflatten back into each parameter's .grad.
    Returns comm time (s).
    """
    world_size = dist.get_world_size()

    # Collect all gradients that need to be communicated
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    params_with_grad = [p for p in model.parameters() if p.grad is not None]

    t0 = time.perf_counter()

    # Flatten: concatenate all grad tensors into one 1-D tensor
    flat = torch._utils._flatten_dense_tensors(grads)

    # Single all_reduce on the flat tensor
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size

    # Unflatten: write results back into each parameter's .grad
    for param, new_grad in zip(
        params_with_grad,
        torch._utils._unflatten_dense_tensors(flat, grads),
    ):
        param.grad.copy_(new_grad)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ── Worker ────────────────────────────────────────────────────────────────────
def benchmark_worker(rank, world_size, strategy, bucket_size_mb, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

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

    local_bs = BATCH_SIZE // world_size
    batch = torch.randint(0, VOCAB_SIZE, (local_bs, CONTEXT_LENGTH), device=device)

    # ── Strategy: overlap uses DDP class, others use plain model ─────────────
    if strategy == "overlap":
        # DDP.__init__ handles broadcast internally
        ddp_model = DDP(model)
        optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

        def run_step():
            optimizer.zero_grad()
            ddp_model(batch).mean().backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

    elif strategy == "bucketed":
        ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
        optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

        def run_step():
            optimizer.zero_grad()
            ddp_model(batch).mean().backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

    else:
        broadcast_parameters(model)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        allreduce_fn = allreduce_individual if strategy == "individual" else allreduce_flat

        def run_step():
            optimizer.zero_grad()
            model(batch).mean().backward()
            allreduce_fn(model)
            optimizer.step()

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for _ in range(WARMUP_STEPS):
        run_step()
        torch.cuda.synchronize()

    # ── Timed steps ───────────────────────────────────────────────────────────
    total_times = []
    comm_times = []

    for _ in range(TIMED_STEPS):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        if strategy in ("overlap", "bucketed"):
            optimizer.zero_grad()
            ddp_model(batch).mean().backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()
            comm_times.append(None)
        else:
            optimizer.zero_grad()
            model(batch).mean().backward()
            comm_time = allreduce_fn(model)
            optimizer.step()
            comm_times.append(comm_time)

        torch.cuda.synchronize()
        total_times.append(time.perf_counter() - t_start)

    # ── Gather and aggregate across ranks ─────────────────────────────────────
    all_results = [None] * world_size
    dist.all_gather_object(all_results, {"total": total_times, "comm": comm_times})

    if rank == 0:
        all_total = sum([r["total"] for r in all_results], [])
        mean_total = sum(all_total) / len(all_total) * 1000

        all_comm = [c for r in all_results for c in r["comm"] if c is not None]
        mean_comm = sum(all_comm) / len(all_comm) * 1000 if all_comm else None
        comm_pct = round(100 * mean_comm / mean_total, 2) if mean_comm else "N/A"

        result_queue.put(
            {
                "strategy": strategy,
                "bucket_size_mb": bucket_size_mb if strategy == "bucketed" else "N/A",
                "mean_total_ms": round(mean_total, 2),
                "mean_comm_ms": round(mean_comm, 2) if mean_comm else "N/A",
                "comm_pct": comm_pct,
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
    bucket_sizes = [1, 10, 100, 1000]
    rows = []

    for strategy in ["individual", "flat", "overlap"]:
        print(f"Running strategy={strategy} ...", end="", flush=True)
        q = mp.Queue()
        mp.spawn(
            fn=benchmark_worker,
            args=(world_size, strategy, None, q),
            nprocs=world_size,
            join=True,
        )
        result = q.get()
        rows.append(result)
        print(f"  total={result['mean_total_ms']}ms  comm={result['mean_comm_ms']}ms  ({result['comm_pct']}%)")

    # Bucketed strategies with varying bucket sizes
    for bucket_mb in bucket_sizes:
        strategy = "bucketed"
        print(f"Running strategy=bucketed  bucket={bucket_mb}MB ...", end="", flush=True)
        q = mp.Queue()
        mp.spawn(
            fn=benchmark_worker,
            args=(world_size, strategy, bucket_mb, q),
            nprocs=world_size,
            join=True,
        )
        result = q.get()
        rows.append(result)
        print(f"  total={result['mean_total_ms']}ms")

    if args.markdown:
        import pandas as pd

        df = pd.DataFrame(rows)
        print("\n## DDP Strategy Comparison — XL Model (1 node × 2 GPUs)\n")
        print(df.to_markdown(index=False))
    else:
        print(f"\n{'strategy':>12} {'total_ms':>10} {'comm_ms':>10} {'comm_%':>8}")
        print("-" * 45)
        for r in rows:
            print(f"{r['strategy']:>12} {r['mean_total_ms']:>10} {str(r['mean_comm_ms']):>10} {str(r['comm_pct']):>8}")
