"""
profile_sharded_optimizer.py — Profile memory and timing with/without optimizer state sharding.

Measures peak memory at three checkpoints:
  1. After model initialization
  2. Directly before optimizer step
  3. Directly after optimizer step

Also measures time per training iteration.

Setup: 1 node, 2 GPUs, XL model (d_model=1600, d_ff=6400, num_layers=48, num_heads=25)

Usage:
    uv run python profile_sharded_optimizer.py
    uv run python profile_sharded_optimizer.py --markdown
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
from sharded_optimizer import ShardedOptimizer
from ddp import DDP

# ── Config ────────────────────────────────────────────────────────────────────
XL_CONFIG = dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
VOCAB_SIZE     = 10_000
BATCH_SIZE     = 4
CONTEXT_LENGTH = 128
WARMUP_STEPS   = 3
TIMED_STEPS    = 10


def mem_mb() -> float:
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def peak_mem_mb() -> float:
    """Peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def sync():
    torch.cuda.synchronize()


# ── Worker ────────────────────────────────────────────────────────────────────
def profile_worker(rank, world_size, use_sharding, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    reset_peak()

    # ── Model init ────────────────────────────────────────────────────────────
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

    ddp_model = DDP(model)   # broadcasts rank 0 params, registers hooks
    ddp_model.train()
    sync()

    mem_after_init = mem_mb()
    peak_after_init = peak_mem_mb()
    reset_peak()

    # ── Optimizer init ────────────────────────────────────────────────────────
    if use_sharding:
        optimizer = ShardedOptimizer(
            ddp_model.parameters(),
            optimizer_cls=AdamW,
            lr=1e-4,
            weight_decay=0.1,
        )
    else:
        optimizer = AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=0.1)

    local_bs = BATCH_SIZE // world_size
    batch = torch.randint(0, VOCAB_SIZE, (local_bs, CONTEXT_LENGTH), device=device)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        ddp_model(batch).mean().backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        sync()

    reset_peak()

    # ── Memory profiling step ─────────────────────────────────────────────────
    optimizer.zero_grad()
    ddp_model(batch).mean().backward()
    ddp_model.finish_gradient_synchronization()
    sync()

    mem_before_step  = mem_mb()
    peak_before_step = peak_mem_mb()
    reset_peak()

    optimizer.step()
    sync()

    mem_after_step  = mem_mb()
    peak_after_step = peak_mem_mb()
    reset_peak()

    # ── Timing ────────────────────────────────────────────────────────────────
    step_times = []
    for _ in range(TIMED_STEPS):
        optimizer.zero_grad()
        sync()
        t0 = time.perf_counter()

        ddp_model(batch).mean().backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

        sync()
        step_times.append((time.perf_counter() - t0) * 1000)

    mean_step_ms = sum(step_times) / len(step_times)

    # ── Gather from all ranks ─────────────────────────────────────────────────
    all_results = [None] * world_size
    dist.all_gather_object(all_results, {
        "rank":              rank,
        "mem_after_init":    mem_after_init,
        "peak_after_init":   peak_after_init,
        "mem_before_step":   mem_before_step,
        "peak_before_step":  peak_before_step,
        "mem_after_step":    mem_after_step,
        "peak_after_step":   peak_after_step,
        "mean_step_ms":      mean_step_ms,
    })

    if rank == 0:
        # Average across ranks
        def avg(key):
            return round(sum(r[key] for r in all_results) / world_size, 1)

        result_queue.put({
            "use_sharding":      use_sharding,
            "mem_after_init":    avg("mem_after_init"),
            "peak_after_init":   avg("peak_after_init"),
            "mem_before_step":   avg("mem_before_step"),
            "peak_before_step":  avg("peak_before_step"),
            "mem_after_step":    avg("mem_after_step"),
            "peak_after_step":   avg("peak_after_step"),
            "mean_step_ms":      round(avg("mean_step_ms"), 2),
        })

    dist.destroy_process_group()


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Requires at least 2 CUDA GPUs.")
        return

    world_size = 2
    rows = []

    for use_sharding in [False, True]:
        label = "sharded" if use_sharding else "standard"
        print(f"Profiling {label} optimizer ...", end="", flush=True)
        q = mp.Queue()
        mp.spawn(
            fn=profile_worker,
            args=(world_size, use_sharding, q),
            nprocs=world_size,
            join=True,
        )
        result = q.get()
        rows.append(result)
        print(f"  step={result['mean_step_ms']}ms  "
              f"peak_before_step={result['peak_before_step']}MB  "
              f"peak_after_step={result['peak_after_step']}MB")

    # ── Print results ─────────────────────────────────────────────────────────
    labels = {False: "standard", True: "sharded"}

    if args.markdown:
        import pandas as pd
        df = pd.DataFrame([{
            "optimizer":        labels[r["use_sharding"]],
            "mem_after_init_MB":    r["mem_after_init"],
            "peak_before_step_MB":  r["peak_before_step"],
            "peak_after_step_MB":   r["peak_after_step"],
            "mean_step_ms":         r["mean_step_ms"],
        } for r in rows])
        print("\n## Optimizer State Sharding — Memory & Timing\n")
        print(df.to_markdown(index=False))
    else:
        print(f"\n{'optimizer':>10} {'init_MB':>10} {'pre_step_MB':>12} "
              f"{'post_step_MB':>13} {'step_ms':>9}")
        print("-" * 60)
        for r in rows:
            print(f"{labels[r['use_sharding']]:>10} "
                  f"{r['mem_after_init']:>10} "
                  f"{r['peak_before_step']:>12} "
                  f"{r['peak_after_step']:>13} "
                  f"{r['mean_step_ms']:>9}")

    # ── Memory breakdown estimate ─────────────────────────────────────────────
    total_params = (
        2 * XL_CONFIG["d_model"] * XL_CONFIG["d_ff"] * XL_CONFIG["num_layers"]
        + XL_CONFIG["d_model"] * VOCAB_SIZE * 2   # embedding + lm_head
    )
    param_mb   = total_params * 4 / 1024 / 1024   # FP32
    adam_mb    = total_params * 8 / 1024 / 1024   # m + v in FP32
    grad_mb    = total_params * 4 / 1024 / 1024   # FP32 grads

    print(f"\n## Theoretical Memory Breakdown (XL model, FP32)")
    print(f"  Parameters:        {param_mb:.1f} MB")
    print(f"  Gradients:         {grad_mb:.1f} MB")
    print(f"  Adam states (m+v): {adam_mb:.1f} MB  (standard)")
    print(f"  Adam states (m+v): {adam_mb/world_size:.1f} MB  (sharded, per rank)")
    print(f"  Expected saving:   {adam_mb - adam_mb/world_size:.1f} MB per rank")


if __name__ == "__main__":
    main()
