"""
distributed_single_benchmark.py — Benchmark all-reduce latency across backends, data sizes, and process counts.

Usage:
    # Gloo + CPU
    uv run python benchmark_allreduce.py --backend gloo

    # NCCL + GPU
    uv run python benchmark_allreduce.py --backend nccl

    # Both, output markdown table
    uv run python benchmark_allreduce.py --backend gloo --markdown
    uv run python benchmark_allreduce.py --backend nccl --markdown
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ── Configuration ─────────────────────────────────────────────────────────────
WARMUP_STEPS = 5
TIMED_STEPS = 20

# Data sizes in bytes: 1MB, 10MB, 100MB, 1GB
DATA_SIZES = {
    "1MB": 1 * 1024 * 1024,
    "10MB": 10 * 1024 * 1024,
    "100MB": 100 * 1024 * 1024,
    "1GB": 1024 * 1024 * 1024,
}

WORLD_SIZES = [2, 4, 6]


# ── Worker function ───────────────────────────────────────────────────────────
def benchmark_worker(rank, world_size, backend, data_sizes, result_queue):
    """
    Runs inside each child process.
    Benchmarks all-reduce for each data size and sends results to the main process.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)

    local_results = {}

    for size_label, n_bytes in data_sizes.items():
        # Number of float32 elements for this byte budget
        n_elements = n_bytes // 4
        tensor = torch.ones(n_elements, dtype=torch.float32, device=device)

        # ── Warm-up ───────────────────────────────────────────────────────────
        for _ in range(WARMUP_STEPS):
            dist.all_reduce(tensor, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()

        # ── Timed steps ───────────────────────────────────────────────────────
        times = []
        for _ in range(TIMED_STEPS):
            if backend == "nccl":
                torch.cuda.synchronize()  # ensure GPU idle before start

            t_start = time.perf_counter()
            dist.all_reduce(tensor, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()  # wait for GPU to truly finish

            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000)  # ms

        local_results[size_label] = times

    # ── Gather results from all ranks to rank 0 ───────────────────────────────
    # Use all_gather_object to collect timings from every rank
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)

    if rank == 0:
        # Average timings across ranks for each size
        aggregated = {}
        for size_label in data_sizes:
            all_times = []
            for rank_result in all_results:
                all_times.extend(rank_result[size_label])
            mean_ms = sum(all_times) / len(all_times)
            aggregated[size_label] = round(mean_ms, 3)

        result_queue.put(
            {
                "world_size": world_size,
                "backend": backend,
                "results": aggregated,
            }
        )

    dist.destroy_process_group()


# ── Runner ────────────────────────────────────────────────────────────────────
def run_benchmark(world_size, backend, data_sizes):
    """Spawn workers and collect results."""
    result_queue = mp.Queue()
    mp.spawn(
        fn=benchmark_worker,
        args=(world_size, backend, data_sizes, result_queue),
        nprocs=world_size,
        join=True,
    )
    return result_queue.get()


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        print("NCCL requires CUDA — not available on this machine.")
        return

    n_gpus = torch.cuda.device_count() if args.backend == "nccl" else 99
    world_sizes = [ws for ws in WORLD_SIZES if ws <= n_gpus]

    print(f"Backend: {args.backend.upper()}")
    print(f"World sizes: {world_sizes}\n")

    rows = []
    for world_size in world_sizes:
        print(f"  world_size={world_size} ...", end="", flush=True)
        result = run_benchmark(world_size, args.backend, DATA_SIZES)
        row = {"backend": args.backend, "world_size": world_size}
        row.update(result["results"])
        rows.append(row)
        print(f"  done: {result['results']}")

    # ── Print table ───────────────────────────────────────────────────────────
    if args.markdown:
        import pandas as pd

        df = pd.DataFrame(rows)
        print(f"\n## All-Reduce Benchmark ({args.backend.upper()}) — latency in ms\n")
        print(df.to_markdown(index=False))
    else:
        header = f"{'backend':>6} {'procs':>6} " + " ".join(f"{k:>8}" for k in DATA_SIZES)
        print("\n" + header)
        print("-" * len(header))
        for row in rows:
            vals = " ".join(f"{row[k]:>8}" for k in DATA_SIZES)
            print(f"{row['backend']:>6} {row['world_size']:>6} {vals}")


if __name__ == "__main__":
    main()
