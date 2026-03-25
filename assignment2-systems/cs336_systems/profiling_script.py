"""
End-to-End Benchmarking Script with NVTX Annotations for nsys profiling
Usage:
    # Normal benchmarking
    uv run python profiling_script.py --size small --forward_only

    # With nsys profiling
    uv run nsys profile -o result --pytorch python profiling_script.py --size small --forward_only --annotate

    # Full training step profiling
    uv run nsys profile -o result --pytorch python profiling_script.py --size small --annotate

    # Memory profiling (forward only)
    uv run python profiling_script.py --size 2.7B --context_length 128 --forward_only --memory_profile

    # Memory profiling (full training step)
    uv run python profiling_script.py --size 2.7B --context_length 128 --memory_profile

    # Compile model
    uv run python profiling_script.py --size small --compile --markdown
"""

import argparse
import timeit
from contextlib import nullcontext

import torch

# NVTX is only available on CUDA builds — fall back to no-ops on MPS/CPU
try:
    import torch.cuda.nvtx as nvtx

    nvtx.range_push("test")  # probe: will raise if not supported
    nvtx.range_pop()
except Exception:
    from contextlib import contextmanager

    class _NoOpNVTX:
        """Drop-in replacement for torch.cuda.nvtx on non-CUDA platforms."""

        @staticmethod
        @contextmanager
        def range(msg, *args, **kwargs):
            yield

        @staticmethod
        def range_push(msg):
            pass

        @staticmethod
        def range_pop():
            pass

    nvtx = _NoOpNVTX()
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# ── Model configurations (Table 1) ──────────────────────────────────────────
MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer benchmarking script")

    # Model
    parser.add_argument("--size", choices=MODEL_CONFIGS.keys(), default="small", help="Model size preset (Table 1)")
    parser.add_argument("--context_length", type=int, default=128, help="Sequence length")

    # Override individual model dims (optional)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)

    # Benchmark settings
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up steps (not timed)")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed steps")
    parser.add_argument("--forward_only", action="store_true", help="Only time forward pass (skip backward)")
    parser.add_argument("--markdown", action="store_true", help="Print results as a Markdown table at the end")
    parser.add_argument("--annotate", action="store_true", help="Enable NVTX annotations (for nsys profiling)")
    parser.add_argument("--mixed_precision", action="store_true", help="Use BF16 mixed precision via torch.autocast")
    parser.add_argument(
        "--memory_profile",
        action="store_true",
        help="Record memory history and dump a snapshot pickle for pytorch.org/memory_viz",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model with torch.compile before benchmarking"
    )

    return parser.parse_args()


def build_model(args, device):
    cfg = dict(MODEL_CONFIGS[args.size])  # copy preset
    # Allow per-field overrides
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        rope_theta=10000,
    ).to(device)

    return model, cfg


def run_step(model, batch, forward_only, optimizer=None, autocast_ctx=None):
    """One forward (+ optional backward + optimizer) step with NVTX annotations."""
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    with nvtx.range("forward"):
        with autocast_ctx:
            logits = model(batch)

    if not forward_only:
        with nvtx.range("loss"):
            loss = logits.mean()

        with nvtx.range("backward"):
            loss.backward()

        if optimizer is not None:
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()


def sync(device):
    """Synchronize device (CUDA or MPS)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device : {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model, cfg = build_model(args, device)
    model.eval() if args.forward_only else model.train()

    # Build optimizer for full training step
    optimizer = None if args.forward_only else AdamW(model.parameters(), lr=1e-4)

    # ── Optionally compile the model ─────────────────────────────────────────
    if args.compile:
        print("torch.compile: ENABLED — compiling model...")
        model = torch.compile(model)
        print("torch.compile: done")

    # ── Mixed precision context ───────────────────────────────────────────────
    if args.mixed_precision and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("Mixed precision: BF16 autocast ENABLED")
    else:
        autocast_ctx = nullcontext()
        if args.mixed_precision:
            print("WARNING: mixed precision requested but device is not CUDA — running FP32.")
        else:
            print("Mixed precision: disabled (FP32)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Size   : {args.size}  |  Params: {total_params / 1e6:.1f}M")
    print(f"Config : {cfg}")
    print(f"Context: {args.context_length}  |  Batch: {BATCH_SIZE}")
    print(f"Mode   : {'forward only' if args.forward_only else 'forward + backward + optimizer'}")
    print(f"Warmup : {args.warmup}  |  Steps: {args.steps}")
    print("-" * 60)

    # ── Random batch ─────────────────────────────────────────────────────────
    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, args.context_length), device=device)

    # ── Warm-up (not timed, memory profiler not yet started) ─────────────────
    print("Running warm-up steps...")
    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            with torch.set_grad_enabled(not args.forward_only):
                run_step(model, batch, args.forward_only, optimizer, autocast_ctx)
            sync(device)

    # ── Start memory profiler AFTER warm-up ──────────────────────────────────
    if args.memory_profile:
        if device.type != "cuda":
            print("WARNING: memory profiling requires CUDA — skipping.")
            args.memory_profile = False
        else:
            print("Memory profiler: STARTED")
            torch.cuda.memory.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # ── Timed steps ──────────────────────────────────────────────────────────
    print("Timing steps...")
    times = []
    with nvtx.range("timed_steps"):
        for i in range(args.steps):
            sync(device)
            t_start = timeit.default_timer()

            with nvtx.range(f"step_{i}"):
                with torch.set_grad_enabled(not args.forward_only):
                    run_step(model, batch, args.forward_only, optimizer, autocast_ctx)

            sync(device)
            t_end = timeit.default_timer()
            times.append(t_end - t_start)

    # ── Dump memory snapshot BEFORE stopping (captures everything) ────────────
    if args.memory_profile:
        mode = "fwd" if args.forward_only else "full"
        mp = "bf16" if args.mixed_precision else "fp32"
        snapshot_path = f"memory_{args.size}_ctx{args.context_length}_{mode}_{mp}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory profiler: snapshot saved to '{snapshot_path}'")
        print("  → Drag and drop this file onto https://pytorch.org/memory_viz")

    # ── Results ──────────────────────────────────────────────────────────────
    import statistics

    mean_ms = statistics.mean(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000

    print(f"\n{'Metric':<12} {'Value':>10}")
    print("-" * 24)
    print(f"{'Mean':<12} {mean_ms:>9.2f} ms")
    print(f"{'Std Dev':<12} {stdev_ms:>9.2f} ms")
    print(f"{'Min':<12} {min_ms:>9.2f} ms")
    print(f"{'Max':<12} {max_ms:>9.2f} ms")

    # Memory (CUDA only)
    if device.type == "cuda":
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n{'Mem Alloc':<12} {alloc_gb:>9.3f} GB")
        print(f"{'Mem Peak':<12} {peak_gb:>9.3f} GB")

    results = {
        "size": args.size,
        "params (M)": total_params / 1e6,
        "forward_only": args.forward_only,
        "mixed_precision": args.mixed_precision,
        "compiled": args.compile,
        "mean_ms": round(mean_ms, 2),
        "stdev_ms": round(stdev_ms, 2),
    }
    if device.type == "cuda":
        results["peak_mem_GB"] = round(peak_gb, 3)

    return results


def results_to_markdown(results: dict) -> str:
    """Convert a single result dict to a one-row Markdown table."""
    import pandas as pd

    return pd.DataFrame([results]).to_markdown(index=False)


if __name__ == "__main__":
    args = parse_args()
    results = benchmark(args)
    if args.markdown:
        print("\n## Benchmark Results (Markdown)\n")
        print(results_to_markdown(results))
