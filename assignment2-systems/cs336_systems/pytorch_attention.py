"""
benchmark_attention.py — Benchmark naive and compiled attention at different scales.

Usage:
    uv run python benchmark_attention.py                    # uncompiled only
    uv run python benchmark_attention.py --compile          # both uncompiled and compiled
    uv run python benchmark_attention.py --compile --markdown
"""

import argparse
import itertools
import statistics
import timeit

import torch

# ── Fixed hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 8
WARMUP_STEPS = 5
TIMED_STEPS = 100

D_MODEL_LIST = [16, 32, 64, 128]
SEQ_LEN_LIST = [256, 1024, 4096, 8192, 16384]


# ── Naive scaled dot-product attention (no multihead) ────────────────────────
def attention(q, k, v):
    """
    q, k, v: (batch, seq_len, d_head)
    Returns: (batch, seq_len, d_head)
    """
    import math

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, T, T)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)  # (B, T, d_head)


# Compiled version — created once so torch.compile only runs once
compiled_attention = torch.compile(attention)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_one(d_model, seq_len, device, attn_fn=attention, label="uncompiled"):
    """
    Benchmark a single (d_model, seq_len) configuration.
    Returns a dict with timings and memory, or an OOM entry.
    """
    shape = (BATCH_SIZE, seq_len, d_model)

    try:
        # ── Create inputs ────────────────────────────────────────────────────
        q = torch.randn(shape, device=device, requires_grad=True)
        k = torch.randn(shape, device=device, requires_grad=True)
        v = torch.randn(shape, device=device, requires_grad=True)

        # ── Warm-up ──────────────────────────────────────────────────────────
        for _ in range(WARMUP_STEPS):
            out = attn_fn(q, k, v)
            loss = out.sum()
            loss.backward()
            q.grad = k.grad = v.grad = None
            sync()

        # ── Time forward passes ──────────────────────────────────────────────
        fwd_times = []
        for _ in range(TIMED_STEPS):
            sync()
            t0 = timeit.default_timer()
            out = attn_fn(q, k, v)
            sync()
            fwd_times.append(timeit.default_timer() - t0)

        # ── Measure memory before backward ───────────────────────────────────
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _ = attn_fn(q, k, v)
        sync()
        mem_before_bwd_gb = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else float("nan")

        # ── Time backward passes ─────────────────────────────────────────────
        bwd_times = []
        for _ in range(TIMED_STEPS):
            q2 = torch.randn(shape, device=device, requires_grad=True)
            k2 = torch.randn(shape, device=device, requires_grad=True)
            v2 = torch.randn(shape, device=device, requires_grad=True)
            out2 = attn_fn(q2, k2, v2)
            loss2 = out2.sum()
            sync()
            t0 = timeit.default_timer()
            loss2.backward()
            sync()
            bwd_times.append(timeit.default_timer() - t0)

        return {
            "label": label,
            "d_model": d_model,
            "seq_len": seq_len,
            "fwd_mean_ms": round(statistics.mean(fwd_times) * 1000, 3),
            "fwd_std_ms": round(statistics.stdev(fwd_times) * 1000, 3),
            "bwd_mean_ms": round(statistics.mean(bwd_times) * 1000, 3),
            "bwd_std_ms": round(statistics.stdev(bwd_times) * 1000, 3),
            "mem_before_bwd_GB": round(mem_before_bwd_gb, 3),
            "status": "OK",
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "label": label,
            "d_model": d_model,
            "seq_len": seq_len,
            "fwd_mean_ms": "OOM",
            "fwd_std_ms": "OOM",
            "bwd_mean_ms": "OOM",
            "bwd_std_ms": "OOM",
            "mem_before_bwd_GB": "OOM",
            "status": "OOM",
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="Print results as Markdown table")
    parser.add_argument("--compile", action="store_true", help="Also benchmark torch.compile'd attention")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}\n")

    configs = list(itertools.product(D_MODEL_LIST, SEQ_LEN_LIST))

    # decide which (label, fn) pairs to run
    runs = [("uncompiled", attention)]
    if args.compile:
        runs.append(("compiled", compiled_attention))

    results = []
    for label, attn_fn in runs:
        print(f"\n=== {label} ===")
        for d_model, seq_len in configs:
            print(f"  d_model={d_model:4d}  seq_len={seq_len:6d} ... ", end="", flush=True)
            row = benchmark_one(d_model, seq_len, device, attn_fn=attn_fn, label=label)
            results.append(row)
            if row["status"] == "OOM":
                print("OOM")
            else:
                print(f"fwd={row['fwd_mean_ms']} ms  bwd={row['bwd_mean_ms']} ms  mem={row['mem_before_bwd_GB']} GB")

    # ── Print table ──────────────────────────────────────────────────────────
    if args.markdown:
        import pandas as pd

        df = pd.DataFrame(results)
        print("\n## Attention Benchmark Results\n")
        print(df.to_markdown(index=False))
    else:
        header = (
            f"{'label':>12} {'d_model':>8} {'seq_len':>8} {'fwd_ms':>10} {'bwd_ms':>10} {'mem_GB':>10} {'status':>6}"
        )
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['label']:>12} {r['d_model']:>8} {r['seq_len']:>8} "
                f"{str(r['fwd_mean_ms']):>10} {str(r['bwd_mean_ms']):>10} "
                f"{str(r['mem_before_bwd_GB']):>10} {r['status']:>6}"
            )


if __name__ == "__main__":
    main()
