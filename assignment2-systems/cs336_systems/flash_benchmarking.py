"""
flash_benchmarking.py — Compare FlashAttention-2 (Triton+compile) vs vanilla PyTorch attention.

Uses triton.testing.do_bench for accurate GPU timing.

Usage:
    uv run python flash_benchmarking.py
    uv run python flash_benchmarking.py --markdown
"""

import argparse
import itertools
import math

import pandas as pd
import torch
import triton
import triton.testing
from flash_attention_triton import FlashAttentionTriton


# ── Vanilla PyTorch attention (causal) ───────────────────────────────────────
def vanilla_attention(Q, K, V, is_causal=True):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N]
    if is_causal:
        mask = torch.ones(S.shape[-2], S.shape[-1], device=Q.device, dtype=torch.bool).tril()
        S = S.masked_fill(~mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


# ── Benchmark one configuration ───────────────────────────────────────────────
def benchmark_config(seq_len, d_head, dtype, device="cuda"):
    """
    Returns a dict with forward, backward, and end-to-end latencies (ms)
    for both vanilla and FlashAttention-2, or "OOM" on failure.
    """
    B, H = 1, 1  # batch=1, single head (no multihead dim needed by flash impl)

    def make_inputs():
        Q = torch.randn(B, H, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        return Q, K, V

    row = {
        "seq_len": seq_len,
        "d_head": d_head,
        "dtype": str(dtype).replace("torch.", ""),
    }

    # ── Vanilla attention ─────────────────────────────────────────────────────
    try:
        Q, K, V = make_inputs()

        def vanilla_fwd():
            return vanilla_attention(Q, K, V, is_causal=True)

        def vanilla_fwd_bwd():
            out = vanilla_attention(Q, K, V, is_causal=True)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None

        # forward only
        out_v = vanilla_fwd()

        def vanilla_bwd():
            out_v.sum().backward(retain_graph=True)

        row["vanilla_fwd_ms"] = triton.testing.do_bench(vanilla_fwd)
        row["vanilla_bwd_ms"] = triton.testing.do_bench(vanilla_bwd)
        row["vanilla_e2e_ms"] = triton.testing.do_bench(vanilla_fwd_bwd)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        row["vanilla_fwd_ms"] = row["vanilla_bwd_ms"] = row["vanilla_e2e_ms"] = "OOM"

    # ── FlashAttention-2 (Triton) ─────────────────────────────────────────────
    try:
        Q, K, V = make_inputs()

        def flash_fwd():
            return FlashAttentionTriton.apply(Q, K, V, True)

        def flash_fwd_bwd():
            out = FlashAttentionTriton.apply(Q, K, V, True)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None

        out_f = flash_fwd()

        def flash_bwd():
            out_f.sum().backward(retain_graph=True)

        row["flash_fwd_ms"] = triton.testing.do_bench(flash_fwd)
        row["flash_bwd_ms"] = triton.testing.do_bench(flash_bwd)
        row["flash_e2e_ms"] = triton.testing.do_bench(flash_fwd_bwd)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        row["flash_fwd_ms"] = row["flash_bwd_ms"] = row["flash_e2e_ms"] = "OOM"

    # ── Speedup ───────────────────────────────────────────────────────────────
    for phase in ("fwd", "bwd", "e2e"):
        v = row.get(f"vanilla_{phase}_ms")
        f = row.get(f"flash_{phase}_ms")
        if isinstance(v, float) and isinstance(f, float):
            row[f"speedup_{phase}"] = round(v / f, 2)
        else:
            row[f"speedup_{phase}"] = "N/A"

    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="Print results as Markdown table")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this benchmark.")
        return

    device = "cuda"

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_heads = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    results = []
    configs = list(itertools.product(seq_lens, d_heads, dtypes))

    for seq_len, d_head, dtype in configs:
        print(f"  seq={seq_len:6d}  d={d_head:3d}  dtype={str(dtype).replace('torch.', '')} ...", end="", flush=True)
        row = benchmark_config(seq_len, d_head, dtype, device=device)
        results.append(row)
        print(f"  vanilla_fwd={row['vanilla_fwd_ms']}  flash_fwd={row['flash_fwd_ms']}")

    df = pd.DataFrame(results)

    # Round float columns
    float_cols = [c for c in df.columns if df[c].dtype == float]
    df[float_cols] = df[float_cols].round(3)

    if args.markdown:
        print("\n## FlashAttention-2 vs Vanilla Attention Benchmark\n")
        print(df.to_markdown(index=False))
    else:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
