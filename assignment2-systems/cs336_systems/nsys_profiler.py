"""
nsys_profiler.py — Profile forward, backward, and optimizer step using NVTX annotations.

Run with nsys:
    uv run nsys profile -o result --pytorch python nsys_profiler.py --size small --context_length 128
    uv run nsys profile -o result --pytorch python nsys_profiler.py --size small --context_length 128 --forward_only

Profile all sizes and context lengths (run separately for each):
    for size in small medium large xl 2.7B; do
        for ctx in 128 256 512 1024; do
            uv run nsys profile -o profiles/${size}_${ctx} --pytorch \
                python nsys_profiler.py --size $size --context_length $ctx
        done
    done
"""

import argparse
import math

import cs336_basics.model as basics_model
import torch
import torch.cuda.nvtx as nvtx
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
WARMUP_STEPS = 5  # not profiled
PROFILE_STEPS = 3  # profiled (keep small — nsys output can be large)


# ── Annotated attention: breaks attention into sub-ranges for nsys ───────────
@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]

    with nvtx.range("attention_scores_matmul"):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        with nvtx.range("apply_mask"):
            scores = scores.masked_fill(mask == 0, float("-inf"))

    with nvtx.range("softmax"):
        attn_weights = torch.softmax(scores, dim=-1)

    with nvtx.range("output_matmul"):
        out = torch.matmul(attn_weights, v)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="nsys profiling script for Transformer")
    parser.add_argument("--size", choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument(
        "--forward_only", action="store_true", help="Profile forward pass only (no backward / optimizer)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type != "cuda":
        print("WARNING: nsys GPU profiling requires CUDA. Running on CPU — results may be limited.")
    print(f"Device : {device}")

    # ── Swap in annotated attention ──────────────────────────────────────────
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # ── Build model ──────────────────────────────────────────────────────────
    cfg = dict(MODEL_CONFIGS[args.size])
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        rope_theta=10000,
    ).to(device)

    mode = "forward_only" if args.forward_only else "fwd+bwd+optimizer"
    model.eval() if args.forward_only else model.train()

    optimizer = None if args.forward_only else AdamW(model.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Size: {args.size} | Params: {total_params / 1e6:.1f}M | Context: {args.context_length} | Mode: {mode}")

    # ── Random batch ─────────────────────────────────────────────────────────
    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, args.context_length), device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ── Warm-up (NVTX labelled — filter this out in nsys GUI) ────────────────
    with nvtx.range("warmup"):
        for _ in range(WARMUP_STEPS):
            with torch.set_grad_enabled(not args.forward_only):
                logits = model(batch)
                if not args.forward_only:
                    logits.mean().backward()
                    optimizer.step()
                    optimizer.zero_grad()
            sync()

    # ── Profiled steps (NVTX labelled — focus on this in nsys GUI) ───────────
    with nvtx.range("profile_region"):
        for i in range(PROFILE_STEPS):
            sync()
            with nvtx.range(f"step_{i}"):
                with nvtx.range("forward"):
                    logits = model(batch)

                if not args.forward_only:
                    with nvtx.range("loss"):
                        loss = logits.mean()

                    with nvtx.range("backward"):
                        loss.backward()

                    with nvtx.range("optimizer_step"):
                        optimizer.step()
                        optimizer.zero_grad()

            sync()

    print("Profiling complete. Open the .nsys-rep file in Nsight Systems GUI.")
    print("Tip: filter on the 'profile_region' NVTX range to exclude warm-up.")


if __name__ == "__main__":
    main()
