"""
Training script for TransformerLM
Usage:
    python train.py \
        --train_path data/owt_train_ids.uint16 \
        --val_path data/owt_val_ids.uint16 \
        --vocab_size 32000 \
        --d_model 512 \
        --num_heads 8 \
        --num_layers 6 \
        --d_ff 2048 \
        --context_length 256 \
        --batch_size 32 \
        --max_iters 10000 \
        --max_lr 3e-4 \
        --min_lr 3e-5 \
        --warmup_iters 1000 \
        --checkpoint_dir checkpoints/ \
        --checkpoint_every 500 \
        --val_every 100 \
        --device cuda
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.transformer_lm import TransformerLM


def get_args():
    parser = argparse.ArgumentParser(description="Train a TransformerLM")
    # Data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_tying", action="store_true")
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # Logging & checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_iters", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336_lm")
    # System
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--n_cpus", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="uint16")

    return parser.parse_args()


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, val_iters):
    model.eval()
    losses = []
    for _ in range(val_iters):
        x, y = data_loading(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = get_args()
    if args.device == "mps":
        pass
    elif args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- load data (memmap) -------------------------------
    train_data = np.memmap(args.train_path, dtype=np.dtype(args.dtype), mode="r")
    val_data = np.memmap(args.val_path, dtype=np.dtype(args.dtype), mode="r")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # -- initialize model ---------------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        weight_tying=args.weight_tying,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.device == "cpu":
        model = torch.compile(model)
    elif args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    # -- intialize optimizer ------------------------------
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # -- load from checkpoint -----------------------------
    start_iter = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # -- initialize wandb ----------------------------------
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # -- training ------------------------------------------
    model.train()
    t0 = time.time()
    train_start_time = time.time()

    pbar = tqdm(range(start_iter + 1, args.max_iters + 1))
    for iteration in pbar:
        lr = learning_rate_schedule(
            t=iteration,
            lr_max=args.max_lr,
            lr_min=args.min_lr,
            Tw=args.warmup_iters,
            Tc=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), args.max_grad_norm)

        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        # -- logging ----------------------------------------
        if iteration % args.val_every == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            val_loss = estimate_val_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                device,
                args.val_iters,
            )
            train_ppl = torch.exp(loss).item()
            val_ppl = math.exp(val_loss)
            print(
                f"iter {iteration:6d} | "
                f"loss {loss.item():.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"ppl {train_ppl:.2f} | "
                f"val_ppl {val_ppl:.2f} | "
                f"lr {lr:.2e} | "
                f"dt {dt:.1f}s"
            )
            if args.wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/ppl": train_ppl,
                        "val/loss": val_loss,
                        "val/ppl": val_ppl,
                        "lr": lr,
                        "wallclock_time": time.time() - train_start_time,
                    },
                    step=iteration,
                )
        if iteration % args.checkpoint_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"ckpt_{iteration:07d}.pt"
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete!")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
