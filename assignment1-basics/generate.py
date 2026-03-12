"""
python generate.py \
    --checkpoint checkpoints/lr_1e-4/ckpt_0005000.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda
"""


import torch
import argparse
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import decode
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default="outputs/tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default="outputs/tinystories_merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    # 模型结构参数，要和训练时一致
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    return parser.parse_args()

def main():
    args = get_args()
    print("Args loaded")

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"]
    )
    print("Tokenizer loaded")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=0.0,
    )
    print("Model created")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print("Checkpoint loaded")

    model.load_state_dict(checkpoint["model"])
    print("State dict loaded")

    model = model.to(args.device)

    # 生成文本
    output = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print(f"Prompt: {args.prompt}")
    print(f"Generated:\n{args.prompt + output}")

if __name__ == "__main__":
    main()
