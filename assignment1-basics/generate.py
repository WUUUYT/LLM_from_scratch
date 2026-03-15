import argparse

import torch

from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


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
    # 模型结构参数，必须与训练配置严格一致
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. 加载 Tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    print("Tokenizer loaded")

    # 2. 实例化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=0.0,  # 推理时不需要 Dropout
    )
    print("Model architecture initialized")

    # 3. 加载 Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # --- 核心修复：处理 torch.compile 带来的 _orig_mod. 前缀 ---
    state_dict = checkpoint["model"]
    # 如果发现 Key 带有编译器前缀，则将其剔除
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    print(f"Weights loaded from {args.checkpoint}")

    model = model.to(device)

    # 4. 设置为评估模式（至关重要！）
    model.eval()

    # 5. 执行解码生成
    print(f"\n--- Generating (temp={args.temperature}, top_p={args.top_p}) ---")
    output = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Output:\n{args.prompt + output}")


if __name__ == "__main__":
    main()
