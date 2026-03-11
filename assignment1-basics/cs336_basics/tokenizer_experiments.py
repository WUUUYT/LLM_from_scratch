import os
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

DOC_SEP = "<|endoftext|>"


def safe_read_chunks(file_handle, chunk_size: int) -> Iterator[str]:
    leftover = b""
    chunk_count = 0

    while True:
        chunk = file_handle.read(chunk_size)
        chunk_count += 1
        if not chunk:
            if leftover:
                yield leftover.decode("utf-8", errors="replace")
            print(f"Finished reading {chunk_count} chunks")
            break

        data = leftover + chunk

        decoded = False
        for i in range(len(data) - 1, max(-1, len(data) - 5) - 1, -1):
            try:
                text = data[: i + 1].decode("utf-8")
                leftover = data[i + 1 :]
                yield text
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            yield data.decode("utf-8", errors="replace")
            leftover = b""


def sample_documents(path, k=10, seed=42):
    rng = random.Random(seed)
    sample = []
    seen = 0
    buf = []

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if DOC_SEP in line:
                doc = "".join(buf).split(DOC_SEP)[0]
                buf = []  # reset

                doc = doc.strip()
                if not doc:
                    continue

                seen += 1
                if len(sample) < k:
                    sample.append(doc)
                else:
                    j = rng.randrange(seen)
                    if j < k:
                        sample[j] = doc
    return sample


def bytes_and_tokens(tokenizer: Tokenizer, doc: str) -> tuple[int, int]:
    # bytes measured in UTF-8 bytes (same basis as byte-level BPE input)
    n_bytes = len(doc.encode("utf-8"))
    ids = tokenizer.encode(doc)
    n_tokens = len(ids)
    return n_bytes, n_tokens


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        b, t = bytes_and_tokens(tokenizer, d)
        total_bytes += b
        total_tokens += t
    return total_bytes / max(total_tokens, 1)


def measure_throughput(tokenizer, path, chunk_bytes=64_000_000):
    size = os.path.getsize(path)
    total_bytes = 0
    total_tokens = 0
    t0 = time.time()

    with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True) as pbar:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break

            total_bytes += len(b)
            pbar.update(len(b))

            text = b.decode("utf-8", errors="ignore")
            ids = tokenizer.encode(text)
            total_tokens += len(ids)

    elapsed = time.time() - t0
    mbps = total_bytes / elapsed / 1e6
    tokens_per_sec = total_tokens / elapsed
    print(f"  Throughput: {mbps:.2f} MB/s, {tokens_per_sec:.2e} tokens/s")
    return total_bytes / elapsed


def encode_corpus_to_uint16(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    chunk_bytes: int = 64_000_000,
    num_procs=1,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    file_size = input_path.stat().st_size
    max_tokens_est = file_size // 2

    mmap = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=(max_tokens_est,),
    )

    write_pos = 0

    with (
        open(input_path, "rb") as f,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Encoding") as pbar,
    ):
        for text in safe_read_chunks(f, chunk_bytes):
            pbar.update(len(text.encode("utf-8")))

            ids = tokenizer.encode(text, num_procs)
            n = len(ids)

            if write_pos + n > max_tokens_est:
                new_size = max_tokens_est * 2
                mmap.flush()
                mmap = np.memmap(
                    output_path,
                    dtype=np.uint16,
                    mode="r+",
                    shape=(new_size,),
                )
                max_tokens_est = new_size

            mmap[write_pos : write_pos + n] = ids
            write_pos += n

            if write_pos % 10_000_000 == 0:
                mmap.flush()

    mmap.flush()
    del mmap

    final = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="r+",
        shape=(write_pos,),
    )
    final.flush()

    print(f"Saved {write_pos:,} tokens to {output_path}")


def compression_ratio_tinystories():
    tinystories_text_path = "data/TinyStoriesV2-GPT4-train.txt"
    ts_docs = sample_documents(tinystories_text_path, k=10, seed=42)
    ts_vocab = "outputs/tinystories_vocab.pkl"
    ts_merges = "outputs/tinystories_merges.pkl"
    special_tokens = ["<|endoftext|>"]

    ts_tokenizer = Tokenizer.from_files(ts_vocab, ts_merges, special_tokens)
    ts_ratio = compression_ratio(ts_tokenizer, ts_docs)
    print(f"TinyStories tokenizer compression ratio: {ts_ratio:.4f} bytes/token (10 docs)")


def compression_ratio_owt():
    owt_text_path = "data/owt_train.txt"
    owt_docs = sample_documents(owt_text_path, k=10, seed=42)
    owt_vocab = "outputs/owt_vocab.pkl"
    owt_merges = "outputs/owt_merges.pkl"
    special_tokens = ["<|endoftext|>"]
    owt_tok = Tokenizer.from_files(owt_vocab, owt_merges, special_tokens)
    owt_ratio = compression_ratio(owt_tok, owt_docs)
    print(f"OpenWebText tokenizer compression ratio: {owt_ratio:.4f} bytes/token (10 docs)")


def throughput_estimate_pile():
    tokenizer = Tokenizer.from_files(
        "outputs/owt_vocab.pkl",
        "outputs/owt_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    bps = measure_throughput(tokenizer, "data/owt_valid.txt")
    print(f"Throughput: {bps:.2e} bytes/sec")

    pile_bytes = 825e9
    hours = pile_bytes / bps / 3600
    print(f"Estimated time for Pile: {hours:.2f} hours")


def encode_tinystories():
    # TinyStories
    print("\n=== Encoding TinyStories ===")
    ts_tok = Tokenizer.from_files(
        "outputs/tinystories_vocab.pkl",
        "outputs/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    encode_corpus_to_uint16(
        ts_tok,
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/tinystories_train_ids.uint16",
        chunk_bytes=64_000_000,
        num_procs=4,
    )


def encode_tinystories_test():
    print("\n=== Encoding TinyStories Test Set ===")
    ts_tok = Tokenizer.from_files(
        "outputs/tinystories_vocab.pkl",
        "outputs/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    encode_corpus_to_uint16(
        ts_tok,
        "data/TinyStoriesV2-GPT4-valid.txt",
        "data/tinystories_test_ids.uint16",
        chunk_bytes=64_000_000,
        num_procs=4,
    )


def encode_owt():
    # OpenWebText
    print("\n=== Encoding OpenWebText ===")
    owt_tok = Tokenizer.from_files(
        "outputs/owt_vocab.pkl",
        "outputs/owt_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    encode_corpus_to_uint16(
        owt_tok,
        "data/owt_train.txt",
        "data/owt_train_ids.uint16",
        chunk_bytes=256_000_000,
        num_procs=1,
    )


if __name__ == "__main__":
    encode_tinystories_test()
