import os
import random
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer, _worker_encode

DOC_SEP = "<|endoftext|>"


def _worker_encode_with_len(payload: tuple[str, int]) -> tuple[list[int], int]:
    """
    接收 (文本, 原始字节数)，并原样返回 (Token ID列表, 原始字节数)
    这样我们在 imap 流式处理时就不会丢失更新进度条所需的信息。
    """
    text, raw_len = payload
    return _worker_encode(text), raw_len


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def safe_read_chunks(file_handle, chunk_size: int) -> Iterator[str]:
    """
    Yield (decoded_text, raw_bytes_consumed) pairs from a binary file handle.

    raw_bytes_consumed is the number of bytes actually read from the file in
    this iteration - used to advance a tqdm byte-level progress bar accurately.
    The leftover mechanism ensures we never decode a partial multi-byte UTF-8
    sequence at a chunk boundary.
    """
    leftover = b""
    chunk_count = 0

    while True:
        chunk = file_handle.read(chunk_size)
        chunk_count += 1
        if not chunk:
            if leftover:
                yield leftover.decode("utf-8", errors="replace"), len(leftover)
            print(f"Finished reading {chunk_count} chunks")
            break

        raw_len = len(chunk)  # bytes actually read from disk this round
        data = leftover + chunk

        # Walk back from the end to find the last valid UTF-8 boundary.
        decoded = False
        for i in range(len(data) - 1, max(-1, len(data) - 5) - 1, -1):
            try:
                text = data[: i + 1].decode("utf-8")
                leftover = data[i + 1 :]
                yield text, raw_len
                decoded = True
                break
            except UnicodeDecodeError:
                continue

        if not decoded:
            # Extremely unlikely – entire tail is invalid; replace and continue.
            yield data.decode("utf-8", errors="replace"), raw_len
            leftover = b""


def sample_documents(path: str, k: int = 10, seed: int = 42) -> list[str]:
    """Reservoir-sample *k* documents from a DOC_SEP-delimited file."""
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    buf: list[str] = []

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if DOC_SEP in line:
                doc = "".join(buf).split(DOC_SEP)[0].strip()
                buf = []
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def bytes_and_tokens(tokenizer: Tokenizer, doc: str) -> tuple[int, int]:
    n_bytes = len(doc.encode("utf-8"))
    n_tokens = len(tokenizer.encode(doc))
    return n_bytes, n_tokens


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = total_tokens = 0
    for d in docs:
        b, t = bytes_and_tokens(tokenizer, d)
        total_bytes += b
        total_tokens += t
    return total_bytes / max(total_tokens, 1)


def measure_throughput(tokenizer: Tokenizer, path: str, chunk_bytes: int = 64_000_000) -> float:
    """Return throughput in bytes/second (single-process)."""
    size = os.path.getsize(path)
    total_raw = total_tokens = 0
    t0 = time.perf_counter()

    with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True) as pbar:
        for text, raw_len in safe_read_chunks(f, chunk_bytes):
            total_raw += raw_len
            pbar.update(raw_len)  # ← track real file bytes, not re-encoded len
            ids = tokenizer.encode(text)
            total_tokens += len(ids)

    elapsed = time.perf_counter() - t0
    mbps = total_raw / elapsed / 1e6
    tps = total_tokens / elapsed
    print(f"  Throughput: {mbps:.2f} MB/s, {tps:.2e} tokens/s")
    return total_raw / elapsed


# ---------------------------------------------------------------------------
# Corpus encoder
# ---------------------------------------------------------------------------


def encode_corpus_to_uint16(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    chunk_bytes: int = 64_000_000,
    num_procs: int = 1,
) -> None:
    """
    Encode an entire text corpus to a uint16 memory-mapped numpy array.

    Key design decisions
    --------------------
    * The worker Pool is created **once** before the read loop and reused for
      every chunk.  Creating a Pool per chunk would dominate runtime due to
      fork + tokenizer pickle overhead (the vocab + merge table can be tens of
      MB).
    * With num_procs == 1 no Pool is created at all - single-process path has
      zero multiprocessing overhead.
    * The progress bar is updated with the raw byte count from the file, not
      with len(text.encode()) which can diverge when errors="replace" is used.
    * uint16 safety is checked in Tokenizer.__init__; we add an assertion here
      as a belt-and-suspenders guard.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    assert max(tokenizer.vocab.keys()) <= 65_535, (
        "Vocab IDs exceed uint16 range - switch dtype to uint32 or trim vocab."
    )
    file_size = input_path.stat().st_size
    max_tokens_est = file_size // 2

    mmap = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=(max_tokens_est,),
    )
    write_pos = 0
    last_flush_pos = 0

    pool: Pool | None = None
    if num_procs > 1:
        pool = tokenizer.make_pool(num_procs)

    try:
        with (
            open(input_path, "rb") as f,
            tqdm(total=file_size, unit="B", unit_scale=True, desc="Encoding") as pbar,
        ):
            chunk_gen = safe_read_chunks(f, chunk_bytes)
            if pool is not None:
                # 多进程流式处理分支
                results_iter = pool.imap(_worker_encode_with_len, chunk_gen, chunksize=1)

                for ids, raw_len in results_iter:
                    pbar.update(raw_len)
                    n = len(ids)

                    if write_pos + n > max_tokens_est:
                        new_size = max(max_tokens_est * 2, write_pos + n)
                        mmap.flush()
                        mmap = np.memmap(output_path, dtype=np.uint16, mode="r+", shape=(new_size,))
                        max_tokens_est = new_size

                    mmap[write_pos : write_pos + n] = ids
                    write_pos += n

                    # Keeping the fixed flush logic from earlier
                    if write_pos - last_flush_pos >= 10_000_000:
                        mmap.flush()
                        last_flush_pos = write_pos
            else:
                # 单进程备用分支
                for text, raw_len in chunk_gen:
                    pbar.update(raw_len)
                    ids = tokenizer.encode(text)
                    n = len(ids)

                    if write_pos + n > max_tokens_est:
                        new_size = max(max_tokens_est * 2, write_pos + n)
                        mmap.flush()
                        mmap = np.memmap(output_path, dtype=np.uint16, mode="r+", shape=(new_size,))
                        max_tokens_est = new_size

                    mmap[write_pos : write_pos + n] = ids
                    write_pos += n

                    if write_pos - last_flush_pos >= 10_000_000:
                        mmap.flush()
                        last_flush_pos = write_pos
    finally:
        # Always clean up the pool, even if an exception occurs.
        if pool is not None:
            pool.close()
            pool.join()

    mmap.flush()
    del mmap

    bytes_to_keep = write_pos * np.dtype(np.uint16).itemsize
    with open(output_path, "r+b") as f:
        f.truncate(bytes_to_keep)

    print(f"Saved {write_pos:,} tokens to {output_path}")


# ---------------------------------------------------------------------------
# Named experiment functions
# ---------------------------------------------------------------------------


def compression_ratio_tinystories() -> None:
    ts_docs = sample_documents("data/TinyStoriesV2-GPT4-train.txt", k=10, seed=42)
    ts_tokenizer = Tokenizer.from_files(
        "outputs/tinystories_vocab.pkl",
        "outputs/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    ratio = compression_ratio(ts_tokenizer, ts_docs)
    print(f"TinyStories tokenizer compression ratio: {ratio:.4f} bytes/token (10 docs)")


def compression_ratio_owt() -> None:
    owt_docs = sample_documents("data/owt_train.txt", k=10, seed=42)
    owt_tok = Tokenizer.from_files(
        "outputs/owt_vocab.pkl",
        "outputs/owt_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    ratio = compression_ratio(owt_tok, owt_docs)
    print(f"OpenWebText tokenizer compression ratio: {ratio:.4f} bytes/token (10 docs)")


def throughput_estimate_pile() -> None:
    tokenizer = Tokenizer.from_files(
        "outputs/owt_vocab.pkl",
        "outputs/owt_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    bps = measure_throughput(tokenizer, "data/owt_valid.txt")
    pile_bytes = 825e9
    hours = pile_bytes / bps / 3600
    print(f"Estimated time for Pile: {hours:.2f} hours")


# -------------------------------------------------------
# Encode text files
# -------------------------------------------------------


def encode_tinystories(num_procs):
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
        "dataset/tinystories_train_ids.uint16",
        chunk_bytes=64_000_000,
        num_procs=num_procs,
    )
    encode_corpus_to_uint16(
        ts_tok,
        "data/TinyStoriesV2-GPT4-valid.txt",
        "dataset/tinystories_valid_ids.uint16",
        chunk_bytes=64_000_000,
        num_procs=num_procs,
    )


def encode_owt(num_procs, chunk_bytes):
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
        "dataset/owt_train_ids.uint16",
        chunk_bytes=chunk_bytes,
        num_procs=num_procs,
    )
    encode_corpus_to_uint16(
        owt_tok,
        "data/owt_valid.txt",
        "dataset/owt_valid_ids.uint16",
        chunk_bytes=chunk_bytes,
        num_procs=num_procs,
    )


if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_cpus} CPU cores")
    encode_owt(num_cpus, 64_000_000)
