# cs336_basics/tokenizer.py

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# ---------------------------------------------------------------------------
# Module-level worker state
# Must be top-level (not methods / lambdas) so multiprocessing can pickle them.
# ---------------------------------------------------------------------------
_worker_tokenizer: Tokenizer | None = None


def _init_worker(tokenizer: Tokenizer) -> None:
    """Called once per worker process when the Pool is created."""
    global _worker_tokenizer
    _worker_tokenizer = tokenizer
    # Each worker gets its own independent LRU cache – no cross-process locking.
    _worker_tokenizer._encode_word_cached = lru_cache(maxsize=10_000)(_worker_tokenizer._encode_word)


def _worker_encode(text: str) -> list[int]:
    """Top-level function executed in each worker process."""
    assert _worker_tokenizer is not None, "Worker not initialised"
    return _worker_tokenizer._encode(text)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    specials = sorted(set(special_tokens), key=len, reverse=True)
    pattern = "|".join(re.escape(s) for s in specials)
    parts = re.split(f"({pattern})", text)
    return [p for p in parts if p]


def _pretokenize_to_bytes(text: str) -> list[bytes]:
    tokens = []
    for m in PAT.finditer(text):
        word = m.group(0)
        # 强制切分极其长的连续字符串（如连续几万个空格/标点）
        # 将最大长度限制在 1000 字符，彻底避免 O(N^2) 算力卡死
        for i in range(0, len(word), 1000):
            tokens.append(word[i : i + 1000].encode("utf-8"))
    return tokens


def _apply_bpe_merges(word: bytes, merge_ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    if len(word) == 1:
        return [word]
    tokens = [bytes([b]) for b in word]
    while len(tokens) >= 2:
        min_rank = float("inf")
        min_idx = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merge_ranks.get(pair, float("inf"))
            if rank < min_rank:
                min_rank = rank
                min_idx = i
        if min_idx == -1 or min_rank == float("inf"):
            break
        tokens[min_idx : min_idx + 2] = [tokens[min_idx] + tokens[min_idx + 1]]
    return tokens


def _group_parts_into_chunks(parts: list[str], n: int) -> list[str]:
    """
    Merge a list of string parts into exactly *n* balanced chunks (by char length).

    Crucially, parts are only merged - never split - so we never cut through a
    special token or mid-word.  If len(parts) <= n we just return parts as-is.
    """
    if len(parts) <= n:
        return [p for p in parts if p]

    total_len = sum(len(p) for p in parts)
    target = total_len / n

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for part in parts:
        buf.append(part)
        buf_len += len(part)
        # Flush when we've accumulated ≥ target AND we still have room for more chunks
        if buf_len >= target and len(chunks) < n - 1:
            chunks.append("".join(buf))
            buf = []
            buf_len = 0

    if buf:
        chunks.append("".join(buf))

    return chunks


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------


class Tokenizer:
    """
    Byte-pair-encoding tokenizer.

    For single-process use:
        ids = tokenizer.encode(text)

    For multi-process use, create a Pool *once* with the helper and reuse it:
        pool = tokenizer.make_pool(num_workers)
        try:
            ids = tokenizer.encode_parallel(text, pool, num_workers)
        finally:
            pool.close(); pool.join()
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        self.id_to_bytes = self.vocab
        self.bytes_to_id = {b: i for i, b in self.vocab.items()}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
        self._special_str_to_id = {}
        for s in self.special_tokens:
            sb = s.encode("utf-8")
            if sb in self.bytes_to_id:
                sid = self.bytes_to_id[sb]
            else:
                sid = next_id
                next_id += 1
                self.vocab[sid] = sb
                self.bytes_to_id[sb] = sid
            self._special_str_to_id[s] = sid

        max_id = max(self.vocab.keys()) if self.vocab else 0
        if max_id > 65_535:
            raise ValueError(
                f"Max vocab ID {max_id} exceeds uint16 range (65 535). "
                "Either trim the vocabulary or switch the output dtype to uint32."
            )

        self._encode_word_cached = lru_cache(maxsize=10_000)(self._encode_word)

    # ------------------------------------------------------------------
    # Core encode helpers
    # ------------------------------------------------------------------

    def _encode_word(self, word_bytes: bytes) -> tuple[int, ...]:
        tokens = _apply_bpe_merges(word_bytes, self.merge_ranks)
        return tuple(self.bytes_to_id[tok] for tok in tokens)

    def _encode(self, text: str) -> list[int]:
        """Single-threaded encode (no Pool overhead)."""
        if not text:
            return []
        ids: list[int] = []
        for chunk in _split_by_special_tokens(text, self.special_tokens):
            if chunk in self._special_str_to_id:
                ids.append(self._special_str_to_id[chunk])
            else:
                for word_bytes in _pretokenize_to_bytes(chunk):
                    ids.extend(self._encode_word_cached(word_bytes))
        return ids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* in the calling process (no multiprocessing overhead)."""
        return self._encode(text)

    def make_pool(self, num_workers: int) -> Pool:
        """
        Create a worker Pool with this tokenizer pre-loaded into every worker.

        The Pool must be closed/joined by the caller:
            pool = tok.make_pool(4)
            try:
                ...
            finally:
                pool.close(); pool.join()

        Why initializer?
        - The Tokenizer object is sent to workers *once* at pool creation time.
        - Each subsequent task sends only a plain string (cheap).
        - Workers share nothing: no locks, no shared memory, no risk of deadlock.
        - Each worker has its own independent LRU cache that warms up over its
          lifetime without cross-process synchronisation.
        """
        return Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self,),
        )

    def encode_parallel(self, text: str, pool: Pool, num_workers: int) -> list[int]:
        """
        Encode *text* across *num_workers* processes using a pre-created *pool*.

        The pool must have been created with :meth:`make_pool` (or equivalently,
        ``Pool(n, initializer=_init_worker, initargs=(self,))``).

        Falls back to single-process :meth:`encode` when the text is too short
        to be worth distributing.
        """
        # Split only at special-token boundaries so we never break a token.
        parts = _split_by_special_tokens(text, self.special_tokens)

        # Not worth the IPC overhead for tiny inputs.
        if len(parts) < num_workers * 2:
            return self._encode(text)

        chunks = _group_parts_into_chunks(parts, num_workers)

        # pool.map keeps results ordered; chunksize controls IPC batch size.
        results: list[list[int]] = pool.map(_worker_encode, chunks, chunksize=1)

        final: list[int] = []
        for r in results:
            final.extend(r)
        return final

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode(chunk)

    def decode(self, ids: list[int]) -> str:
        try:
            b = b"".join(self.id_to_bytes[i] for i in ids)
            return b.decode("utf-8", "replace")
        except KeyError as e:
            raise ValueError(f"Invalid token ID: {e}") from e

    # ------------------------------------------------------------------
    # Pickle support  (lru_cache wraps a bound method → not picklable)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_encode_word_cached", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._encode_word_cached = lru_cache(maxsize=10_000)(self._encode_word)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        import pickle

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
