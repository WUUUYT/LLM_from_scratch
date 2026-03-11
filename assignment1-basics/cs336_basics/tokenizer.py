# cs336_basics/tokenizer.py

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    specials = sorted(set(special_tokens), key=len, reverse=True)
    pattern = "|".join(re.escape(s) for s in specials)
    parts = re.split(f"({pattern})", text)
    return [p for p in parts if p]


def _pretokenize_to_bytes(text: str) -> list[bytes]:
    return [m.group(0).encode("utf-8") for m in PAT.finditer(text)]


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


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

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
                self.id_to_bytes[sid] = sb
            self._special_str_to_id[s] = sid
        self._encode_word_cached = lru_cache(maxsize=10000)(self._encode_word)

    def _encode_word(self, word_bytes: bytes) -> tuple[int, ...]:
        tokens = _apply_bpe_merges(word_bytes, self.merge_ranks)
        return tuple(self.bytes_to_id[tok] for tok in tokens)

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

    def _encode(self, text: str) -> list[int]:
        if text == "":
            return []

        ids = []
        chunks = _split_by_special_tokens(text, self.special_tokens)
        for chunk in chunks:
            if chunk in self._special_str_to_id:
                ids.append(self._special_str_to_id[chunk])
                continue

            for word_bytes in _pretokenize_to_bytes(chunk):
                ids.extend(self._encode_word_cached(word_bytes))

        return ids

    def __getstate__(self):
        # 序列化时删除不可 pickle 的缓存对象
        state = self.__dict__.copy()
        if "_encode_word_cached" in state:
            del state["_encode_word_cached"]
        return state

    def __setstate__(self, state):
        # 子进程拿到数据后，重新初始化缓存
        self.__dict__.update(state)
        self._encode_word_cached = lru_cache(maxsize=10000)(self._encode_word)

    def encode(self, text: str, num_workers: int = 1) -> list[int]:
        if num_workers == 1:
            return self._encode(text)
        chunks = _split_by_special_tokens(text, self.special_tokens)

        if len(chunks) < num_workers * 2:
            return self._encode(text)

        with Pool(num_workers) as pool:
            final_ids = []
            for result in pool.imap(self._encode, chunks):
                final_ids.extend(result)

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        try:
            b = b"".join(self.id_to_bytes[_id] for _id in ids)
            return b.decode("utf-8", "replace")
        except KeyError as e:
            raise ValueError(f"Invalid token ID: {e}")
