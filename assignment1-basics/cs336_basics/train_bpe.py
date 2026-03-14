import heapq
import logging
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# '(?:[sdmt]|ll|ve|re) — 's, 're, 'll
# ?\p{L}+ — letter sequence
# ?\p{N}+ — number sequence
# ?[^\s\p{L}\p{N}]+ — characters
# \s+(?!\S)|\s+ — space


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += len(mini_chunk)

    return sorted(set(chunk_boundaries))  # return a list


def pretokenize(text: str, special_tokens: list[str]) -> list[bytes]:
    if not special_tokens:
        chunks = [text]
    else:
        # re.escape: Escapes all special regex characters in a string so it can be used as a literal pattern.
        pattern = "|".join(re.escape(t) for t in special_tokens)
        # not keeping tokens themselves
        chunks = re.split(pattern, text)
    words = []
    for chunk in chunks:
        if chunk:
            # Iterator
            matches = re.finditer(PAT, chunk)
            for match in matches:
                words.append(match.group(0).encode("utf-8"))
                # group() or group(0) returns the whole match result
                # group(1) return the subgroup captured by parentheses
                # m = re.search(r'(\w+)@(\w+)', "user@gmail")
                # print(m.group(0))  # 'user@gmail'
                # print(m.group(1))  # 'user'
                # print(m.group(2))  # 'gmail'
    return words


######## Heap ########
# counts might get outdated after merge. Thus, we need to check if the pop item counts eqauls to pair_counts[word]
class HeapItem:
    def __init__(self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]):
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes


def build_pair_heap(pair_counts: Counter, vocab: dict[int, bytes]):
    h = []
    for (a, b), f in pair_counts.items():
        if f > 0:
            item = HeapItem(-f, (vocab[a], vocab[b]), (a, b))
            heapq.heappush(h, item)
    return h


def pop_most_frequent_pair(h, pair_counts: Counter) -> tuple[int, int]:
    while h:
        item = h[0]
        pair = item.pair
        neg_f = item.neg_freq
        cur_f = pair_counts.get(pair, 0)
        if cur_f <= 0 or -neg_f != cur_f:
            heapq.heappop(h)
            continue
        heapq.heappop(h)
        return pair
    raise ValueError("No positive-frequency pairs remain")


######################


######## Heap + index ########
def merge_pairs_with_heap_index(
    word_counts: Counter,
    pair_counts: Counter,
    best_pair: tuple[int, int],
    next_id: int,
    vocab: dict[int, bytes],
    h,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int, ...], int],
    Counter,
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    """
    1. Find affected words for the best pair
    2. iterate word
        1. delete word count
        2. subtract pair count for each pair in the word
        3. add pair to 'changed_pair'
        4. discard word in 'pair_to_words'
        5. get new word, update word counts
        6. for each new pair in the new word: update pair counts, update 'changed_pairs' set, update 'pair_to_words'
    3. for each changed pair (count > 0), construct a heapitem and push.

    """
    changed_pairs = set()
    # only work with words that contain 'best_pair'
    affected_words = list(pair_to_words.get(best_pair, set()))

    for word in affected_words:
        count = word_counts.get(word, 0)
        if count <= 0 or len(word) < 2:
            continue

        # clear old words and related pairs
        del word_counts[word]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] -= count
            changed_pairs.add(pair)
            if pair in pair_to_words:
                pair_to_words[pair].discard(word)

        # merge
        new_word = _get_new_word(word, best_pair, next_id)
        word_counts[new_word] += count

        # pairs after merge
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                new_p = new_word[i : i + 2]
                pair_counts[new_p] += count
                changed_pairs.add(new_p)
                pair_to_words.setdefault(new_p, set()).add(new_word)

    for p in changed_pairs:
        count = pair_counts.get(p, 0)
        if count > 0:
            item = HeapItem(-count, (vocab[p[0]], vocab[p[1]]), p)
            heapq.heappush(h, item)
    return word_counts, pair_counts, h, pair_to_words


def _get_new_word(word: tuple[int, ...], pair: tuple[int, int], next_id: int) -> tuple[int, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(next_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


######################


def _process_chunk(input_path: str, start: int, end: int, special_tokens: list[str], queue):
    local_counts = Counter()
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", "ignore")
        words = pretokenize(chunk, special_tokens)
        for word in words:
            local_counts[tuple(word)] += 1  # b"hi" -> (104, 105)
        logging.info(f"DONE chunk {start}-{end}, words={len(local_counts)}")
        queue.put(local_counts)
    except Exception as e:
        print(f"Chunk {start}-{end} wrong: {e}")


def _process_chunk_streaming(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
    queue,
    mini_chunk_size: int = 8 * 1024 * 1024,  # 8MB mini-chunks
):
    local_counts = Counter()
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            remaining = end - start
            buffer = b""

            processed_bytes = 0
            while remaining > 0:
                read_size = min(mini_chunk_size, remaining)
                chunk_bytes = buffer + f.read(read_size)
                remaining -= read_size
                processed_bytes += read_size

                if remaining > 0:
                    buffer = b""
                    # Preserve incomplete UTF-8 sequence
                    for i in range(1, min(5, len(chunk_bytes)) + 1):
                        try:
                            text = chunk_bytes[:-i].decode("utf-8")
                            buffer = chunk_bytes[-i:]
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = chunk_bytes.decode("utf-8", "ignore")
                        buffer = b""
                else:
                    text = chunk_bytes.decode("utf-8", "ignore")

                words = pretokenize(text, special_tokens)
                for word in words:
                    local_counts[tuple(word)] += 1
                del text, words, chunk_bytes

                if processed_bytes % (50 * 1024 * 1024) == 0:
                    logging.info(f"Chunk {start}-{end}: processed {processed_bytes // (1024 * 1024)}MB")

        logging.info(f"Chunk {start}-{end} done, unique words={len(local_counts)}")
        queue.put(local_counts)
    except Exception as e:
        logging.error(f"Chunk {start}-{end} error: {e}")
        queue.put(Counter())


def _process_chunk_worker(args):
    input_path, start, end, special_tokens, mini_chunk_size = args
    local_counts = Counter()
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            remaining = end - start
            buffer = b""

            while remaining > 0:
                read_size = min(mini_chunk_size, remaining)
                chunk_bytes = buffer + f.read(read_size)
                remaining -= read_size

                if remaining > 0:
                    buffer = b""
                    for i in range(1, min(5, len(chunk_bytes)) + 1):
                        try:
                            text = chunk_bytes[:-i].decode("utf-8")
                            buffer = chunk_bytes[-i:]
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = chunk_bytes.decode("utf-8", "ignore")
                        buffer = b""
                else:
                    text = chunk_bytes.decode("utf-8", "ignore")

                words = pretokenize(text, special_tokens)
                for word in words:
                    local_counts[tuple(word)] += 1

                del text, words, chunk_bytes

        logging.info(f"✓ Chunk {start}-{end} done, unique words={len(local_counts)}")
        return local_counts

    except Exception as e:
        logging.error(f"✗ Chunk {start}-{end} error: {e}")
        return Counter()


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    profile=False,
    num_procs=2,
    mini_chunk_size=8 * 1024 * 1024,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    if profile:
        ctx = mp.get_context("fork")
    else:
        ctx = mp

    logging.info(f"Using {num_procs} processes")
    split_token = (special_tokens[0] if special_tokens else " <|endoftext|>").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_procs, split_token)
    logging.info(f"Chunk boundaries: {boundaries}")
    ######## Begin Parallel ########
    tasks = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens, mini_chunk_size)
        for i in range(len(boundaries) - 1)
    ]
    word_counts = Counter()

    with ctx.Pool(processes=num_procs) as pool:
        for idx, partial_counter in enumerate(pool.imap_unordered(_process_chunk_worker, tasks, chunksize=1)):
            word_counts.update(partial_counter)
            logging.info(f"Merged result {idx + 1}/{len(tasks)}, total unique words: {len(word_counts)}")
            del partial_counter
            if (idx + 1) % 5 == 0:
                import gc

                gc.collect()

    logging.info(f"All chunks processed. Total unique words: {len(word_counts)}")
    ######## End parallel ########

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    h = build_pair_heap(pair_counts, vocab)

    merges = []
    logging.info("Merge starts...")
    while len(vocab) < vocab_size:
        best_pair = pop_most_frequent_pair(h, pair_counts)

        pair_bytes = (vocab[best_pair[0]], vocab[best_pair[1]])
        merges.append(pair_bytes)
        vocab[next_id] = pair_bytes[0] + pair_bytes[1]

        word_counts, pair_counts, h, pair_to_words = merge_pairs_with_heap_index(
            word_counts, pair_counts, best_pair, next_id, vocab, h, pair_to_words
        )
        next_id += 1
        if next_id % 1000 == 0:
            logging.info(f"cur vocab/total vocab: {len(vocab)}/{vocab_size}")
    return vocab, merges


if __name__ == "__main__":
    import os
    import pickle
    import time

    ################Configurations#####################
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    data_path = "data/"
    input_path = "owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    profile = True
    num_procs = 4
    output_path_vocab = "owt_vocab.pkl"
    output_path_merges = "owt_merges.pkl"
    mini_chunk_size = 8 * 1024 * 1024
    ###############################################

    print("Training BPE...")
    start = time.time()
    vocab, merges = train_bpe(
        input_path=data_path + input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        profile=profile,
        num_procs=num_procs,
        mini_chunk_size=mini_chunk_size,
    )
    end = time.time()
    print("Done.")
    print(f"Training took {end - start:.2f} seconds")
    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest.decode('utf-8', errors='replace')} ({len(longest)} bytes)")

    with open(output_path_vocab, "wb") as f:
        pickle.dump(vocab, f)
    with open(output_path_merges, "wb") as f:
        pickle.dump(merges, f)
