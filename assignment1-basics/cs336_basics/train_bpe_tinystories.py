import os
import pickle
import time

from cs336_basics.train_bpe import train_bpe

################Configurations#####################
output_dir = "/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics/outputs/"
os.makedirs(output_dir, exist_ok=True)
data_path = "/storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics/data/"
input_path = "TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
profile = False
num_procs = 4
mini_chunk_size = 8 * 1024 * 1024
output_path_vocab = output_dir + "/" + "tinystories_vocab.pkl"
output_path_merges = output_dir + "/" + "tinystories_merges.pkl"
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
