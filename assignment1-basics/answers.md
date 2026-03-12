# Answers


## BPE Tokenizer
### Unicode
1. 字符编码标准，用于统一表示世界上几乎所有语言的字符
2. Unicode 给每个字符分配一个唯一 代码点（code point），形式为 U+xxxx。
   U+0041 → 字母 A；U+4E2D → 中文 中；U+1F600 → 表情符号 😀
3. U+0041：Unicode 码点编号（十六进制写法）
4. In Python, use the `ord()` function to convert a single Unicode character into its integer representation.
5. The `chr()` function converts an integer Unicode code point into a string.
- #### P1: The Unicode Standard
- 1. chr(0) → Unicode U+0000（Null character, NULL, \x00）print无结果显示
- 2. Python中，`__str()__()`是给人看的输出，什么都不显示；`__repr__()`是对程序员的表示。 Its repr shows "\x00", but printing it produces no visible output.
`print(repr(chr(0))) leads to '\x00'`
- 3. It behaves as an invisible null character, so printed text appears concatenated with no visible separator.
---
### Bytes and utf-8
- Impractical to train on Unicode codepoints. Too large and sparse vocab.
- 先用 Unicode 编码（encoding）把字符转换成「字节序列」，再在字节或字节序列之上训练 tokenizer。
- python string本身表示的就是Unicode。
- UTF-8：变长编码（1–4 字节）
- 用`encode()`把Unicode string转换成UTF-8。` test_string.encode("utf-8")`, 类型是bytes。
- 用`decode()`把UTF-8转换为Unicode string。
- `b'hello! \xe3\x81\x93...'`：
   - b'...' 表示这是 字节字面量； ASCII 范围内的字节（如hello!）直接显示为字符。
   - 非 ASCII 字节用 \xNN 十六进制转义显示（例如 \xe3）。
- `list(utf8_encoded)` -> `[129, 147, ...]`
   - bytes 本质是 0–255 的整数序列。
   - 每个数就是 一个字节的数值。中文占三个字节
- `casdc`

<div align="center">

| 二进制高位 | 含义            | 十六进制范围 |
| ---------- | --------------- | ------------ |
| `0xxxxxxx` | 单字节（ASCII） | `00–7F`      |
| `10xxxxxx` | 续字节          | `80–BF`      |
| `110xxxxx` | 2 字节起始      | `C0–DF`      |
| `1110xxxx` | 3 字节起始      | `E0–EF`      |
| `11110xxx` | 4 字节起始      | `F0–F7`      |

</div>

### Unicode Encodings
1. We use utf-8 because it is more compact for most real-world text (especially ASCII-heavy data) and produces byte sequences that are easy for tokenizers to learn from. UTF-16 and UTF-32 use more bytes per character on average, making sequences unnecessarily long and increasing memory and training cost.
2. Because it decodes each byte independently, but UTF-8 uses multi-byte sequences to represent many characters; for example, b"\xc3\xb1" (the UTF-8 encoding of "ñ") is incorrectly decoded as "Ã±".
3. UTF-8 continuation bytes must be like 10xxxxxx (0x80–0xBF).b"\xC3\x28" is invalid because 0xC3 expects a UTF-8 continuation byte, but 0x28 is not a valid continuation byte.
   - 二进制里以 10 开头的是 UTF-8 的续字节
   - `\x41" == "A"      # True`
   - `print("\x41")     # A`

### Subword Tokenization Note
If the byte sequence b'the' often occurs in our raw text training data, assigning it an entry in the vocabulary would reduce this 3-token sequence to a single token.
Use byte-pair encoding, a compression algorithm that iteratively replaces (“merges”) the most frequent pair of bytes with a single, new unused index. The process of constructing the BPE tokenizer vocabulary is known as “training” the BPE tokenizer.

### BPE Tokenizer Training
1. Our initial vocabulary is of size 256。
2. Pre-tokenize. Split the words (eg. by spaces), and count the byte pair in that word.
3. We use regex-based pre-tokenizer
4. `PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
   1. `'(?:[sdmt]|ll|ve|re)`, match common ends like 's, 'd, 'm, 't 'll, 've, 're
   2. `| ?\p{L}+`, match optional space + continuous characters. \p{L} = Unicode. Treat a word and its preceding space as a pre-token.
   3. `| ?\p{N}+`, mtach optional space + continuous numbers.
   4. `| ?[^\s\p{L}\p{N}]+`, match optional space + non-character.
   5. `| \s+(?!\S)`, match line-ending space
   6. `| \s+`, match general space.
5. When coding, use `re.finditer` instead of `re.findall` to avoid storing.
6. When computing merges, deterministically break ties in pair frequency by preferring the lexicographically greater pair.
7. “special tokens” should never be split (the end-of-sequence string <|endoftext|>). These special tokens must be added to the vocabulary, so they have a corresponding fixed token ID.

### Experiment on `train_bpe.py`
1. Training took 37.83 seconds
2. Longest token:  accomplishment (15 bytes)
3. `/usr/bin/time -l python cs336_basics/train_bpe.py`
   - Memory: 1847508992 (1.72 GB) maximum resident set size
4. Time
   `python -m pstats profile.stats`
   `% sort cumulative`
   `% stats 20`
   `% sort tottime`
   `% stats 20`
   - File I/O (reading chunks in parallel) takes the most time:
      - (~91% or 48.8s out of 53.4s total).

   - The actual BPE merge algorithm only takes ~3% of total time (1.67s), which shows the heap-based implementation is quite efficient. The merge loop runs 9,743 times with an average of 0.17ms per merge.
   - Breakdown:
      1. File I/O: 91.4%
      2. BPE merging: 3.1%
      3. Heap operations: 1.1%
      4. Other (Counter, etc.): 4.4%

### Experiment on `Tokenizer` （to be updated)
1. TS Compression ratio: `4.1043 bytes/token (10 docs)`
   OWT Compression ratio: `4.6653 bytes/token (10 docs)`
2. OWT TK TS: `3.1172 bytes/token (10 docs)`
   TS TK OWT: `3.1851 bytes/token (10 docs)`
3. Throughput: 2.18e+04 bytes/sec
   Estimated time for Pile: 10530.67 hours

--------------

## Transformer Architecture
### Transformer accounting
Consider GPT-2 XL, which has the following configuration:
   `vocab_size: 50,257,
      context_length: 1,024,
      num_layers: 48,
      d_model: 1,600,
      num_heads: 25,
      d_ff: 6,400`

1. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

   **Answer**:
   1. Embedding: `vocab * d_model = 50257 * 1600 = 80411200 `
   2. Each Transformer block:
      - W_qkv: `3 * d_model * d_model = 3 * 1600 * 1600 = 7680000`
      - W_o: `d_model × d_model = 2,560,000`
      - W1, W3: `2 × d_ff × d_model = 2 × 6400 × 1600 = 20,480,000`
      - `W2: d_ff × d_model = 10,240,000`
      - `RMSNorm × 2: 2 × d_model = 3,200`
      - Total: `40,963,200`
   3. num_layers: `48 × 40,963,200 = 1,966,233,600`
   4. Final RMSNorm: `1,600`
   5. lm_head: `vocab_size × d_model = 80,411,200`
   Total Parameter count: `2,127,057,600`
   Memory: (4 bytes/float32) `2,127,057,600 × 4 ≈ 8.51 GB`

2. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

   **Answer**:
   *Per TransformerBlock:*
   | Operation              | Shape                   | FLOPs      |
   | ---------------------- | ----------------------- | ---------- |
   | QKV projection         | `(T×d) × (d×3d)`        | `6Td²`     |
   | QKᵀ (attention scores) | `(T×d_k) × (d_k×T) × H` | `2T²d`     |
   | AV (attention output)  | `(T×T) × (T×d_k) × H`   | `2T²d`     |
   | Output projection W_o  | `(T×d) × (d×d)`         | `2Td²`     |
   | FFN W1, W3             | `2 × (T×d) × (d×d_ff)`  | `4Td·d_ff` |
   | FFN W2                 | `(T×d_ff) × (d_ff×d)`   | `2Td·d_ff` |

   **Per block total: `8Td² + 4T²d + 6Td·d_ff`**

   Substituting values:
   - `8Td² = 8 × 1024 × 1600² ≈ 20.97G FLOPs`
   - `4T²d = 4 × 1024² × 1600 ≈ 6.71G FLOPs`
   - `6Td·d_ff = 6 × 1024 × 1600 × 6400 ≈ 62.91G FLOPs`
   - *Per block ≈ 90.6G FLOPs*

   *48 blocks:* `48 × 90.6G ≈ 4,350G FLOPs`

   *lm_head:* `2 × T × d × V = 2 × 1024 × 1600 × 50257 ≈ 164.9G FLOPs`

   **Total ≈ 4,515G ≈ 4.5 × 10¹² FLOPs**

3. Most Expensive Components

   **Answer**: The FFN layers (W1/W2/W3) dominate, accounting for roughly **69%** of per-block FLOPs. This is because d_ff = 4×d_model makes the FFN matrix multiplies significantly larger than the attention projections. Attention score computation (QKᵀ and AV) is relatively cheap at the default context length.

4. Repeat your analysis with
   - GPT-2 small (12 layers, 768 d_model, 12 heads),
   - GPT-2 medium (24 layers, 1024 d_model, 16 heads),
   - GPT-2 large (36 layers, 1280 d_model, 20 heads).

   As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

   **Answer**: Using d_ff = 4×d_model throughout:

   | Model  | L   | d    | Total FLOPs | FFN | QKV proj | Attn scores | lm_head |
   | ------ | --- | ---- | ----------- | --- | -------- | ----------- | ------- |
   | Small  | 12  | 768  | ~176G       | 67% | 14%      | 8%          | 11%     |
   | Medium | 24  | 1024 | ~627G       | 67% | 14%      | 7%          | 12%     |
   | Large  | 36  | 1280 | ~1,613G     | 67% | 14%      | 6%          | 13%     |
   | XL     | 48  | 1600 | ~4,515G     | 68% | 15%      | 6%          | 4%      |

   1. **FFN** and **QKV** projections maintain a stable share across model sizes since both scale as O(Td²L).
   2. **Attention score** computation scales as O(T²dL) and therefore shrinks proportionally as d grows.
   3. The **lm_head** share fluctuates based on the ratio of V to d×L, and becomes relatively less significant at larger model sizes.

5. Extending Context Length to 16,384

   Attention score computation scales as O(T²), so increasing T by 16× inflates that component by **256×**, while all other operations scale linearly with T (16×).

   - **Attention scores:** `4 × 16384² × 1600 × 48 ≈ 82,200G FLOPs`
   - **Everything else (linear in T):** `(4,515 - 270) × 16 ≈ 67,920G FLOPs`
   - **New total ≈ 150,000G ≈ 1.5 × 10¹⁴ FLOPs**, roughly **33×** the original.

   Attention score computation jumps from ~6% to ~55% of total FLOPs, becoming the dominant bottleneck.
   This is precisely why methods like FlashAttention and sparse attention are critical for long-context inference.

## Training a Transformer LM
### Cross Entropy
### AdamW
#### Resource accounting
(a) Peak Memory
- Parameters:
   - Embedding：$V \times D$
   - Transformer：self attention $4D^2$ and MLP $2 \times D \times 4D = 8D^2$. Total $L \times 12D^2$.
   - $M_{params} = 4 \times (V \times D + 12 \times L \times D^2)$ bytes
- Gradients:same size as parameters
   - $M_{grads} = 4 \times (V \times D + 12 \times L \times D^2)$ bytes
- Optimizer State:
   - AdamW preserves two states：Momentum & Variance
   - $M_{opt} = 2 \times M_{params} = 8 \times (V \times D + 12 \times L \times D^2)$ bytes
- Activations: most complex part，depends on $B \times C$
   Each layer：
   - RMSNorm: $B \times C \times D$
   - QKV Projections: $3 \times B \times C \times D$
   - Attention ($Q K^T$): $B \times num\_heads \times C^2$
   - Softmax/Weighted Sum: $B \times num\_heads \times C^2$
   - MLP ($W_1, SiLU, W_2$): $B \times C \times (4D + 4D) = 8 \times B \times C \times D$
   - $M_{acts} = 4 \times B \times C \times (L \times (12D + 2 \times num\_heads \times C) + V)$ bytes
- $M_{total} = M_{params} + M_{grads} + M_{opt} + M_{acts}$

(b) GPT-2 XL
- $L=48, D=1600, V=50257, C=1024, num\_heads=25$
- $M_{static} = M_{params} + M_{grads} + M_{opt} = 4 \times M_{params}$
- $M_{params} \approx 1.55 \times 10^9$ (1.55B)
- $M_{static} \approx 4 \times 1.55 \times 4 \text{ bytes} \approx 24.8 \text{ GB}$
- $M_{acts} \approx B \times 1.1 \text{ GB}$ (具体数值取决于层数和序列长度的乘积)。
- 结果与最大 Batch Size:表达式示例：$1.1 \times B + 24.8 \leq 80 \text{ GB}$。解得 $B \approx 50$ 左右

(c) 单步 FLOPs：
- Forward：对于每个 token，每个参数约进行 2 次浮点运算（乘加）
   - $FLOPs_{fwd} \approx 2 \times N \times B \times C$（$N$ 为参数量）。
- Backward：需要计算输入梯度和权重梯度，计算量翻倍。
   - $FLOPs_{bwd} \approx 4 \times N \times B \times C$。
- 总 $FLOPs \approx 6 \times N \times B \times C$

(d) 训练时间估算
- 总任务 FLOPs:
   - $N = 1.55 \times 10^9$, $B = 1024$, $C = 1024$, $Steps = 400,000$。
   - $Total\_FLOPs = 6 \times N \times (B \times C \times Steps) \approx 6 \times 1.55 \times 10^9 \times 4.19 \times 10^{11} \approx 3.9 \times 10^{21}$。
- 有效计算速度:
   - A100 理论峰值 = 19.5 TFLOPS。
   - 50% MFU 实际速度 = $19.5 \times 0.5 = 9.75 \text{ TFLOPS} = 9.75 \times 10^{12} \text{ FLOPS}$。
- 时间计算:
   - $Time = \frac{Total\_FLOPs}{Actual\_Speed} = \frac{3.9 \times 10^{21}}{9.75 \times 10^{12}} \approx 400,000,000 \text{ 秒}$。
   - 转换为天数：$400,000,000 / 86400 \approx 4630 \text{ 天}$。结论：在单卡 A100 上训练 GPT-2 XL 级别的大模型需要极其漫长的时间（超过 12 年），这说明了分布式训练（多卡/集群）的必要性。


## Experiments
### Tuning the learning rate
- 搜索策略：先做粗粒度的对数尺度扫描，找到大概范围：
  - 第一轮：`1e-4, 3e-4, 1e-3, 3e-3, 1e-2`
- 根据结果找到最优区间，再细搜：
  - 第二轮：`比如 1e-3 最好，就在 5e-4, 8e-4, 1e-3, 2e-3, 3e-3 之间细搜`

- "edge of stability" 的含义：
最优学习率往往就在 loss 开始发散的临界点附近，太小收敛慢，太大直接发散。







--
-
-
-
-
-
-
-
-
-
-
-
--
-

-
-
-
-
