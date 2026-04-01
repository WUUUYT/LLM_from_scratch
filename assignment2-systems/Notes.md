# Systems Basics
## Profiling and Benchmarking
### Model size
Measure first, then optimize.
1. Python standard library to time forward and backward pass.
2. Profile compute with the NVIDIA Nsight Systems tool.
3. Profile memory usage.

Fix vocab size to 10000, batch size to 4.
| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 1600    | 6400  | 48         | 25        |
| 2.7B   | 2560    | 10240 | 32         | 32        |

### End-to-End Benchmarking: `timeit`
#### Results
**Forward only (warmup = 5, context_length=128)**
| size   |   params (M) | forward_only   |   mean_ms |   stdev_ms |
|:-------|-------------:|:---------------|----------:|-----------:|
| small  |      128.625 | True           |     26.83 |       0.65 |
| medium |      423.183 | True           |     78.64 |       0.22 |
| large  |      969.412 | True           |    170.47 |       0.18 |
| xl     |      1998.24 | True           |    328.06 |       1.18 |
| 2.7B   |      3406.81 | True           |    514.75 |       1.39 |


**Forward and Backward (warmup = 5, context_length=128)**
| size   |   params (M) | forward_only   |   mean_ms |   stdev_ms |
|:-------|-------------:|:---------------|----------:|-----------:|
| small  |      128.625 | False          |     83.07 |       1.28 |
| medium |      423.183 | False          |    247.43 |       0.44 |
| large  |      969.412 | False          |    525.73 |        0.7 |
| xl     |      1998.24 | False          |   1024.57 |       2.11 |

**Forward only (warmup = 0, fcontext_length=128)**
| size   |   params (M) | forward_only   |   mean_ms |   stdev_ms |
|:-------|-------------:|:---------------|----------:|-----------:|
| small  |      128.625 | True           |     47.45 |      65.44 |
| medium |      423.183 | True           |     98.01 |      60.52 |
| large  |      969.412 | True           |     190.5 |         66 |
| xl     |      1998.24 | True           |     348.5 |      63.21 |
| 2.7B   |      3406.81 | True           |    533.85 |      62.11 |

#### Analysis
**Forward Pass Latency**: Scales roughly linearly with model size, from 26.83ms (small) to 514.75ms (2.7B).
**Backward Pass Latency**: Backward pass takes approximately 3× longer than the forward pass. This matches theoretical expectations — the backward pass must recompute gradients through every layer, requiring roughly 2× the work of the forward pass, totaling ~3× end-to-end.
**Variability**: Standard deviation is consistently below 2ms across all configurations, representing less than 1% of the mean. Measurements are highly stable, confirming that the warm-up steps successfully eliminated initialization noise.

**Effect of Warm-up Steps**
Without warm-up, the mean is slightly higher and the standard deviation explodes. The first few steps are dramatically slower than the rest.

**The first forward pass is always slow** because:
- CUDA kernel compilation — PyTorch compiles and caches CUDA kernels on first use. This one-time cost inflates the first measurement significantly.
- Memory allocation — GPU memory for activations and weights is allocated and paged in on the first run, adding overhead that disappears in subsequent steps.
- CPU/GPU cache cold start — Caches are empty on the first pass; subsequent passes benefit from warm caches.

**1-2 warm-up steps may not be enough** because:
- CUDA kernel compilation can span multiple steps — some kernels are only triggered in specific layers, so a single pass may not compile all of them.
- Memory allocator behavior — PyTorch's caching allocator may still be adjusting its allocation strategy after just 1-2 passes.
- CPU-side JIT overhead — Python-level overhead (e.g., autograd graph construction) also takes a few steps to stabilize.

A safe default is 5+ warm-up steps to ensure all one-time costs are fully absorbed before timing begins.

### Nsight Systems Profiler
`nsys` can do running time analysis for functions and CUDA kernels which are executed asynchronously on the GPU. For use, just add `nsys profile -o result`:
`uv run nsys profile -o result python benchmark.py`
It will write outputs to `result.nsys.rep`.

We can then view the profile on your local machine with the *NVIDIA Nsight Systems desktop application*. Selecting a particular CUDA API call (on the CPU) in the CUDA API row of the profile will highlight all corresponding kernel executions (on the GPU) in the CUDA HW row. Possible usages are:
1. Get Python backtraces for each CUDA API call with `--python-backtrace=cuda`, may introduce overhead.
2. NVTX (NVIDIA Tools Extension): Annotate a range of codes, thus in visualization, they will appear as blocks in the NVTX row. (Ignore warmup steps)
- It could isolate kernels for different parts:
    ```python
    import torch.cuda.nvtx as nvtx

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(... # Q, K, V, mask)
        ...
        with nvtx.range("computing attention scores"):
            ... # compute attention scores between Q and K
        with nvtx.range("computing softmax")
            ... # compute softmax of attention scores
        with nvtx.range("final matmul")
            ... # compute output projection
        return ...
    ```
    - Decorater: `@nvtx.range("...")`
    - Context Manager: `with nvtx.range("..."):`
- `cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention` swap the implementation with annotated version.
  - No need to adjust source code.
- `--pytorch` command-line option with `nsys` to annotate calls to PyTorch C++ API to do lower level analysis.

#### Steps
Run on clusters with:
`uv run nsys profile -o profiles/small_128 --pytorch python nsys_profiler.py --size small --context_length 128`
On local terminal:
`scp username@login-ice.pace.gatech.edu:~/path/to/profiles/small_128.nsys-rep ~/LocalDocuments/Courses/CS336_LLM_from_scratch/assignment2-systems/profiles`
Then open the file in *NVIDIA Nsight Systems desktop application*.

#### Results (TODO)

#### Analysis (TODO)

### Mixed Precision
Earlier `torch.float32` is used. But at tensor cores (NVIDIA DPUs) can achieve >16$\times$ performance with FP16 or BF16.
1. To overcome Underflow, use **loss scaling**: the loss is simply multiplied by a scaling factor, increasing gradient magnitudes so they don’t flush to zero. The result will be divided by the factor before update.
2. For Overflow (NaN), use bfloat16 (same dynamic range as FP32).

Mixed Precision: `torch.autocast` context manager. It automatically identify which operations to perform in lower-precision.

#### `mixed_precision_accumulation`
- The FP32 accumulator (case 1) gives the most accurate result of 10.0001, very close to the true value of 10.0.
- Pure FP16 accumulation (case 2) is the least accurate at 9.9531, because FP16 has limited precision and rounding errors compound over 1000 additions.
- Interestingly, cases 3 and 4 both yield 10.0021 — whether you accumulate FP16 values directly into a FP32 accumulator or explicitly cast each value to FP32 first, the result is the same and significantly better than pure FP16, which demonstrates why mixed precision keeps accumulators in FP32 even when the operands are in lower precision.

**Keep accumulations in higher precision even if the tensors themselves being accumulated have been downcasted.**

#### Data types under FP16 Autocast
| Component | Data Type |
|-----------|-----------|
| Model parameters (stored) | `torch.float32` |
| Output of `fc1` (Linear) | `torch.float16` |
| Output of `ln` (LayerNorm) | `torch.float32` |
| Predicted logits (output of `fc2`) | `torch.float16` |
| Loss | `torch.float32` |
| Gradients | `torch.float32` |

- Matrix multiplications (Linear layers) -> FP16
- LayerNorm -> FP32.
- Parameters are never modified in storage — only temporarily cast during computation.
- Gradients and loss are always accumulated in FP32 regardless of the autocast dtype.

**Why layernorm treated differently**
LayerNorm internally computes the mean and variance of its inputs,
$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2,\quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$both of which are **summation-based reductions**.
If switched from FP16 to BF16, the dynamic range issue (overflow/underflow) is resolved, but reduction accuracy remains a concern.
In practice, PyTorch's autocast keeps LayerNorm in FP32 for both FP16 and BF16 to avoid this precision loss.

#### Results and Analysis (TODO)



### Profiling Memory
PyTorch has a powerful memory profiler which can keep track of allocations over time. To use:
```python
... # warm-up phase in your benchmarking script
# Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
... # what you want to profile in your benchmarking script
# Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)
```
1. Download `memory_snapshot.pickle`.
2. visit https://pytorch.org/memory_viz.
3. Drag pickle to it.

---

## FlashAttention-2
**Problem**
Standard attention computes:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right)V$$Memory cost:
```
seq_len × seq_len × batch_size × num_heads × bytes_per_element
```
For long sequences (e.g., 4096, 8192 tokens), this matrix becomes **enormous**, causing out-of-memory (OOM) errors.
**Solution**
FlashAttention-2 computes attention **by tiles** — splitting Q, K, V into small blocks and processing them one block at a time, never storing the full `seq_len × seq_len` matrix.

| Property | Standard Attention | FlashAttention-2 |
|----------|--------------------|-----------------|
| Peak memory | O(seq_len²) | O(seq_len) |
| OOM on long sequences | Yes | No |
| Numerical result | Exact | **Identical**|
| Speed | Baseline | Faster (better memory access patterns) |

- The speedup and memory savings come purely from a smarter execution order that avoids unnecessary reads and writes to slow HBM (high-bandwidth memory).

### PyTorch Attention

#### Results and Analysis (TODO)

### JIT-Compiled Attention
Just-In-Time compiler works at the first runtime. it automatically generate fused Triton kernels by dynamically analyzing your computation graph.



### Triton
Write more specific optimized GPU kernel using python

1. Triton decomposes computation into **Program Instances**. Each instance represents a block of threads running in parallel on the GPU. Work is distributed using a Program ID (pid); for example, instance i is responsible for processing the i-th tile of data.
2. Triton processes data in small, rectangular sub-sections called **Tiles**.
   - Pointers and Strides: Triton accesses memory using the base pointer of a tensor and strides (which define the memory jump required to move along specific axes).
   - `tl.make_block_ptr`: It creates a "virtual window" over a global tensor by defining its shape, strides, and the desired tile size.
3. `.advance`: Once a block pointer is initialized, you can navigate the tensor grid using the `.advance` method. As shown in the tiling schematics:
   - `.advance((ROWS_TILE, 0))` shifts the window down to the next set of rows.
   - `.advance((0, D_TILE))` shifts the window to the right to process the next segment of features.
4. Explicit Memory Management (Load/Store)
    - `tl.load`: You must manually pull data from High Bandwidth Memory (HBM) into fast on-chip SRAM before performing calculations.
    - `tl.store`: After the computation, the result must be explicitly written back from SRAM to the designated memory address in the output tensor.

#### weighted_sum_fwd
**Forward pass**
```python
import triton
import triton.language as tl
@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # input pointer
    output_ptr, # outputpoint
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim, # Likely 1
    output_stride_row, # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):
    # `tl.program_id` check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0), # Order of the dimensions in memory from major to minor
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),  # likely 1
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),  # likely 1
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))
```
To wrap it up in PyTorch Autograd function:
- `torch.autograd.Function` can self-define forward and backward，and connect to autograd. `ctx` is a context manager to pass information between forward and backward.
```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d") # -> 2D

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16 # store in ctx to reuse in backward
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device) # faster than torch.zeros
        n_rows = y.numel()

        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            # Lauch grid: show num of program instances to start
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])
```

**Backward pass**

$$(\nabla_x \mathcal{L})_{ij} = \sum_{k=1}^{n} \frac{\partial f(x,w)_k}{\partial x_{ij}} (\nabla_{f(x,w)}\mathcal{L})_k=w_j\cdot (\nabla_{f(x, w)}\mathcal{L})_i$$$$\Rightarrow \nabla_x\mathcal{L}=\nabla_{f(x, w)}\mathcal{L}\otimes w\quad \text{(Outer Product)}$$

$$(\nabla_w \mathcal{L})_j = \sum_{i=1}^{n} \frac{\partial f(x,w)_i}{\partial w_j} (\nabla_{f(x,w)}\mathcal{L})_i= \sum_{i=1}^{n} x_{ij} \cdot (\nabla_{f(x,w)}\mathcal{L})_i$$

```python
@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr, # input
    grad_output_ptr,  #grad input [NUM_ROWS]
    grad_x_ptr, partial_grad_weight_ptr,  # Outputs：∇x and partial ∇w
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles  = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0,), padding_option="zero"
        )
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,), padding_option="zero"
        )
        grad_x_row = grad_output[:, None] * weight[None, :]
        # [ROWS_TILE_SIZE, 1]*[1, D_TILE_SIZE]
        # => [ROWS_TILE_SIZE, D_TILE_SIZE]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1), padding_option="zero"
        )  # [ROWS_TILE_SIZE, D_TILE_SIZE]
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(
            partial_grad_weight_block_ptr, grad_weight_row,
            boundary_check=(1,)   # dim 0 永远不越界（只有1行），只检查 dim 1
        )

        x_block_ptr                   = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr              = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr              = grad_x_block_ptr.advance((0, D_TILE_SIZE))
```

- $\nabla w$ cannot be reduced directly inside the kernel (race condition across instances), so each instance writes its partial row-sum to a separate row of partial_grad_weight [n_row_tiles, D], which is then collapsed to [D] outside the kernel via a single `partial_grad_weight.sum(dim=0)`.

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty(
            (tl.cdiv(x.shape[0], ctx.ROWS_TILE_SIZE), x.shape[1]),
            device=x.device
        )

        # 启动 backward kernel
        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(dim=0)
        # [n_row_tiles, D] → [D]
        grad_x = grad_x.view(ctx.input_shape)

        return grad_x, grad_weight
```

Obtain a function like those implemented in `torch.nn.functional`: `f_weightedsum=WeightedSumFunc.apply`. Call it on $x$ and $w$ give tensor result:
`tensor([ 90.8563, -93.6815, -80.8884, ..., 103.4840],
    device='cuda:0', grad_fn=<WeightedSumFuncBackward>)`
PyTorch knows what to call in backward pass.




### FlashAttention-2 forward pass
$$\mathbf{S} = \mathbf{QK}^\top/\sqrt{d}; \quad \mathbf{P} = \text{softmax}(\mathbf{S}); \quad \mathbf{O} = \mathbf{PV}$$Backward needs $\mathbf{P}$, which causes large memory and IO costs.

The main goal is to avoid reading and writing the attention matrix to and from HBM, to reduce IO and peak memory costs. Three techniques:
1. **Tiling**: Split the input into tiles and make several passes over input tiles, thus incrementally performing the softmax reduction.
2. **Recomputation**: Save “activation checkpoints” in HBM, then recompute part of the forward pass to get other activations we need for computing gradients.
   - FlashAttention-2 stores the logsumexp of the attention scores, L: $$L_i=\log(\sum_j\exp(S_{ij}))$$
   - Compute this in an online manner.
3. Operator fusion: Performing all our operations in a single kernel to avoid repeated memory IO. Operator fusion is partly enabled by recomputation.

#### Backward pass with recomputation
- Since compute speed is faster than HBM IO, so recompute $P$.
- $L$ is calculated in the forward pass, and saved.
- $D$ (a vector) is calculated at the start of backward pass: $D=\text{row-sum}(O\circ dO)$.
  - $PdP^T = P(dOV^T)^T = PV(dO)^T = OdO^T$

With $L$ and $D$, backward pass can be computed without softmax.
$$\begin{align*}
    &D =\text{row-sum}(O\circ dO)\\
    \text{Reconstruct Attention}:\quad &
    S = QK^T / \sqrt{d};\; P_{ij} = \exp(S_{ij} - L_i)\\
    \text{Grads for }V:\quad &
    dV = P^T dO\\
    \text{Grads for softmax}:\quad &
    dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)\\
    & \text{where}\quad dP = dO V^T\\
    \text{Grads for }Q,K:\quad &dQ = dSK / \sqrt{d};\quad dK = dS^T Q / \sqrt{d}
\end{align*}$$

> **If $O = PV$, then $dV = P^T dO$** and ($dP=dOV^T$)
> *Proof*: Write $O_{ik} = \sum_{j} P_{ij} V_{jk}$. Consider $V_{mn}$'s contribution to $O_{ik}$:$$dV_{mn} = \frac{\partial L}{\partial V_{mn}} = \sum_{i} \sum_{k} \frac{\partial L}{\partial O_{ik}} \frac{\partial O_{ik}}{\partial V_{mn}}$$Only when $j=m$ and $k=n$, partial derivative is not 0：$$\frac{\partial O_{ik}}{\partial V_{mn}} = P_{im} \quad (\text{when } k=n)$$$$\Rightarrow dV_{mn} = \sum_{i} dO_{in} \cdot P_{im}$$$$\Rightarrow dV_{mn} = \sum_{i} (P^T)_{mi} \cdot dO_{in}$$
> **Softmax gradient**
> For a score vector, $P_i = \frac{e^{S_i}}{\sum_j e^{S_j}}$. To compute $\frac{\partial P_i}{\partial S_j}$ with $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$.
> If $i = j$, $\frac{\partial P_i}{\partial S_i} = \frac{e^{S_i} (\sum e^{S_k}) - e^{S_i} (e^{S_i})}{(\sum e^{S_k})^2} = \frac{e^{S_i}}{\sum e^{S_k}} - \left(\frac{e^{S_i}}{\sum e^{S_k}}\right)^2 = P_i - P_i^2 = P_i(1 - P_i)$
> If $i \neq j$, $\frac{\partial P_i}{\partial S_j} = \frac{0 \cdot (\sum e^{S_k}) - e^{S_i} (e^{S_j})}{(\sum e^{S_k})^2} = - \frac{e^{S_i}}{\sum e^{S_k}} \cdot \frac{e^{S_j}}{\sum e^{S_k}} = -P_i P_j$
> This can be written in a matrix: $J = \text{diag}(P) - P P^T$. Given $dP$, want $dS$: $$dS = J^T \cdot dP=J\cdot dP=(\text{diag}(P) - P P^T) dP$$$$\Rightarrow dS = \text{diag}(P) dP - (P P^T) dP= P \circ dP - P \cdot D = P \circ (dP - D)$$$$\Rightarrow dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$$


### Algorithm: FA2 forward pass
**Running Variables**: For each query tile $i$, we maintain across key tiles $j$:

| Variable | Shape | Meaning |
|----------|-------|---------|
| $m_i^{(j)}$ | $[B_q]$ | Running row-wise **maximum** of $S$ seen so far (for numerical stability) |
| $l_i^{(j)}$ | $[B_q]$ | Running **proxy for the softmax denominator** |
| $\mathbf{O}_i^{(j)}$ | $[B_q, d]$ | Running **unnormalized output** accumulator |

**Algorithm**
![alt text](images/FA2.png)

> $O_i = \frac{\sum_k \exp(S_{ik}) \cdot V_k}{\sum_k \exp(S_{ik})}$. Both **Numerator** $\tilde{O}$ and **Denominator** $l$ are accumulated tile by tile.
> **Final division** is done once at the end.

> If $m$ increases , all previously computed $\exp(S - m_\text{old})$ values should be multiply old results by $\exp(m_\text{old} - m_\text{new}) \leq 1$ to rescale them to the new baseline.
> New:  `exp(S - m_new) = exp(S - m_old) × exp(m_old - m_new)`
> This applies to **both** $l$ and $O$:
> - `l^(j) = exp(m_old - m_new) * l^(j-1) + rowsum(P̃^(j))`
> - `O^(j) = exp(m_old - m_new)[:, None] * O^(j-1) + P̃^(j) @ V^(j)`



# Distributed Data Parallel Training
## A quick recap about Distributed Computing
**Process**
A running instance of a program. Processes are completely isolated by default — variables in one process are invisible to another. In distributed training, each GPU typically runs its own process.

**Worker**: an instance of a program that’s participating in the distributed training. A worker may use multiple processes (e.g., to load data for training).

**Node**
A physical machine (server). One node can run multiple processes. A single-node job runs all processes on the same machine; a multi-node job spreads them across multiple servers.

**Local Rank**
Index of a local worker on the machine
**Global Rank**
Index of a worker in the process group (across all nodes).

**IP Address and Port**
An IP address identifies a machine on a network. A port is a number that identifies a specific program running on that machine. Together they form a unique communication endpoint, like a street address plus a room number.

**localhost**
A special address that always refers to the current machine itself. Used as the master address when all processes run on the same machine.

**Master Node / Rendezvous Point**
When processes start up, they need a known address to check in with so they can find each other. In PyTorch this is set via `MASTER_ADDR` and `MASTER_PORT`. Once all processes have checked in, communication can begin.

**Rank and World Size**
Rank is the unique integer ID of a process (starting from 0). World size is the total number of processes. Every process knows its own rank and uses it to determine what data to work on and where to send results.

**Communication Backend**
The underlying engine that actually moves data between processes. `nccl` is used for GPU-to-GPU communication and is the standard choice for training. `gloo` works on CPU and is useful for debugging.

**Parallel compute**
1. **all_reduce**: A collective operation where every process contributes a tensor, the tensors are summed, and every process receives the same result.
2. **broadcast**: One process sends its tensor to all other processes.
3. **all_gather**: Every process contributes a tensor, and every process ends up with all of them concatenated.
4. **reduce_scatter**: Aggregates tensors across ranks and then distributes a different chunk of the result to each rank.

**Synchronous vs Asynchronous Communication**
- Synchronous: block the process until communication is complete.
- Asynchronous operations return immediately and let the process continue computing, with an explicit wait later. Async communication enables overlapping computation and communication, which is a key performance optimization in DDP.

**Data Parallel Training (DDP)**
Each GPU processes a different mini-batch, computes its own gradients, and then uses all_reduce to sum gradients across all GPUs.

**What You Don't Need to Know**
TCP/IP protocol internals, socket programming, NCCL's ring-allreduce implementation, OS-level process scheduling, and InfiniBand or NVLink hardware details. The PyTorch distributed API abstracts all of this away.

### Single-node

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # rendezvous address (same machine)
    os.environ["MASTER_PORT"] = "29500"      # rendezvous port (any free port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # "gloo" = CPU backend; use "nccl" for GPU
    # blocks until all world_size processes have checked in

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} before: {data}")
    dist.all_reduce(data, async_op=False)
    # sum tensors across all ranks, result written back in-place on every rank
    # async_op=False = block until this operation is done (not the same as join=True)
    print(f"rank {rank} after:  {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(
        fn=distributed_demo,
        args=(world_size,),   # extra args beyond rank, which is prepended automatically
        nprocs=world_size,    # spawn this many child processes with rank=0,1,2,3
        join=True             # main process waits here until all children finish
    )
    # main process never calls setup() or all_reduce() — it only spawns and waits
```
- `mp.spawn` automatically use rank as the first parameter when calling `fn`, followed by `*args`.
  - Thus, `rank` must be the 1st param in `fn`.
- `MASTER` is the process with rank 0.
- When running multi-GPU jobs, make sure that different ranks use different GPUs. (if not stated, all use `cuda:0`)
    ```python
    def setup(rank, world_size):
        torch.cuda.set_device(rank)   # all .to("cuda") use cuda:rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    ```
    ```python
    device = f"cuda:{rank}"
    tensor.to(device)  # explicit specify
    ```

### Practices for benchmarking distributed compute
1. Run benchmarks on the same machine (if possible) to facilitate controlled comparisons.
2. Perform several warm-up steps before timing the operation of interest.
   - Important for `NCCL` communication calls. 5 iterations of warmup is generally suﬀicient.
3. Call `torch.cuda.synchronize()` to wait for CUDA operations to complete.
   - Necessary even when `async_op=False` (doesn't mean the communication is completed, returns if queued on the GPU).
   - `synchronize() -> time -> operation -> synchronize() -> time`
4. Aggregate measurements across ranks to improve estimates.
   -  `all-gather`(`dist.all_gather_object`) collective is useful for collecting results from all ranks.
   - `synchronize() -> time -> operation -> synchronize() -> time`
5. Debug locally with `Gloo` on CPU, and benchmark with `NCCL` on GPU. Switching between the backends just involves changing the `init_process_group` call and tensor device casts.


## Naive Implementation of DDPT
A single GPU has limited memory. With $d$ GPUs each handling a batch of size $n/d$, the effective batch size becomes $n$, scaling linearly with the number of devices.

**Initialization**
Each GPU randomly initializes its own model. To ensure all replicas start identically, `broadcast` sends rank 0's parameters to all other ranks. All GPUs now hold the same parameters and optimizer states.

**Training Loop**
1. **Shard the batch**. Split the global batch of $n$ examples into $d$ equal chunks. Each GPU receives $n/d$ disjoint examples. $n$ must be divisible by $d$ since training speed is bottlenecked by the slowest process.
2. **Local forward and backward**. Each GPU independently runs a forward and backward pass on its local shard. The resulting gradients reflect only the $n/d$ examples that GPU saw.
3. **All-reduce gradients**. Average gradients across all GPUs with `all_reduce`. After this, every GPU holds the same gradient — mathematically equivalent to computing it on the full batch of $n$ examples.
4. **Optimizer step**. Each GPU updates its local parameters using the averaged gradient. No additional synchronization needed.


## Improvement
**Problem 1 — Too many communication calls**
Naive DDP calls all_reduce once per parameter tensor. Each call has a fixed startup overhead regardless of data size.
**Problem 2 — Communication waits for the full backward pass**
Naive DDP only starts communicating after the entire backward pass finishes, leaving the network idle while the GPU computes gradients. Since backward proceeds layer by layer (output → input), each parameter's gradient is ready before the full backward completes.

### Reduce Communication Calls
Flatten all parameter gradients into a single 1-D tensor, issue **one** `all_reduce`, then unflatten back into each parameter's `.grad`. Total data transferred is identical — only the number of calls changes.

```python
# Flatten
flat = torch._utils._flatten_dense_tensors(grads)

# Single all_reduce
dist.all_reduce(flat, op=dist.ReduceOp.SUM)
flat /= world_size

# Unflatten back into each parameter's .grad
for param, new_grad in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
    param.grad.copy_(new_grad)
```

### Overlapping Backward with Gradient Communication
**`register_post_accumulate_grad_hook`** — fires automatically on a parameter as soon as its gradient is fully accumulated during backward. No need to poll or wait for the full backward.

**`async_op=True`** — returns a handle immediately without blocking, so the backward computation continues while the all-reduce runs in the background. Use `handle.wait()` before `optimizer.step()` to ensure all communications are queued.

```python
handles = []

def hook(param):
    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
    handles.append(handle)

for param in model.parameters():
    param.register_post_accumulate_grad_hook(hook)

# After backward:
for handle in handles:
    handle.wait()   # ensure all all-reduces are queued before optimizer step
handles.clear()

optimizer.step()
```

`wait()` guarantees the operation has been **queued on the GPU**, not that it has completed. Subsequent GPU operations that depend on the result (like `optimizer.step()`) will automatically wait in the GPU queue, so this is safe.


### Bucketed overlap
Group parameters into K buckets. As soon as all gradients within a bucket are ready, immediately all-reduce that bucket.
This reduces communication call overhead while preserving overlap.

**Mechanism**
Each bucket tracks a **pending counter** (number of gradients not yet computed). A post-accumulate-grad hook decrements the counter for each parameter. When a bucket's counter reaches zero, all its gradients are flattened and all-reduced asynchronously.

```
Backward (output → input):
  param_n ready → bucket 2 counter: 3 → 2 → 1 → 0 → all_reduce(bucket 2)
  (backward continues computing bucket 1 params while bucket 2 communicates)
  param_m ready → bucket 1 counter: 3 → 2 → 1 → 0 → all_reduce(bucket 1)
  ...
```
- **Bucket order**: assign parameters to buckets in **reverse backward order** so the first bucket to be ready is all-reduced first, maximizing overlap.
- **Bucket size**: larger buckets → fewer calls but longer wait per bucket. PyTorch's default is 25MB per bucket.
- **finish_gradient_synchronization()**: wait for all bucket handles before `optimizer.step()`, same as before.
- `torch.nn.parallel.DistributedDataParallel` uses exactly this approach internally.

## Parallelism Strategies
**Data Parallelism (DP).** Each GPU holds a full model copy and processes a different data shard. Gradients are averaged via all_reduce. Solves compute throughput but not memory — the full model must fit on each GPU.

**Fully-Sharded Data Parallelism (FSDP).** Parameters, gradients, and optimizer states are sharded across GPUs. Each GPU stores only its shard. Before forward/backward, weights are temporarily reconstructed via all_gather, then discarded. Solves the memory problem for large models.

**Tensor Parallelism (TP).** A single layer's computation is split across GPUs — each GPU computes results for a slice of the activations or weights. Requires an all_reduce to combine partial results. Typically combined with FSDP, sharding weights and activations along matching dimensions.

**Pipeline Parallelism (PP).** Model layers are split across GPUs in stages. Data flows through stages like an assembly line, with multiple micro-batches overlapping to reduce idle time (pipeline bubbles). Useful when the model has too many layers to fit on one GPU.

**Expert Parallelism (EP).** In Mixture-of-Experts models, different experts are placed on different GPUs. Not discussed further here (focus is on dense models).

### The 4D View
FSDP and TP are typically combined (both involve weight sharding along compatible dimensions), so they count as one axis. This leaves **three practical axes: DP, FSDP/TP, PP and EP**.

## Device Mesh
GPUs are organized into a multi-dimensional grid where each axis corresponds to one parallelism strategy.

```
16 GPUs → 4×4 mesh:

         FSDP/TP dimension →
DP   [ 0,  1,  2,  3 ]
dim  [ 4,  5,  6,  7 ]
↓    [ 8,  9, 10, 11 ]
     [12, 13, 14, 15 ]

Row  [0,1,2,3]   → one FSDP/TP group (model sharded across 4 GPUs)
Col  [0,4,8,12]  → one DP group (gradient all_reduce across 4 replicas)
```

## Communication Accounting — XXL Model


- d_model = 16384, d_ff = 53248, num_blocks = 126
- Each block: 2 linear layers (d_model→d_ff, d_ff→d_model), no attention
- **Total parameters:** P = 2 × d_model × d_ff × num_blocks ≈ **219.85B**

### (a) Single Device Memory

| Component | Dtype | Size |
|-----------|-------|------|
| Master weights | FP32 | P × 4 = 879.4 GB |
| Accumulated gradients | FP32 | P × 4 = 879.4 GB |
| Optimizer states (m, v) | FP32 | P × 8 = 1,758.8 GB |
| **Total** | | **3,517.6 GB ≈ 3.52 TB** |

→ Requires **44 H100 80GB GPUs** on a single device.

**Activations saved for backward (BF16):**
$$\text{Act} = B \times T \times (d_\text{model} + d_\text{ff}) \times 2\ \text{bytes} \times \text{num\_blocks} = B \times T \times 16.74\ \text{MB}$$

### (b) FSDP Memory per Device

$$\text{Memory/device} = \frac{P \times 16\ \text{bytes}}{N_\text{FSDP}} + \frac{B \times T \times 16.74\ \text{MB}}{2}$$

For total < 95 GB (1 v5p TPU), ignoring activations:
$$N_\text{FSDP} > \frac{3517.6}{95} \approx 37 \implies \boxed{N_\text{FSDP} \geq 38}$$

(Round to next power of 2: $N_\text{FSDP} = 64$)

### (c) Compute-Bound Condition

**Setup:** X=16 (FSDP), Y=4 (TP), M_X=2, M_Y=1, W_ICI = 1.8×10¹¹ bytes/s, C = 4.6×10¹⁴ FLOPS/s

**Compute time per block per device:**
$$T_\text{compute} = \frac{4 \times B \times T \times d_\text{model} \times d_\text{ff}}{Y \times C}$$

**FSDP all-gather time per block:**
$$T_\text{FSDP} = \frac{4 \times d_\text{model} \times d_\text{ff}}{W_\text{ICI}}$$

**Compute-bound when** $T_\text{compute} > T_\text{FSDP}$:

$$\boxed{B \times T > \frac{Y \times C}{W_\text{ICI}} = \frac{4 \times 4.6 \times 10^{14}}{1.8 \times 10^{11}} \approx 10{,}222\ \text{tokens/device}}$$

At T=2048: B_per_device ≥ 5 sequences.

---

## (d) Reducing Batch Size While Retaining Throughput

The threshold $B \times T > YC/W_\text{ICI}$ shows minimum batch is set by the compute-to-bandwidth ratio. Three main strategies reduce this:

**Pipeline Parallelism (PP):** Splits model depth across stages. Each stage has fewer parameters ($P/\text{stages}$), so the all-gather communication per stage is smaller, and compute-bound can be achieved with smaller micro-batches. Multiple micro-batches are pipelined to keep all stages busy.

**Sequence Parallelism (SP):** Shards activations along the sequence dimension within a TP group. Reduces memory per device, allowing longer $T$ with smaller $B$ while keeping $B \times T$ above the threshold.

**Gradient Accumulation:** Run $k$ small compute-bound micro-steps before each optimizer update. Each micro-step individually satisfies the compute-bound condition, while the effective optimization batch is $k \times B$, allowing smaller per-step batches without sacrificing gradient quality.

# Optimizer State Sharding
Standard DDP replicates the full optimizer state on every rank. AdamW stores two FP32 values (m, v) per parameter, so each rank holds 12 bytes/param (4 weights + 4 m + 4 v) — entirely redundant across ranks.

**Idea**
Each rank owns and updates the optimizer state for only **1/world_size** of the parameters. After the optimizer step, each rank broadcasts its updated shard so all ranks end up with the same complete, up-to-date weights.

**Training Step**
```
1. Forward          — all ranks use full model weights (identical)
2. Backward         — all ranks compute full gradients locally
3. all_reduce grads — average gradients across ranks (same as DDP)
4. Optimizer step   — each rank updates only its own 1/N parameter shard
5. Broadcast params — each rank broadcasts its updated shard to all others
```

**Memory Savings**

| Component | Standard DDP | Sharded (N ranks) |
|-----------|-------------|-------------------|
| Weights | P × 4 bytes | P × 4 bytes |
| Adam m | P × 4 bytes | P/N × 4 bytes |
| Adam v | P × 4 bytes | P/N × 4 bytes |
| **Total** | **P × 12 bytes** | **P × (4 + 8/N) bytes** |

At N=4: 12 → 6 bytes/param — **50% reduction**.

**Trade-off**: An extra broadcast of the full parameters is required after each optimizer step. Total communication roughly doubles, but optimizer memory scales as 1/N.

## Our Optimizer Sharding vs ZeRO Stage 1 (ZeRO-DP Pos)

### What ZeRO Stage 1 Does
ZeRO-1 shards optimizer states across ranks, exactly like our implementation. Each rank holds 1/N of the Adam m and v states and only updates its own shard. After the optimizer step, ranks synchronize parameters.

### Key Difference: Communication Pattern

**Our implementation — broadcast per shard:**
Each rank broadcasts its updated parameter shard to all other ranks after step().
Total communication volume = P × 4 bytes (one full model copy, same as all_reduce).
Uses N separate broadcast operations (one per rank).

**ZeRO Stage 1 — reduce_scatter + all_gather:**
Gradients are first reduce_scattered (each rank receives the averaged gradient only for its own shard).
Each rank runs the optimizer step on its shard using the already-reduced gradient.
Parameters are then all_gathered so every rank has the full updated model.
Total communication volume = P × 4 bytes for reduce_scatter + P × 4 bytes for all_gather = 2× all_reduce.

### Memory Difference
Both approaches save the same amount of optimizer state memory: Adam states reduce from P×8 to P/N×8 bytes per rank.
ZeRO-1 additionally avoids storing the full gradient tensor — each rank only needs gradients for its own shard after reduce_scatter, saving another P×4 bytes.
Our implementation must retain full gradients on each rank before the optimizer step.

| Aspect | Our implementation | ZeRO Stage 1 |
|--------|-------------------|--------------|
| Optimizer state memory | P×8/N bytes | P×8/N bytes |
| Gradient memory | P×4 bytes (full) | P×4/N bytes (shard only) |
| Communication ops | N broadcasts | reduce_scatter + all_gather |
| Communication volume | P×4 bytes | P×8 bytes |
| Implementation complexity | Simple | More complex |
