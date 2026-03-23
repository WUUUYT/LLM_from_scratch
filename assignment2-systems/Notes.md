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

