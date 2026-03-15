# Model Architecture
## nn.Module
1. 参数自动追踪：用 `nn.Parameter` 包裹的张量会被自动注册，`model.parameters()` 就能遍历到它，优化器直接接管。
```python
self.weight = nn.Parameter(torch.empty(...))
# 用 ModuleList / ModuleDict
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
```
2. 子模块递归注册：赋值给 `self` 的 `nn.Module` 实例会被递归追踪，`model.named_parameters()` 会展开所有层级。
```python
self.layer = nn.Linear(10, 20)
```
3. 训练/推理模式切换：`self.training` 标志位会递归同步到所有子模块。
```python
model.train()   # 启用 Dropout、BatchNorm 的训练行为
model.eval()
```
4. 设备/类型迁移: 同样递归作用于子模块
```python
model.cuda()
model.to(torch.half)
```
5. 状态字典序列化：`state_dict()` 收集所有参数和 buffer（如 BN 的 running mean），是标准的存档/恢复方式。
```python
torch.save(model.state_dict(), "ckpt.pt")
model.load_state_dict(torch.load("ckpt.pt"))
```
6. hook 机制：常用于调试、特征提取、梯度裁剪等场景。
```python
model.register_forward_hook(fn)
model.register_backward_hook(fn)
```


## Linear (without bias)
```python
# uv run pytest -k test_linear
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 权重初始化 并在三倍标准差内截断
        # trunc_normal_ 原地操作
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 等价于 x @ self.weight.T
        return torch.einsum("...i,oi->...o", x, self.weight)
```

## Embedding
```python
# uv run pytest -k test_embedding
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.2, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
```


## RMS Norm
$\text{RMSNorm}(a_i)=
    \frac{a_i}
    {\sqrt{\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_\text{model}} a_i^2 + \varepsilon}} g_i$
Here, $g_i$ is a learnable parameter, $\varepsilon=10^{-5}$
- Upcast input to `torch.float32` to prevent overflow when squaring.
```python
# uv run pytest -k test_rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        # broadcast
        x_norm = x / rms * self.weight
        return x_norm.to(in_dtype)
```

## SwiGLU (positionwise feeddorward)
- SiLU (Switch activation): $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
- GLU (Gated Linear Units): $\text{GLU}(x, W_1, W_2) = \sigma(W_1x)\odot W_2x$
- SwiGLU: $\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3)=W_2(\text{SiLU}(W_1x)\odot W_3x)$
    - where $d_ff=\frac{8}{3} d_{\text{model}}$

```python
# uv run pytest -k test_swiglu
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff else int(((d_model * 8 / 3 + 63) // 64) * 64)

        self.W1 = nn.Parameter(torch.empty((self.d_ff, self.d_model)))
        self.W2 = nn.Parameter(torch.empty((self.d_model, self.d_ff)))
        self.W3 = nn.Parameter(torch.empty((self.d_ff, self.d_model)))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.d_model + self.d_ff))
        for p in [self.W1, self.W2, self.W3]:
            nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_linear = torch.einsum("...m,fm->...f", x, self.W1)
        swish_gate = gate_linear * torch.sigmoid(gate_linear)

        linear = torch.einsum("...m,fm->...f", x, self.W3)

        intermediate = linear * swish_gate

        return torch.einsum("... f, m f -> ... m", intermediate, self.W2)
```

## RoPE (Rotary Position Embedding)
For a given query token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$ at token position \(i\), we apply a pairwise rotation matrix \(R^i\), giving $q'^{(i)} = R^i q^{(i)}$.

\(R^i\) rotates pairs of embedding elements $q^{(i)}_{2k-1:2k}$ as 2D vectors by the angle
\[
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}},
\qquad k \in \{1,\ldots,d/2\},
\]
where \(\Theta\) is a constant.

Consider \(R^i\) to be a block-diagonal matrix of size \(d \times d\), with blocks \(R_k^i\) for \(k \in \{1,\ldots,d/2\}\), where
\[
R_k^i =
\begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}.
\]

The full rotation matrix
\[
R^i =
\begin{bmatrix}
R_1^i & 0 & 0 & \cdots & 0 \\
0 & R_2^i & 0 & \cdots & 0 \\
0 & 0 & R_3^i & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R_{d/2}^i
\end{bmatrix}.
\]

```python
# uv run pytest -k test_rope
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.d_k = d_k
        # 一共 d/2 个 2k/d
        pow_indices = torch.arange(0, d_k, 2).float()
        # 1 / theta^(2k / d)
        inv_freq = 1.0 / theta ** (pow_indices / d_k)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # freqs.shape = (seq_len, d/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
```
`register_buffer`: 把一个 tensor 注册为 module 的“状态”
- 不是可训练参数，不被optimizer更新
- 不出现在`parameters()`
- 会出现在`named_buffers()`
- 会随`.to(deice)`移动
- `persistent=True`决定是否被state_dict()保存。这里cos/sin可以重新计算，所以不保存。

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
    """
    x.shape = (batch, heads, seq_len, d_k)
    token_positions.shape = (seq_len)
    """
    token_positions = token_positions.to(self.cos.device)
    cos = self.cos[token_positions]
    sin = self.sin[token_positions].to(x.dtype)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # (1, seq_len, d_k//2) - broadcasts over heads
    cos = cos.unsqueeze(-3)
    sin = sin.unsqueeze(-3)

    x_paired = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_paired[..., 0]
    x2 = x_paired[..., 1]  # (..., d_k / 2)

    out1 = cos * x1 - sin * x2 # （..., seq_len, d_k / 2)
    out2 = sin * x1 + cos * x2
    # （..., seq_len, d_k / 2, 2) --> (..., seq_len, d_k)
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

## Softmax
$\text{softmax}(v)_i = \frac{\text{exp}(v_i)}{\sum_j \text{exp}(v_j)}$

```python
# uv run pytest -k test_softmax_matches_pytorch
def softmax(x: torch.Tensor, dim: int):
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x_safe = x - max_vals
    exp_x = torch.exp(x_safe)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
```
## Attention
$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q^TK}{\sqrt{d_k}})V$

```python
# uv run pytest -k test_scaled_dot_product_attention
# uv run pytest -k test_4d_scaled_dot_product_attention
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = torch.einsum("...nd, ...md -> ...nm", Q, K) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = torch.einsum("...nm, ...md -> ...nd", attn_weights, V)
    return output
```

## Causal Multi-Head Self-Attention
$\text{MultiHead-SelfAttention}(x) = W_o\text{MultiHead}(W_Qx, W_kx, W_Vx)$
- $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)$
- $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$

```python
# uv run pytest -k test_multihead_self_attention
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: bool,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_qkv = Linear(d_model, 3 * d_model)
        self.W_o = Linear(d_model, d_model)

        if rope:
            self.rope = RotaryPositionEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len
            )
        else:
            self.rope = None

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        *batch_dims, seq_len, _ = x.shape

        Q, K, V = (
            self.W_qkv(x)
            .view(*batch_dims, seq_len, 3, self.num_heads, self.d_k)
            .permute(-3, *range(len(batch_dims)), -2, -4, -1)
            .unbind(dim=0)
        )
        # Each: (*batch_dims, num_heads, seq_len, d_k)
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        out = scaled_dot_product_attention(Q, K, V, causal_mask)
        # (..., num_heads, seq_len, d_k)
        out = (
            out.transpose(-2, -3).contiguous().view(*batch_dims, seq_len, self.d_model)
        )

        return self.W_o(out)
```

## Transformer Block
$y = x + \text{MultiHead-SelfAttention}(\text{RMSNorm}(x))$

```python
# uv run pytest -k test_transformer_block
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope=True,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        dropout=0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=rope,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), token_positions)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ff_norm(x))
        x = x + self.dropout(ff_out)
        return x
```


## Transformer LM

```python
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope=True,
        theta: float = 10000.0,
        dropout: float = 0.0,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=rope,
                    max_seq_len=context_length,
                    theta=theta,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if weight_tying:
            self.lm_head.weight = self.embedding.weight
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.norm(x)
        return self.lm_head(x)
```

## Cross Entropy

```python
# run uv run pytest -k test_cross_entropy
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: [..., classes]
    targets: [samples, ]
    """
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max
    log_sums_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sums_exp - target_logits
    return loss.mean()
```



## AdamW
1. $g_t = \nabla f(\theta_{t-1})$
2. 更新一阶和二阶动量
    - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
    - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
4. $\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
```python
# uv run pytest -k test_adamw
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                # p is a Parameter object
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # memory_format=torch.preserve_format 保证优化器的状态字典（状态张量 m 和 v）与模型权重（p）在物理层面上对齐

                m, v = state["m"], state["v"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                t = state["step"]

                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                step_size = group["lr"] * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add(group["eps"]), value=-step_size)

        return loss
```

## Learning_Rate Schedule
```python
# uv run pytest -k test_get_lr_cosine_schedule
def learning_rate_schedule(t, lr_max, lr_min, Tw, Tc):
    if Tc <= Tw:
        Tc = Tw + 1e-8

    if t < Tw:
        return t / Tw * lr_max
    elif t <= Tc:
        return lr_min + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * pi)) * (lr_max - lr_min)
    else:
        return lr_min
```

## Gradient Clipping
```python
# uv run pytest -k test_gradient_clipping
def gradient_clipping(params, M, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return
    # detach: avoid autograd
    grads_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in grads]))
    if grads_norm > M:
        scale = M / (grads_norm + eps)
        for g in grads:
            g.mul_(scale)
```

## Data Loading
```python
# uv run pytest -k test_get_batch
def data_loading(x, batch_size, context_length, device):
    # 随机取起点
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets
```

## Checkpointing
```python
def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(src, model, optimizer):
    device = next(model.parameters()).device
    ckpt = torch.load(src, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]
```

## Train
