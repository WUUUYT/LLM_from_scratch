"""
flash_attention_triton.py
FlashAttention-2 forward pass implemented as a Triton kernel.
Follows Algorithm 1 from the FlashAttention-2 paper.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd import Function


# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # ── Program indices ───────────────────────────────────────────────────────
    query_tile_index = tl.program_id(0)  # which query tile (i)
    batch_index = tl.program_id(1)  # which batch element

    # ── Block pointers ────────────────────────────────────────────────────────
    # Q tile: [Q_TILE_SIZE, D] — fixed for the entire kernel (we only process one query tile)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # K tile: [K_TILE_SIZE, D] — advances along the key dimension each loop iteration
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V tile: [K_TILE_SIZE, D] — advances along the key dimension each loop iteration
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # O tile: [Q_TILE_SIZE, D] — written once at the end
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L tile: [Q_TILE_SIZE] — written once at the end
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # ── Global query indices for this tile (used for causal mask) ────────────
    # q_global[r] = global row index of the r-th query in this tile
    q_global = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # [Q_TILE_SIZE]

    # ── Load Q tile (stays fixed for all key iterations) ─────────────────────
    Q_i = tl.load(Q_block_ptr)  # [Q_TILE_SIZE, D]

    # ── Initialize on-chip running variables (must be float32) ───────────────
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # row max
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # softmax denom proxy
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # unnormalized output

    # ── Inner loop: iterate over key tiles j = 0 ... T_k-1 ───────────────────
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(T_k):
        # Load K^(j) and V^(j)
        K_j = tl.load(K_block_ptr)  # [K_TILE_SIZE, D]
        V_j = tl.load(V_block_ptr)  # [K_TILE_SIZE, D]

        # Step 1: S_i^(j) = Q_i @ K_j^T * scale   [Q_TILE_SIZE, K_TILE_SIZE]
        S_ij = tl.dot(Q_i, tl.trans(K_j)).to(tl.float32) * scale

        if IS_CAUSAL:
            # k_global[c] = global column index of the c-th key in this tile
            k_global = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # [K_TILE_SIZE]

            # mask[r, c] = True where key position > query position (must be -inf)
            # q_global[:, None]: [Q_TILE_SIZE, 1]
            # k_global[None, :]: [1, K_TILE_SIZE]
            causal_mask = q_global[:, None] < k_global[None, :]  # [Q_TILE_SIZE, K_TILE_SIZE]
            S_ij = tl.where(causal_mask, float("-inf"), S_ij)

        # Step 2: update running row maximum
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))  # [Q_TILE_SIZE]

        # Step 3: unnormalized attention weights P̃ = exp(S - m_new)
        P_tilde = tl.exp(S_ij - m_new[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Step 4: correction factor for old running values
        correction = tl.exp(m_i - m_new)  # [Q_TILE_SIZE]

        # Step 5: update softmax denominator proxy
        l_i = correction * l_i + tl.sum(P_tilde, axis=1)  # [Q_TILE_SIZE]

        # Step 6: update unnormalized output accumulator
        # O_acc = correction * O_acc + P̃ @ V_j
        # Cast P̃ to V dtype before matmul, use acc= for numerical stability
        O_acc = correction[:, None] * O_acc + tl.dot(
            P_tilde.to(V_j.dtype), V_j, acc=tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        )

        # Update running maximum
        m_i = m_new

        # Advance K and V block pointers to the next key tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # ── After all key tiles: normalize and write outputs ─────────────────────

    # O_i = O_acc / l_i
    O_i = O_acc / l_i[:, None]  # [Q_TILE_SIZE, D]

    # L_i = m_i + log(l_i)  (logsumexp, saved for backward)
    L_i = m_i + tl.log(l_i)  # [Q_TILE_SIZE]

    # Cast O to the original dtype before writing to HBM
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i)


# ── Autograd Function ─────────────────────────────────────────────────────────
class FlashAttentionTriton(Function):
    """
    FlashAttention-2 forward pass using the Triton kernel above.

    Interface:
        O = FlashAttentionTriton.apply(Q, K, V, is_causal)

    Inputs:
        Q, K, V: [batch, n_heads, seq_len, d_head]
        is_causal: bool (ignored for now)

    Outputs:
        O: [batch, n_heads, seq_len, d_head]
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda, "FlashAttentionTriton requires CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Q, K, V must be contiguous"

        *batch_shape, N_q, d = Q.shape
        *_, N_k, _ = K.shape
        batch_size = math.prod(batch_shape) if batch_shape else 1

        # Tile sizes (powers of 2, at least 16)
        Q_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_q)))
        K_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_k)))

        scale = 1.0 / math.sqrt(d)

        # Allocate outputs
        Outputs = torch.empty_like(Q)
        L = torch.empty(*batch_shape, N_q, device=Q.device, dtype=torch.float32)

        # Flatten batch and heads into a single "batch" dimension for the kernel
        # Shape: [batch * n_heads, N_q, d]
        Q_2d = Q.reshape(batch_size, N_q, d).contiguous()
        K_2d = K.reshape(batch_size, N_k, d).contiguous()
        V_2d = V.reshape(batch_size, N_k, d).contiguous()
        O_2d = Outputs.reshape(batch_size, N_q, d)
        L_2d = L.reshape(batch_size, N_q)

        # Launch grid: (T_q, batch * n_heads)
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        grid = (T_q, batch_size)

        flash_fwd_kernel[grid](
            Q_2d,
            K_2d,
            V_2d,
            O_2d,
            L_2d,
            Q_2d.stride(0),
            Q_2d.stride(1),
            Q_2d.stride(2),
            K_2d.stride(0),
            K_2d.stride(1),
            K_2d.stride(2),
            V_2d.stride(0),
            V_2d.stride(1),
            V_2d.stride(2),
            O_2d.stride(0),
            O_2d.stride(1),
            O_2d.stride(2),
            L_2d.stride(0),
            L_2d.stride(1),
            N_QUERIES=N_q,
            N_KEYS=N_k,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, Outputs, L)
        ctx.is_causal = is_causal
        ctx.batch_shape = batch_shape

        return Outputs

    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError("FlashAttention-2 Triton backward not yet implemented.")


# ── Convenience wrapper ───────────────────────────────────────────────────────
def flash_attention_triton(Q, K, V, is_causal=False):
    return FlashAttentionTriton.apply(Q, K, V, is_causal)


# ── Adapter for the test suite ────────────────────────────────────────────────
def get_flash_autograd_function_triton():
    """
    Returns the FlashAttentionTriton autograd.Function class.
    Called by adapters.get_flash_autograd_function_triton().
    """
    return FlashAttentionTriton


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    torch.manual_seed(42)

    if not torch.cuda.is_available():
        print("CUDA not available — skipping Triton test.")
        exit()

    device = "cuda"
    B, H, N, d = 2, 4, 64, 32

    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)

    # Reference: standard attention
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V)

    # Triton FlashAttention
    O_triton = flash_attention_triton(Q, K, V)

    max_diff = (O_triton - O_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.2e}")
    print("✅ PASS" if max_diff < 1e-2 else "❌ FAIL — difference too large")
