"""
flash_attention_pytorch.py
Pure PyTorch implementation of FlashAttention-2 forward pass.
No Triton — useful for debugging the Triton kernel.
"""

import math

import torch
from torch.autograd import Function


class FlashAttentionPytorch(Function):
    """
    Interface:
        O = FlashAttentionPytorch.apply(Q, K, V, is_causal)

    Inputs:
        Q: [..., N_q, d]
        K: [..., N_k, d]
        V: [..., N_k, d]
        is_causal: bool (ignored for now)

    Outputs:
        O: [batch, n_heads, N_q, d]
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *input_shape, N_q, d = Q.shape
        *_, N_k, _ = K.shape

        # ── Tile sizes (at least 16×16) ──────────────────────────────────────
        B_q = max(16, min(64, N_q))  # query tile size
        B_k = max(16, min(64, N_k))  # key tile size

        scale = 1.0 / math.sqrt(d)

        # ── Output tensors ───────────────────────────────────────────────────
        Outputs = torch.zeros(*input_shape, N_q, d, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(*input_shape, N_q, device=Q.device, dtype=Q.dtype)

        T_q = math.ceil(N_q / B_q)  # number of query tiles
        T_k = math.ceil(N_k / B_k)  # number of key tiles

        # ── Outer loop: query tiles ───────────────────────────────────────────
        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)

            # Load Q tile: [batch, n_heads, B_q, d]
            Q_i = Q[..., q_start:q_end, :]

            actual_B_q = q_end - q_start  # handle last tile boundary

            # m: running row maximum,        [..., actual_B_q]
            # l: softmax denominator proxy,  [..., actual_B_q]
            # O_acc: unnormalized output,    [..., actual_B_q, d]
            m_i = torch.full((*input_shape, actual_B_q), float("-inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((*input_shape, actual_B_q), device=Q.device, dtype=torch.float32)
            O_acc = torch.zeros((*input_shape, actual_B_q, d), device=Q.device, dtype=torch.float32)

            # ── Inner loop: key tiles ─────────────────────────────────────────
            for j in range(T_k):
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)

                # Load K, V tiles: [batch, n_heads, B_k, d]
                K_j = K[..., k_start:k_end, :]
                V_j = V[..., k_start:k_end, :]

                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                m_new = torch.maximum(m_i, S_ij.amax(dim=-1))
                P_tilde = torch.exp(S_ij - m_new.unsqueeze(-1))
                correction = torch.exp(m_i - m_new)
                l_i = correction * l_i + P_tilde.sum(dim=-1)
                O_acc = correction.unsqueeze(-1) * O_acc + torch.matmul(P_tilde, V_j)

                m_i = m_new

            Outputs[..., q_start:q_end, :] = O_acc / l_i.unsqueeze(-1)
            L[..., q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, Outputs, L)
        ctx.is_causal = is_causal

        return Outputs

    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError(
            "FlashAttention-2 backward not yet implemented. Use the Triton kernel for the full implementation."
        )


# ── Convenience wrapper ───────────────────────────────────────────────────────
def flash_attention_pytorch(Q, K, V, is_causal=False):
    """
    Convenience function wrapping FlashAttentionPytorch.apply.

    Args:
        Q, K, V: [batch, n_heads, seq_len, d_head]
        is_causal: bool

    Returns:
        O: [batch, n_heads, seq_len, d_head]
    """
    return FlashAttentionPytorch.apply(Q, K, V, is_causal)


# ── Reference implementation (standard attention) for testing ─────────────────
def reference_attention(Q, K, V, is_causal=False):
    """
    Standard scaled dot-product attention for correctness comparison.
    Equations 4-6 from the FlashAttention-2 paper.
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]

    if is_causal:
        mask = torch.ones(S.shape[-2], S.shape[-1], device=Q.device).tril()
        S = S.masked_fill(mask == 0, float("-inf"))

    P = torch.softmax(S, dim=-1)  # [B, H, N_q, N_k]
    O = torch.matmul(P, V)  # [B, H, N_q, d]
    return O


# ── Adapter for the test suite ────────────────────────────────────────────────
def get_flashattention_autograd_function_pytorch():
    """
    Returns the FlashAttentionPytorch autograd.Function class.
    Called by adapters.get_flashattention_autograd_function_pytorch().
    """
    return FlashAttentionPytorch


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, N, d = 2, 4, 64, 32
    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)

    O_flash = flash_attention_pytorch(Q, K, V)
    O_ref = reference_attention(Q, K, V)

    max_diff = (O_flash - O_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.2e}")
    print("✅ PASS" if max_diff < 1e-4 else "❌ FAIL — difference too large")
