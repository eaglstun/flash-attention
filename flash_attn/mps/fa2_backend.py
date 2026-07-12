"""MPS drop-in for the ``flash_attn_2_cuda`` C extension (FA2 backend ABI).

``flash_attn/flash_attn_interface.py`` binds its entire compute backend to a single
module named ``flash_attn_gpu`` and calls exactly five functions on it, all
positionally: ``fwd``, ``varlen_fwd``, ``bwd``, ``varlen_bwd``, ``fwd_kvcache``.
This module mirrors that ABI for Apple Silicon.

Phase 0 (current): every function raises ``NotImplementedError`` with a clear,
actionable message. The signatures below are the contract for Phase 1, transcribed
from the actual call sites in ``flash_attn_interface.py`` — do not reorder arguments.
See docs/apple_silicon/PORT_PLAN.md for the plan.

Conventions shared by all five functions (matching the C extension):

- Tensor layout is ``(batch, seqlen, nheads, headdim)`` (or ``(total, nheads,
  headdim)`` for varlen), last dim contiguous.
- ``out`` / ``dq`` / ``dk`` / ``dv`` arguments are optional preallocated output
  tensors; when ``None`` the backend allocates them.
- ``gen`` is an optional ``torch.Generator`` for dropout RNG; the interface always
  passes ``None``.
- ``softmax_lse`` is float32 with shape ``(batch, nheads, seqlen_q)`` (dense) or
  ``(nheads, total_q)`` (varlen).
"""

_PHASE0_MSG = (
    "flash_attn MPS backend: {name}() is not implemented yet. "
    "The Apple Silicon (MPS) backend currently provides import-level compatibility "
    "only (Phase 0). The torch-based attention math for MPS lands in Phase 1 — "
    "see docs/apple_silicon/PORT_PLAN.md in the flash-attention repo."
)


def _not_implemented(name):
    raise NotImplementedError(_PHASE0_MSG.format(name=name))


def fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    """Dense forward. Mirrors ``flash_attn_2_cuda.fwd``.

    Called from ``_flash_attn_forward`` (flash_attn_interface.py).

    Returns:
        (out, softmax_lse, S_dmask, rng_state)
        - out: (batch, seqlen_q, nheads, headdim), dtype of q
        - softmax_lse: (batch, nheads, seqlen_q), float32
        - S_dmask: attention probs / dropout mask; empty tensor unless
          ``return_softmax`` and ``dropout_p > 0``
        - rng_state: (2,) int64 dropout RNG state (may be a dummy when
          ``dropout_p == 0``)
    """
    _not_implemented("fwd")


def varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    num_splits,
):
    """Variable-length forward. Mirrors ``flash_attn_2_cuda.varlen_fwd``.

    Called from ``_flash_attn_varlen_forward`` (flash_attn_interface.py).
    ``q``/``k``/``v`` are packed as (total_tokens, nheads, headdim) with boundaries
    given by ``cu_seqlens_q``/``cu_seqlens_k`` (int32, (batch + 1,)).

    Returns:
        (out, softmax_lse, S_dmask, rng_state)
        - out: (total_q, nheads, headdim), dtype of q
        - softmax_lse: (nheads, total_q), float32
        - S_dmask, rng_state: as in ``fwd``
    """
    _not_implemented("varlen_fwd")


def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen,
    rng_state,
):
    """Dense backward. Mirrors ``flash_attn_2_cuda.bwd``.

    Called from ``_flash_attn_backward`` (flash_attn_interface.py). ``dq``/``dk``/
    ``dv`` are preallocated by the caller and must be filled in place.

    Returns:
        (dq, dk, dv, softmax_d)
        - dq: (batch, seqlen_q, nheads, headdim)
        - dk, dv: (batch, seqlen_k, nheads_k, headdim)
        - softmax_d: (batch, nheads, seqlen_q_rounded), float32 (only softmax_d is
          consumed by the caller; dq/dk/dv are read through the preallocated tensors)
    """
    _not_implemented("bwd")


def varlen_bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen,
    rng_state,
):
    """Variable-length backward. Mirrors ``flash_attn_2_cuda.varlen_bwd``.

    Called from ``_flash_attn_varlen_backward`` (flash_attn_interface.py).

    Returns:
        (dq, dk, dv, softmax_d)
        - dq: (total_q, nheads, headdim)
        - dk, dv: (total_k, nheads_k, headdim)
        - softmax_d: (nheads, total_q_rounded), float32
    """
    _not_implemented("varlen_bwd")


def fwd_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    rotary_cos,
    rotary_sin,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    alibi_slopes,
    out,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    rotary_interleaved,
    num_splits,
):
    """Forward with KV cache (inference decode path). Mirrors
    ``flash_attn_2_cuda.fwd_kvcache``.

    Called from ``flash_attn_with_kvcache`` (flash_attn_interface.py). ``k``/``v``
    (optional new tokens) are appended into ``k_cache``/``v_cache`` in place at
    ``cache_seqlens``; ``block_table`` selects paged-KV layout.

    Returns:
        (out, softmax_lse)
        - out: (batch, seqlen_q, nheads, headdim), dtype of q
        - softmax_lse: (batch, nheads, seqlen_q), float32
    """
    _not_implemented("fwd_kvcache")
