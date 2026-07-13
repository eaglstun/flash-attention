"""MPS backend for the FA4 (``flash_attn.cute``) public API.

``flash_attn.cute.interface.flash_attn_func`` / ``flash_attn_varlen_func``
dispatch here for MPS tensors, ABOVE their ``autograd.Function``s — everything
below is differentiable torch (the shared core ``flash_attn/mps/core.py`` and
the varlen driver ``flash_attn/mps/varlen.py``), so autograd provides the
backward.

Contract preserved from the CUDA path:

- Both functions always return an ``(out, lse)`` 2-tuple; ``lse`` is ``None``
  when ``return_lse=False``.
- ``lse`` is fp32; fully-masked rows get ``-inf`` — the core's native
  convention IS the FA4/CuTe convention (flash_attn/cute/softmax.py:225), so
  unlike the FA2 adapter no sign flip happens here.
- causal/local/window resolution reuses the interface's own
  ``_resolve_causal_local_window`` (flash_attn/cute/interface.py), including
  its ``left + right < 0`` collapse quirk — one source of truth, not a copy.
- Perf-only knobs (``num_splits``, ``pack_gqa``, ``deterministic``,
  ``min_seqlen_k``, ``max_seqlen_*``) are accepted and ignored.
- Genuinely unimplementable features (cute-typed ``score_mod``/``mask_mod``
  callables, block sparsity, paged KV, MLA ``qv``, ``gather_kv_indices``)
  raise ``NotImplementedError`` naming the feature.
"""

from typing import Optional, Tuple

import torch

from flash_attn.mps.core import mps_flash_attn_func
from flash_attn.mps.varlen import mps_flash_attn_varlen

__all__ = ["fa4_mps_flash_attn_func", "fa4_mps_flash_attn_varlen_func"]


def _unsupported(feature):
    raise NotImplementedError(
        f"flash_attn.cute on MPS (Apple Silicon): {feature} is not supported by the "
        "torch-based MPS backend. See docs/apple_silicon/MPS_STATUS.md for the "
        "feature matrix."
    )


def _reject_unsupported(**features):
    for name, value in features.items():
        if isinstance(value, (tuple, list)) and len(value) == 0:
            continue  # empty aux containers are as good as absent
        if value is not None:
            _unsupported(name)


def _resolve_window(causal, window_size):
    """Delegate to the interface's own resolver (single source of the quirks)."""
    from flash_attn.cute.interface import _resolve_causal_local_window

    causal, _local, left, right = _resolve_causal_local_window(
        causal, window_size[0], window_size[1]
    )
    return causal, (left, right)


def _zero_kv_result(q, k, v, head_dim_v, return_lse):
    """seqlen_k == 0: out = 0, lse = -inf (mirrors interface.py's early exit),
    graph-connected so autograd sees zero (not missing) gradients."""
    batch, seqlen_q, nheads, _ = q.shape
    hook = (
        0.0 * (q.sum() + k.sum() + v.sum())
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        else 0.0
    )
    out = q.new_zeros((batch, seqlen_q, nheads, head_dim_v)) + hook
    lse = (
        torch.full((batch, nheads, seqlen_q), float("-inf"), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    return out, lse


def fa4_mps_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv=None,
    gather_kv_indices=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    """MPS twin of flash_attn.cute.interface.flash_attn_func. Returns (out, lse)."""
    _reject_unsupported(
        qv=qv,
        gather_kv_indices=gather_kv_indices,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
        block_sparse_tensors_bwd=block_sparse_tensors_bwd,
    )
    causal, window = _resolve_window(causal, window_size)
    if k.shape[1] == 0:
        return _zero_kv_result(q, k, v, v.shape[-1], return_lse)
    return mps_flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window,
        softcap=softcap,
        learnable_sink=learnable_sink,
        return_lse=return_lse,
    )


def fa4_mps_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices=None,
    page_table=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    aux_scalars=None,
    return_lse: bool = False,
):
    """MPS twin of flash_attn.cute.interface.flash_attn_varlen_func.

    Supports packed 3D q/k/v with cu_seqlens (optionally capped by seqused_*)
    and batched 4D tensors with seqused_*. Returns (out, lse) with the FA4
    layouts: out like q; lse (nheads, total_q) for packed q, (batch, nheads,
    seqlen_q) for batched q.
    """
    _reject_unsupported(
        qv=qv,
        gather_kv_indices=gather_kv_indices,
        page_table=page_table,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
    )
    causal, window = _resolve_window(causal, window_size)
    if (
        q.dim() == 4
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    ):
        # No varlen metadata at all: this is dense attention.
        if k.shape[1] == 0:
            return _zero_kv_result(q, k, v, v.shape[-1], return_lse)
        return mps_flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window,
            softcap=softcap,
            learnable_sink=learnable_sink,
            return_lse=return_lse,
        )
    return mps_flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window,
        softcap=softcap,
        learnable_sink=learnable_sink,
        return_lse=return_lse,
    )
