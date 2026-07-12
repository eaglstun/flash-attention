"""Shared MPS attention core for the Apple Silicon (MPS) backend.

This module holds the one implementation of flash-attention's math that both
seams (the FA2 ``flash_attn_gpu`` ABI in ``fa2_backend.py`` and the FA4
``flash_attn.cute.interface`` device branch) build on. See
docs/apple_silicon/PORT_PLAN.md, Phase 1.

Semantics (verified against the CUDA kernels and the reference oracle
``flash_attn/cute/testing.py::attention_ref``):

- Tensor layout: ``q`` is ``(batch, seqlen_q, nheads_q, head_dim)``; ``k``/``v``
  are ``(batch, seqlen_k, nheads_kv, head_dim[_v])``. ``nheads_q`` must be a
  multiple of ``nheads_kv`` (GQA/MQA); query head ``i`` attends with KV head
  ``i // (nheads_q // nheads_kv)``.
- ``softmax_scale`` defaults to ``head_dim ** -0.5``. Scores are
  ``softmax_scale * q @ k^T``.
- ``softcap``: when ``> 0``, scores become ``tanh(scores / softcap) * softcap``
  (applied to the *scaled* scores, before masking — matches ``attention_ref``
  and the CUDA kernels).
- ``causal`` / ``window_size=(left, right)``: bottom-right aligned, exactly as
  ``construct_local_mask``. ``None`` means an infinite window on that side
  (note: the FA2 interface encodes "infinite" as ``-1``; translation to
  ``None`` is the seam adapter's job, because genuinely negative windows are
  legal here). ``causal=True`` forces ``window_size = (left, 0)``. Key ``j``
  is visible to query ``i`` iff
  ``i + seqlen_k - seqlen_q - left <= j <= i + seqlen_k - seqlen_q + right``.
- ``alibi_slopes``: ``(nheads_q,)`` or ``(batch, nheads_q)`` fp32. Adds
  ``-slope * |i + seqlen_k - seqlen_q - j|`` to the scaled (and softcapped)
  scores, after masking — the same quantity ``attn_bias_from_alibi_slopes``
  (tests/test_flash_attn.py) feeds the oracle. For causal attention the CUDA
  kernel uses the row-shifted equivalent ``slope * j``; attention output and
  gradients are identical, ``lse`` differs by a per-row constant.
- ``learnable_sink``: ``(nheads_q,)``. Per-head sink logit added to the
  softmax denominator: with ``m = max(row_max, sink)``, the attention weights
  are ``exp(scores - m) / (sum(exp(scores - m)) + exp(sink - m))`` and
  ``lse = log(sum(exp(scores - m)) + exp(sink - m)) + m`` — matches
  ``attention_ref`` and the FA4 kernels.
- ``lse``: natural-log logsumexp of the scaled, softcapped, masked, biased
  scores. fp32, shape ``(batch, nheads_q, seqlen_q)`` (the FA2 CUDA extension
  and the FA4 dense path both use this layout; FA4 varlen flattens it to
  ``(nheads, total_q)``). Fully-masked rows (possible with causal
  ``seqlen_q > seqlen_k`` or narrow windows) produce ``out = 0`` and
  ``lse = -inf`` with no NaN anywhere, forward or backward. ``-inf`` matches
  the FA4 CuTe kernel (flash_attn/cute/softmax.py:225); the FA2 CUDA kernel
  returns ``+inf`` instead (csrc/flash_attn/src/softmax.h:180) — the FA2 seam
  adapter is responsible for flipping the sign if bit-parity with CUDA
  matters there.

The #1 hazard of this port is silent fp16/bf16 wrongness on MPS, so every
reduction and accumulation here happens in fp32 regardless of input dtype;
outputs are cast back to the input dtype at the very end. ``lse`` stays fp32.
"""

from typing import Optional, Tuple

import torch

__all__ = ["_attention_forward"]


def _build_score_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size: Tuple[Optional[int], Optional[int]],
    device: torch.device,
    row_offset: int = 0,
    seqlen_q_total: Optional[int] = None,
    col_offset: int = 0,
    seqlen_k_total: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Boolean mask (seqlen_q, seqlen_k), True = masked out.

    Bottom-right aligned local mask, identical to
    ``flash_attn/cute/testing.py::construct_local_mask`` for the no-padding
    case. ``row_offset``/``col_offset`` position a tile of a larger score
    matrix (used by the chunked paths); ``seqlen_q_total``/``seqlen_k_total``
    are the full sequence lengths the diagonal is aligned to.
    """
    window_left, window_right = window_size
    if window_left is None and window_right is None:
        return None
    sq = seqlen_q_total if seqlen_q_total is not None else seqlen_q
    sk = seqlen_k_total if seqlen_k_total is not None else seqlen_k
    row_idx = torch.arange(row_offset, row_offset + seqlen_q, device=device).unsqueeze(-1)
    col_idx = torch.arange(col_offset, col_offset + seqlen_k, device=device)
    shift = sk - sq
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    if window_right is not None:
        mask |= col_idx > row_idx + shift + window_right
    if window_left is not None:
        mask |= col_idx < row_idx + shift - window_left
    return mask


def _alibi_bias(
    alibi_slopes: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    row_offset: int = 0,
    seqlen_q_total: Optional[int] = None,
    col_offset: int = 0,
    seqlen_k_total: Optional[int] = None,
) -> torch.Tensor:
    """ALiBi bias, fp32, shape (1 or batch, nheads, seqlen_q, seqlen_k)."""
    sq = seqlen_q_total if seqlen_q_total is not None else seqlen_q
    sk = seqlen_k_total if seqlen_k_total is not None else seqlen_k
    row_idx = torch.arange(row_offset, row_offset + seqlen_q, device=device).unsqueeze(-1)
    col_idx = torch.arange(col_offset, col_offset + seqlen_k, device=device)
    relative_pos = (row_idx + (sk - sq) - col_idx).abs().float()  # (sq, sk)
    slopes = alibi_slopes.float()
    if slopes.dim() == 1:  # (nheads,)
        slopes = slopes.view(1, -1, 1, 1)
    elif slopes.dim() == 2:  # (batch, nheads)
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(
            f"alibi_slopes must be (nheads,) or (batch, nheads), got {tuple(alibi_slopes.shape)}"
        )
    return -slopes * relative_pos


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    _row_offset: int = 0,
    _seqlen_q_total: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain, differentiable, correctness-obvious attention forward.

    All reductions and accumulation in fp32 regardless of input dtype; the
    output is cast back to the input dtype at the end. Returns
    ``(out, lse)`` with ``out`` shaped like ``q`` (``head_dim_v`` in the last
    dim) and ``lse`` fp32 ``(batch, nheads_q, seqlen_q)``.

    ``_row_offset`` / ``_seqlen_q_total`` are private hooks for the chunked
    backward: they declare that this ``q`` is rows
    ``[_row_offset, _row_offset + seqlen_q)`` of a query sequence of total
    length ``_seqlen_q_total``, so masks and ALiBi use global row indices.
    """
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    assert k.shape == (batch, seqlen_k, nheads_kv, head_dim)
    assert v.shape == (batch, seqlen_k, nheads_kv, head_dim_v)
    assert nheads_q % nheads_kv == 0, "nheads_q must be a multiple of nheads_kv (GQA/MQA)"
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    if causal:
        window_size = (window_size[0], 0)
    out_dtype = q.dtype
    device = q.device

    # fp32 everywhere: MPS fp16/bf16 accumulation is the silent-wrongness zone.
    qf, kf, vf = q.float(), k.float(), v.float()
    heads_per_kv = nheads_q // nheads_kv
    if heads_per_kv > 1:
        # Query head i uses KV head i // heads_per_kv (repeat pattern "h -> (h g)").
        kf = kf.repeat_interleave(heads_per_kv, dim=2)
        vf = vf.repeat_interleave(heads_per_kv, dim=2)

    scores = torch.einsum("bthd,bshd->bhts", qf * softmax_scale, kf)
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    mask = _build_score_mask(
        seqlen_q,
        seqlen_k,
        window_size,
        device,
        row_offset=_row_offset,
        seqlen_q_total=_seqlen_q_total,
    )
    if mask is not None:
        scores = scores.masked_fill(mask.view(1, 1, seqlen_q, seqlen_k), float("-inf"))
    if alibi_slopes is not None:
        scores = scores + _alibi_bias(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            device,
            row_offset=_row_offset,
            seqlen_q_total=_seqlen_q_total,
        )

    # Max-shifted softmax that never produces NaN, even for fully-masked rows
    # (all -inf): exp(-inf - 0) == 0, and a zero denominator is replaced by 1
    # so those rows come out as exactly 0 with lse = -inf. The shift is
    # detached — softmax/logsumexp are invariant to it, so gradients are exact.
    row_max = scores.amax(dim=-1)  # (b, h, sq)
    if learnable_sink is not None:
        sink = learnable_sink.float().view(1, nheads_q, 1)  # (1, h, 1)
        row_max = torch.maximum(row_max, sink)
    shift = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max)).detach()
    exp_scores = torch.exp(scores - shift.unsqueeze(-1))  # (b, h, sq, sk)
    denom = exp_scores.sum(dim=-1)  # (b, h, sq)
    if learnable_sink is not None:
        denom = denom + torch.exp(sink - shift)
    denom_is_zero = denom == 0.0
    denom_safe = torch.where(denom_is_zero, torch.ones_like(denom), denom)
    lse = torch.log(denom_safe) + shift
    lse = torch.where(denom_is_zero, torch.full_like(lse, float("-inf")), lse)
    attention = exp_scores / denom_safe.unsqueeze(-1)
    out = torch.einsum("bhts,bshd->bthd", attention, vf)
    return out.to(out_dtype), lse
