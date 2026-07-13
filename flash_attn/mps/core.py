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
import torch.nn.functional as F

__all__ = [
    "_attention_forward",
    "_attention_forward_chunked",
    "MPSFlashAttnFunc",
    "mps_flash_attn_func",
]

# Forward tiles the KV sequence; backward recomputes per tile of the Q
# sequence. Both bound peak memory to O(seqlen * chunk) instead of O(seqlen^2).
# Defaults set by the Phase 2 sweep (docs/apple_silicon/BENCHMARKS.md,
# benchmarks/mps/bench_chunks.py): fwd latency is nearly flat in kv_chunk while
# memory scales linearly, so 1024 stays; bwd q_chunk=512 is 4-9% faster than
# 256 on M4 Max with flat-or-lower peak memory.
_KV_CHUNK_SIZE_FWD = 1024
_Q_CHUNK_SIZE_BWD = 512


def _build_score_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size: Tuple[Optional[int], Optional[int]],
    device: torch.device,
    row_offset: int = 0,
    seqlen_q_total: Optional[int] = None,
    col_offset: int = 0,
    seqlen_k_total: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Boolean mask, True = masked out. ``(seqlen_q, seqlen_k)`` in the dense
    case, ``(batch-or-1, seqlen_q-or-1, seqlen_k)`` (broadcastable) when any
    per-batch arg is given.

    Dense: bottom-right aligned local mask, identical to
    ``flash_attn/cute/testing.py::construct_local_mask`` for the no-padding
    case. ``row_offset``/``col_offset`` position a tile of a larger score
    matrix (used by the chunked paths); ``seqlen_q_total``/``seqlen_k_total``
    are the full sequence lengths the diagonal is aligned to.

    Per-batch (padded varlen / kv-cache): ``seqused_q``/``seqused_k``/
    ``key_leftpad`` are ``(batch,)`` integer tensors on ``device``. Query rows
    ``>= seqused_q[i]`` and key columns outside ``[key_leftpad[i],
    seqused_k[i])`` are masked out, and the bottom-right diagonal aligns to
    the per-batch *effective* lengths: with ``c = col - key_leftpad[i]`` and
    ``shift_i = (seqused_k[i] - key_leftpad[i]) - seqused_q[i]``, key ``c`` is
    visible to row ``r`` iff ``r + shift_i - left <= c <= r + shift_i +
    right`` — exactly the loop-over-sequences semantics, batched.
    """
    window_left, window_right = window_size
    varlen = seqused_q is not None or seqused_k is not None or key_leftpad is not None
    if not varlen and window_left is None and window_right is None:
        return None
    row_idx = torch.arange(row_offset, row_offset + seqlen_q, device=device).unsqueeze(-1)
    col_idx = torch.arange(col_offset, col_offset + seqlen_k, device=device)
    if not varlen:
        sq = seqlen_q_total if seqlen_q_total is not None else seqlen_q
        sk = seqlen_k_total if seqlen_k_total is not None else seqlen_k
        shift = sk - sq
        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
        if window_right is not None:
            mask |= col_idx > row_idx + shift + window_right
        if window_left is not None:
            mask |= col_idx < row_idx + shift - window_left
        return mask
    r = row_idx.unsqueeze(0)  # (1, sq, 1), global row indices
    c = col_idx.view(1, 1, -1)  # (1, 1, sk), global col indices
    if key_leftpad is not None:
        c = c - key_leftpad.view(-1, 1, 1)  # effective key position, (b, 1, sk)
    if seqused_q is not None:
        sq_eff = seqused_q.view(-1, 1, 1)
    else:
        sq_eff = seqlen_q_total if seqlen_q_total is not None else seqlen_q
    sk_end = (
        seqused_k.view(-1, 1, 1)
        if seqused_k is not None
        else (seqlen_k_total if seqlen_k_total is not None else seqlen_k)
    )
    sk_eff = sk_end - (key_leftpad.view(-1, 1, 1) if key_leftpad is not None else 0)
    mask = (c < 0) | (c >= sk_eff) | (r >= sq_eff)
    shift = sk_eff - sq_eff
    if window_right is not None:
        mask = mask | (c > r + shift + window_right)
    if window_left is not None:
        mask = mask | (c < r + shift - window_left)
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
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ALiBi bias, fp32, shape (1 or batch, nheads, seqlen_q, seqlen_k).

    With per-batch ``seqused_q``/``seqused_k``/``key_leftpad`` the diagonal
    shift is per-batch (same conventions as :func:`_build_score_mask`);
    positions outside the valid region get a finite garbage bias, which is
    harmless because the mask sets those scores to ``-inf`` first.
    """
    row_idx = torch.arange(row_offset, row_offset + seqlen_q, device=device).unsqueeze(-1)
    col_idx = torch.arange(col_offset, col_offset + seqlen_k, device=device)
    if seqused_q is not None or seqused_k is not None or key_leftpad is not None:
        c = col_idx.view(1, 1, -1)
        if key_leftpad is not None:
            c = c - key_leftpad.view(-1, 1, 1)
        sq_eff = (
            seqused_q.view(-1, 1, 1)
            if seqused_q is not None
            else (seqlen_q_total if seqlen_q_total is not None else seqlen_q)
        )
        sk_end = (
            seqused_k.view(-1, 1, 1)
            if seqused_k is not None
            else (seqlen_k_total if seqlen_k_total is not None else seqlen_k)
        )
        sk_eff = sk_end - (key_leftpad.view(-1, 1, 1) if key_leftpad is not None else 0)
        # (b, sq, sk) -> (b, 1, sq, sk) so it broadcasts against (·, h, 1, 1)
        relative_pos = (row_idx.unsqueeze(0) + (sk_eff - sq_eff) - c).abs().float().unsqueeze(1)
    else:
        sq = seqlen_q_total if seqlen_q_total is not None else seqlen_q
        sk = seqlen_k_total if seqlen_k_total is not None else seqlen_k
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
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
    _row_offset: int = 0,
    _seqlen_q_total: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain, differentiable, correctness-obvious attention forward.

    All reductions and accumulation in fp32 regardless of input dtype; the
    output is cast back to the input dtype at the end. Returns
    ``(out, lse)`` with ``out`` shaped like ``q`` (``head_dim_v`` in the last
    dim) and ``lse`` fp32 ``(batch, nheads_q, seqlen_q)``.

    ``seqused_q`` / ``seqused_k`` / ``key_leftpad`` are optional ``(batch,)``
    int tensors for the padded-varlen / kv-cache formulation (see
    :func:`_build_score_mask` for the exact semantics). Rows past
    ``seqused_q[i]`` come out as ``out = 0`` and ``lse = -inf``, always —
    including with ``learnable_sink`` (whose denominator contribution would
    otherwise leak ``lse = sink`` onto padding rows).

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
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        key_leftpad=key_leftpad,
    )
    if mask is not None:
        # torch.where, not masked_fill: on MPS masked_fill with a broadcast
        # (b, 1, sq, sk) mask is ~6x slower than the equivalent where
        # (measured, M4 Max / torch 2.13). Same values, same (zero) gradient
        # on masked positions.
        mask_b = mask.view(1, 1, seqlen_q, seqlen_k) if mask.dim() == 2 else mask.unsqueeze(1)
        scores = torch.where(
            mask_b, torch.full((), float("-inf"), dtype=scores.dtype, device=device), scores
        )
    if alibi_slopes is not None:
        scores = scores + _alibi_bias(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            device,
            row_offset=_row_offset,
            seqlen_q_total=_seqlen_q_total,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            key_leftpad=key_leftpad,
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
    if seqused_q is not None:
        # Padding rows must report lse = -inf even with learnable_sink (which
        # otherwise puts exp(sink) in every row's denominator). out is already
        # exactly 0 there (all scores masked -> exp_scores row is all zero).
        pad_row = torch.arange(_row_offset, _row_offset + seqlen_q, device=device).view(
            1, 1, -1
        ) >= seqused_q.view(-1, 1, 1)
        lse = torch.where(pad_row, torch.full_like(lse, float("-inf")), lse)
    attention = exp_scores / denom_safe.unsqueeze(-1)
    out = torch.einsum("bhts,bshd->bthd", attention, vf)
    return out.to(out_dtype), lse


def _attention_forward_chunked(
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
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
    kv_chunk_size: int = _KV_CHUNK_SIZE_FWD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax forward, tiled over KV chunks (the flash-attention
    algorithm in torch ops): running max + running sum + rescaled accumulator,
    so the full (seqlen_q, seqlen_k) score matrix is never materialized.

    Same semantics and return values as :func:`_attention_forward` (whose
    docstring is the contract, including the per-batch ``seqused_q`` /
    ``seqused_k`` / ``key_leftpad`` varlen args), but NOT
    differentiable-friendly — it is meant to run under ``torch.no_grad()``
    inside :class:`MPSFlashAttnFunc`. All accumulation in fp32.

    GQA/MQA is computed with grouped einsums over the native KV heads —
    K/V are never materialized per query head (Phase 3: removes the
    ``repeat_interleave`` copies that made GQA slower than MHA).
    """
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    assert nheads_q % nheads_kv == 0
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    if causal:
        window_size = (window_size[0], 0)
    device = q.device
    heads_per_kv = nheads_q // nheads_kv

    qf = q.float() * softmax_scale  # (b, sq, h, d)
    if heads_per_kv > 1:
        # Query head h uses KV head h // heads_per_kv, so splitting the head
        # dim as (nheads_kv, heads_per_kv) lines groups up with their KV head.
        qg = qf.reshape(batch, seqlen_q, nheads_kv, heads_per_kv, head_dim)
    neg_inf = float("-inf")
    row_max = torch.full((batch, nheads_q, seqlen_q), neg_inf, device=device)
    row_sum = torch.zeros((batch, nheads_q, seqlen_q), device=device)
    acc = torch.zeros((batch, nheads_q, seqlen_q, head_dim_v), device=device)

    for col_start in range(0, seqlen_k, kv_chunk_size):
        col_end = min(col_start + kv_chunk_size, seqlen_k)
        n_cols = col_end - col_start
        kc = k[:, col_start:col_end].float()
        vc = v[:, col_start:col_end].float()
        if heads_per_kv > 1:
            scores = torch.einsum("bthgd,bshd->bhgts", qg, kc).reshape(
                batch, nheads_q, seqlen_q, n_cols
            )
        else:
            scores = torch.einsum("bthd,bshd->bhts", qf, kc)  # (b, h, sq, n_cols)
        if softcap > 0.0:
            scores = torch.tanh(scores / softcap) * softcap
        mask = _build_score_mask(
            seqlen_q,
            n_cols,
            window_size,
            device,
            col_offset=col_start,
            seqlen_k_total=seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            key_leftpad=key_leftpad,
        )
        if mask is not None:
            # where, not masked_fill — see the identical note in
            # _attention_forward (broadcast masked_fill is slow on MPS).
            mask_b = mask.view(1, 1, seqlen_q, n_cols) if mask.dim() == 2 else mask.unsqueeze(1)
            scores = torch.where(
                mask_b, torch.full((), neg_inf, dtype=scores.dtype, device=device), scores
            )
        if alibi_slopes is not None:
            scores = scores + _alibi_bias(
                alibi_slopes,
                seqlen_q,
                n_cols,
                device,
                col_offset=col_start,
                seqlen_k_total=seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                key_leftpad=key_leftpad,
            )
        # Online-softmax update. row_max_new can still be -inf (all keys so
        # far masked); the shift uses 0 there so exp(-inf - 0) == 0 and no NaN
        # appears. exp(row_max - shift) is then also exactly 0 for -inf.
        row_max_new = torch.maximum(row_max, scores.amax(dim=-1))
        shift = torch.where(torch.isfinite(row_max_new), row_max_new, torch.zeros_like(row_max_new))
        p = torch.exp(scores - shift.unsqueeze(-1))
        correction = torch.exp(row_max - shift)
        row_sum = row_sum * correction + p.sum(dim=-1)
        if heads_per_kv > 1:
            pv = torch.einsum(
                "bhgts,bshd->bhgtd",
                p.reshape(batch, nheads_kv, heads_per_kv, seqlen_q, n_cols),
                vc,
            ).reshape(batch, nheads_q, seqlen_q, head_dim_v)
        else:
            pv = torch.einsum("bhts,bshd->bhtd", p, vc)
        acc = acc * correction.unsqueeze(-1) + pv
        row_max = row_max_new

    if learnable_sink is not None:
        sink = learnable_sink.float().view(1, nheads_q, 1)
        row_max_new = torch.maximum(row_max, sink)  # always finite
        correction = torch.exp(row_max - row_max_new)
        row_sum = row_sum * correction + torch.exp(sink - row_max_new)
        acc = acc * correction.unsqueeze(-1)
        row_max = row_max_new

    shift = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    sum_is_zero = row_sum == 0.0
    denom = torch.where(sum_is_zero, torch.ones_like(row_sum), row_sum)
    out = (acc / denom.unsqueeze(-1)).transpose(1, 2).contiguous().to(q.dtype)
    lse = torch.log(denom) + shift
    lse = torch.where(sum_is_zero, torch.full_like(lse, neg_inf), lse)
    if seqused_q is not None:
        # Same fix-up as _attention_forward: padding rows report lse = -inf
        # even when learnable_sink put exp(sink) in the denominator.
        pad_row = torch.arange(seqlen_q, device=device).view(1, 1, -1) >= seqused_q.view(-1, 1, 1)
        lse = torch.where(pad_row, torch.full_like(lse, neg_inf), lse)
    return out, lse


class MPSFlashAttnFunc(torch.autograd.Function):
    """Memory-bounded attention for MPS.

    forward: runs the chunked online-softmax forward under ``no_grad`` — no
    O(seqlen^2) activations are ever saved (plain autograd over a chunked
    forward would save every chunk's score block and be right back at
    O(seqlen^2)).

    backward: recomputes the attention chunk-by-chunk over Q blocks and
    differentiates the small, obviously-correct :func:`_attention_forward`
    per block with ``torch.autograd.grad`` — flash-attention recomputes in
    its backward too, and reusing the same core function means there is no
    second copy of the math to drift. dK/dV are accumulated in fp32.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        softcap=0.0,
        alibi_slopes=None,
        learnable_sink=None,
        seqused_q=None,
        seqused_k=None,
        key_leftpad=None,
        kv_chunk_size=_KV_CHUNK_SIZE_FWD,
        q_chunk_size=_Q_CHUNK_SIZE_BWD,
    ):
        with torch.no_grad():
            out, lse = _attention_forward_chunked(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                learnable_sink=learnable_sink,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                key_leftpad=key_leftpad,
                kv_chunk_size=kv_chunk_size,
            )
        ctx.save_for_backward(
            q, k, v, alibi_slopes, learnable_sink, seqused_q, seqused_k, key_leftpad
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.q_chunk_size = q_chunk_size
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, alibi_slopes, learnable_sink, seqused_q, seqused_k, key_leftpad = ctx.saved_tensors
        seqlen_q = q.shape[1]
        q_chunk_size = ctx.q_chunk_size
        if dout is None:  # only lse was used downstream
            dout = q.new_zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]))
        # fp32 leaves (fp16/bf16 -> fp32 is exact, so the recomputed forward is
        # bit-identical): autograd.grad then returns per-chunk dK/dV in fp32,
        # so the cross-chunk accumulation below is exact and dK/dV get rounded
        # to the storage dtype exactly once, at the end. With low-precision
        # leaves each chunk's gradient would be rounded to fp16/bf16 *before*
        # accumulation, leaking chunk-boundary rounding into multi-chunk sums.
        k_leaf = k.detach().float().requires_grad_()
        v_leaf = v.detach().float().requires_grad_()
        dq_chunks = []
        dk_acc = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv_acc = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        for row_start in range(0, seqlen_q, q_chunk_size):
            row_end = min(row_start + q_chunk_size, seqlen_q)
            q_chunk = q[:, row_start:row_end].detach().requires_grad_()
            with torch.enable_grad():
                out_chunk, lse_chunk = _attention_forward(
                    q_chunk,
                    k_leaf,
                    v_leaf,
                    softmax_scale=ctx.softmax_scale,
                    causal=ctx.causal,
                    window_size=ctx.window_size,
                    softcap=ctx.softcap,
                    alibi_slopes=alibi_slopes,
                    learnable_sink=learnable_sink,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    key_leftpad=key_leftpad,
                    _row_offset=row_start,
                    _seqlen_q_total=seqlen_q,
                )
            outputs = (out_chunk,)
            grad_outputs = (dout[:, row_start:row_end],)
            if dlse is not None:
                outputs = outputs + (lse_chunk,)
                grad_outputs = grad_outputs + (dlse[:, :, row_start:row_end],)
            dq_chunk, dk_chunk, dv_chunk = torch.autograd.grad(
                outputs, (q_chunk, k_leaf, v_leaf), grad_outputs
            )
            dq_chunks.append(dq_chunk)
            dk_acc += dk_chunk.float()
            dv_acc += dv_chunk.float()
        dq = torch.cat(dq_chunks, dim=1)
        return (
            dq,
            dk_acc.to(k.dtype),
            dv_acc.to(v.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _sdpa_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` over (b, s, h, d)-layout tensors,
    with the GQA policy shared by both fast paths: ``enable_gqa`` broadcasts
    KV heads internally (torch >= 2.5) instead of us materializing
    ``repeat_interleave`` copies — on MPS only: on CPU the grouped backward's
    dv accumulation order lands ~1.3x outside the fp32 parity budget
    (tests/mps test_sdpa_fast_path), so CPU keeps the materialized copies.
    """
    heads_per_kv = q.shape[2] // k.shape[2]
    if heads_per_kv > 1 and q.device.type == "mps":
        try:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )
            return out.transpose(1, 2)
        except TypeError:  # older torch: no enable_gqa kwarg
            pass
    kf, vf = k, v
    if heads_per_kv > 1:
        kf = k.repeat_interleave(heads_per_kv, dim=2)
        vf = v.repeat_interleave(heads_per_kv, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        kf.transpose(1, 2),
        vf.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )
    return out.transpose(1, 2)


# The varlen SDPA fast path materializes a (batch, seqlen_q, seqlen_k) boolean
# mask. Cap its size (elements == bytes for bool): past this we fall back to
# the KV-chunked core, which never materializes O(seqlen^2) anything. 2^30 is
# 1 GiB of mask — far beyond every benchmarked shape, well under trouble.
_SDPA_MASK_MAX_ELEMENTS = 2**30


def mps_flash_attn_func(
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
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    kv_chunk_size: int = _KV_CHUNK_SIZE_FWD,
    q_chunk_size: int = _Q_CHUNK_SIZE_BWD,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Memory-bounded MPS attention entry point shared by both seams.

    Returns ``(out, lse)``; ``lse`` is ``None`` when ``return_lse=False`` and
    an SDPA fast path was taken (SDPA does not expose lse — a fabricated one
    would be a lie, so none is returned).

    ``seqused_q`` / ``seqused_k`` / ``key_leftpad`` are optional ``(batch,)``
    int tensors declaring per-batch used lengths for the padded-varlen /
    kv-cache formulation (semantics in :func:`_build_score_mask`). Rows past
    ``seqused_q[i]`` produce ``out = 0`` (and ``lse = -inf``).

    Fast paths (``F.scaled_dot_product_attention``), exact-semantics only:

    - dense: no softcap / window / ALiBi / sink, no lse needed, and causal
      only when ``seqlen_q == seqlen_k`` (SDPA's ``is_causal`` is top-left
      aligned; flash-attention's causal mask is bottom-right aligned, which
      differs whenever the seqlens differ).
    - varlen (any ``seqused``/``leftpad`` given): same flag restrictions, but
      causal/window and the per-batch bottom-right alignment are expressed
      through a boolean ``attn_mask`` built by :func:`_build_score_mask` — so
      any seqlens are fine. Fully-masked rows (padding rows, causal rows with
      no visible keys) are temporarily unmasked to keep SDPA's softmax
      NaN-free, then forced to exactly 0 afterwards; their gradients are
      exactly 0 because the zeroing blocks the upstream gradient.
    """
    plain_flags = (
        softcap == 0.0
        and window_size[0] is None
        and window_size[1] is None
        and alibi_slopes is None
        and learnable_sink is None
    )
    varlen = seqused_q is not None or seqused_k is not None or key_leftpad is not None
    if not varlen and not return_lse and plain_flags and (not causal or q.shape[1] == k.shape[1]):
        out = _sdpa_bshd(q, k, v, is_causal=causal, scale=softmax_scale)
        return out.contiguous(), None
    if (
        varlen
        and not return_lse
        and plain_flags
        and q.shape[0] * q.shape[1] * k.shape[1] <= _SDPA_MASK_MAX_ELEMENTS
    ):
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        masked = _build_score_mask(
            seqlen_q,
            seqlen_k,
            (window_size[0], 0) if causal else window_size,
            q.device,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            key_leftpad=key_leftpad,
        )
        # Fully-masked rows would make SDPA's softmax produce NaN. Unmask
        # them (their scores then softmax to *something* finite) and force
        # the output rows to exactly 0 afterwards; torch.where also routes
        # their upstream gradient to the constant branch, so dq/dk/dv
        # contributions from these rows are exactly 0.
        all_masked = masked.all(dim=-1, keepdim=True)  # (b, sq-or-1, 1)
        keep = ~masked | all_masked
        out = _sdpa_bshd(q, k, v, attn_mask=keep.unsqueeze(1), scale=softmax_scale)
        out = torch.where(
            all_masked.unsqueeze(-1),  # (b, sq-or-1, 1, 1), broadcasts over h and d
            torch.zeros((), dtype=out.dtype, device=out.device),
            out,
        )
        return out.contiguous(), None
    out, lse = MPSFlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        tuple(window_size),
        softcap,
        alibi_slopes,
        learnable_sink,
        seqused_q,
        seqused_k,
        key_leftpad,
        kv_chunk_size,
        q_chunk_size,
    )
    return out, (lse if return_lse else None)
