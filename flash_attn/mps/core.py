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

# Phase 3b split-forward constants (all measured on M4 Max / torch 2.13;
# BENCHMARKS.md has the tables and the sweep provenance):
#
# - _LSE_BLOCK_ELEMENTS bounds the fp32 score block a single _lse_forward
#   Q-block materializes (elements, i.e. bytes / 4). 2^28 is 1 GiB of fp32
#   scores per block — big enough that the block loop never dominates, small
#   enough to stay memory-bounded at any seqlen.
# - _SDPA_AUTOGRAD_MAX_ELEMENTS caps the no-lse fast path when gradients are
#   required: torch SDPA's MPS backward materializes the full
#   (batch, heads, seqlen_q, seqlen_k) attention matrix, which hard-fails
#   with "MPSGraph does not support tensor dims larger than INT_MAX" past
#   2^31 elements (measured: 1x16384 h8 dies) and costs O(s^2) memory below
#   it (12 GiB at 2x8192 h8 fp16 — the cap). Past it the memory-flat
#   Q-chunked SDPA-recompute backward takes over at near-identical speed
#   (measured: 8x2048 h8 fwd+bwd 65.6 ms full-autograd vs 64.8 ms
#   chunked-recompute).
# - _SDPA_BWD_CHUNK_ELEMENTS bounds one recompute chunk's score block in the
#   backward (shrinks the Q-chunk when batch * heads * seqlen_k is huge).
#
# fp32 dtype gate: the NEW Phase 3b routes (split forward, SDPA recompute
# backward) are fp16/bf16-only. fp32's parity contract is ulp-referenced to
# an fp64 oracle AND cross-device deterministic (tests/mps pins CPU-vs-MPS
# agreement of the core to 4 ulp): Apple's fused SDPA and CPU's SDPA differ
# from each other and from the reference reduction order at ~1e-6 — well
# inside every fp16/bf16 budget, but 2-4x outside fp32's. So fp32 keeps the
# hand-rolled fp32 ops on the lse-producing/backward paths (Phase 1
# numerics, bit-stable across devices), while fp16/bf16 — the dtypes anyone
# trains or serves with — get the split speed. The long-shipped no-lse fp32
# SDPA fast paths (budget-gated, no cross-device pin) are unchanged.
_LSE_BLOCK_ELEMENTS = 2**28
_SDPA_AUTOGRAD_MAX_ELEMENTS = 2**30
_SDPA_BWD_CHUNK_ELEMENTS = 2**28


def _window_ints(
    window_size: Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    """Coerce window sizes to plain Python ints (None stays None).

    The FA2 suite (and user code imitating it) passes ``torch.randint``
    RESULTS — 0-dim tensors — as window sizes; the CUDA extension's pybind11
    signature coerces them to int, so this backend must too. It is not just
    hygiene: ``x[lo:hi]`` with a 0-dim *tensor* bound silently returns the
    UNSLICED tensor on torch 2.13, so the visible-range K-truncation in the
    split forward / SDPA recompute backward would quietly evaporate while
    the masks are still built for the truncated width — wrong gradients,
    caught by test_flash_attn_qkvpacked on MPS (Phase 3b).
    """
    wl, wr = window_size
    return (None if wl is None else int(wl), None if wr is None else int(wr))


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
    window_size = _window_ints(window_size)
    if causal:
        window_size = (window_size[0], 0)
    out_dtype = q.dtype
    device = q.device

    # fp32 everywhere: MPS fp16/bf16 accumulation is the silent-wrongness zone.
    qf, kf, vf = q.float(), k.float(), v.float()
    heads_per_kv = nheads_q // nheads_kv
    if heads_per_kv > 1:
        # Query head i uses KV head i // heads_per_kv (repeat pattern
        # "h -> (h g)"). The repeat_interleave copies are deliberate: this is
        # the *reference* core, and the oracle (attention_ref) uses the same
        # repeat formulation, so the backward's reduction order matches it to
        # within the fp32 error budget (a grouped einsum lands ~3x the
        # in-dtype baseline's dk error on fp32 MQA — measured, Phase 3b).
        # The performance-bearing paths never come through here for GQA: the
        # split forward and the SDPA recompute backward use enable_gqa, the
        # chunked forward uses grouped einsums.
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
    window_size = _window_ints(window_size)
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


def _sdpa_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Attention output via ``F.scaled_dot_product_attention``, or ``None``
    when it cannot be expressed within the mask budget.

    The caller must already have checked the flags SDPA cannot express at all
    (softcap, ALiBi, learnable_sink — they change the scores/denominator, not
    just the mask). Everything mask-shaped is handled here:

    - plain/GQA dense, and ``is_causal`` only when ``seqlen_q == seqlen_k``
      (SDPA's causal is top-left aligned; flash-attention's is bottom-right);
    - causal with unequal seqlens, local windows, and the per-batch varlen /
      kv-cache formulation (``seqused_q``/``seqused_k``/``key_leftpad``) via a
      boolean keep-mask built by :func:`_build_score_mask` — one mask
      implementation, no re-derivation.

    Fully-masked rows (padding rows, causal rows with no visible keys) are
    temporarily unmasked to keep SDPA's softmax NaN-free, then forced to
    exactly 0; the ``torch.where`` routes any upstream gradient to the
    constant branch, so those rows contribute exactly-0 gradients too.
    Differentiable throughout — usable both under ``no_grad`` (the split
    forward) and under autograd (the no-lse fast path).
    """
    batch, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = k.shape[1]
    device = q.device
    window_size = _window_ints(window_size)
    varlen = seqused_q is not None or seqused_k is not None or key_leftpad is not None
    if (
        not varlen
        and window_size[0] is None
        and window_size[1] is None
        and (not causal or seqlen_q == seqlen_k)
    ):
        return _sdpa_bshd(q, k, v, is_causal=causal, scale=softmax_scale).contiguous()
    if batch * seqlen_q * seqlen_k > _SDPA_MASK_MAX_ELEMENTS:
        return None
    mask = _build_score_mask(
        seqlen_q,
        seqlen_k,
        (window_size[0], 0) if causal else window_size,
        device,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        key_leftpad=key_leftpad,
    )
    if mask is None:  # nothing to mask after all
        return _sdpa_bshd(q, k, v, is_causal=False, scale=softmax_scale).contiguous()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # (1, sq, sk)
    all_masked = mask.all(dim=-1, keepdim=True)  # (b-or-1, sq-or-1, 1)
    keep = ~mask | all_masked
    out = _sdpa_bshd(q, k, v, attn_mask=keep.unsqueeze(1), scale=softmax_scale)
    out = torch.where(
        all_masked.unsqueeze(-1),  # (b-or-1, sq-or-1, 1, 1), broadcasts over h, d
        torch.zeros((), dtype=out.dtype, device=device),
        out,
    )
    return out.contiguous()


def _lse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    key_leftpad: Optional[torch.Tensor] = None,
    block_elements: int = _LSE_BLOCK_ELEMENTS,
) -> torch.Tensor:
    """lse-only forward: fp32 logsumexp of the scaled, masked scores, streamed
    over Q blocks. Needs QK^T but never touches V — roughly a third of the
    full attention's cost — and materializes at most ``block_elements`` fp32
    scores at a time, so it is O(seqlen * block) memory like the chunked core.

    This is the second half of the Phase 3b split forward (SDPA computes
    ``out``, this pass computes ``lse``); it covers exactly the flags
    :func:`_sdpa_out` covers (causal / window / varlen-seqused / leftpad).
    softcap, ALiBi and learnable_sink stay on the chunked core, which
    produces out and lse together.

    Numerics: scores come from an fp32 matmul (inputs upcast, same as the
    chunked core) and the reduction is ``lse = m - log(max(softmax(scores)))``
    with ``m = amax(scores)`` — mathematically the plain row logsumexp the
    oracle computes, arranged so the shifted exp/sum runs inside PyTorch's
    fused fp32 softmax kernel (a broadcast ``scores - m`` materialized by hand
    is ~3x slower on MPS — measured, torch 2.13 / M4 Max, same pathology as
    the masked_fill note above). Verified against an fp64 oracle to the same
    error as the chunked core's online softmax, including a large-logit
    stress test (docs/apple_silicon/BENCHMARKS.md, Phase 3b).

    Returns fp32 ``(batch, nheads_q, seqlen_q)``; fully-masked and padding
    rows get ``-inf`` (never NaN).
    """
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    assert nheads_q % nheads_kv == 0
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    window_size = _window_ints(window_size)
    if causal:
        window_size = (window_size[0], 0)
    window_left, window_right = window_size
    device = q.device
    neg_inf = float("-inf")
    varlen = seqused_q is not None or seqused_k is not None or key_leftpad is not None
    g = nheads_q // nheads_kv

    lse = torch.empty((batch, nheads_kv, g, seqlen_q), device=device, dtype=torch.float32)
    if seqlen_q == 0:
        return lse.reshape(batch, nheads_q, seqlen_q)
    if seqlen_k == 0:
        lse.fill_(neg_inf)
        return lse.reshape(batch, nheads_q, seqlen_q)

    # Head grouping as in the chunked core: query head h = kv * g + gi uses
    # KV head kv, so splitting the head dim as (nheads_kv, g) lines each
    # group up with its KV head and one grouped einsum serves all query
    # heads. einsum, not a pre-transposed batched matmul: MPS matmul
    # collapses on skinny blocks (decode, n=1: 7.3 ms vs einsum's 2.2 ms —
    # measured, BENCHMARKS.md Phase 3b) and merely ties it on dense
    # blocks. Both casts happen exactly once.
    qg = (q.float() * softmax_scale).reshape(batch, seqlen_q, nheads_kv, g, head_dim)
    kf = k.float()  # (b, sk, hkv, d)

    shift = seqlen_k - seqlen_q  # bottom-right diagonal (dense case)
    # Per-batch varlen diagonals defeat the static causal K-truncation below,
    # but a global bound recovers it: column c is visible to row r of batch i
    # only if c <= r + window_right + (sk_end_i - sq_eff_i) (key_leftpad
    # cancels out of the raw-column form), so max_i(sk_end_i - sq_eff_i)
    # bounds every batch at once. Costs one small GPU->CPU sync, so it is
    # only computed where the win dwarfs it (large seqlen_q; decode's sq=1
    # gains nothing and would pay the sync on every token).
    varlen_shift_bound = None
    if varlen and window_right is not None and seqlen_q > 256:
        sk_end = (
            seqused_k.to(torch.long)
            if seqused_k is not None
            else torch.tensor(seqlen_k, device=device)
        )
        sq_eff = (
            seqused_q.to(torch.long)
            if seqused_q is not None
            else torch.tensor(seqlen_q, device=device)
        )
        varlen_shift_bound = int((sk_end - sq_eff).max())
    q_block = min(seqlen_q, max(128, block_elements // max(1, batch * nheads_q * seqlen_k)))
    truncatable = (
        window_left is not None or window_right is not None
        if not varlen
        else varlen_shift_bound is not None
    )
    if truncatable and seqlen_q >= 512:
        # Causal/windowed: the visible-range slicing only saves work when the
        # sequence splits into several blocks (a single block computes the
        # full rectangle). Forcing >= 2 blocks costs one extra dispatch and
        # roughly halves the computed score area for causal — measured knee
        # at 1024-row blocks for 2k, adaptive above (BENCHMARKS.md, Phase 3b).
        q_block = min(q_block, max(256, seqlen_q // 2))
    ninf_f = torch.full((), neg_inf, device=device)

    for r0 in range(0, seqlen_q, q_block):
        r1 = min(r0 + q_block, seqlen_q)
        n = r1 - r0
        # Visible K range for this block (dense only — with per-batch varlen
        # lengths the range is per-batch, so the full range is kept and the
        # mask does the work, except for the global varlen_shift_bound above).
        if varlen:
            lo_vis, hi_vis = 0, seqlen_k
            if varlen_shift_bound is not None:
                hi_vis = max(0, min(seqlen_k, (r1 - 1) + varlen_shift_bound + window_right + 1))
        else:
            hi_vis = (
                seqlen_k
                if window_right is None
                else max(0, min(seqlen_k, (r1 - 1) + shift + window_right + 1))
            )
            lo_vis = 0 if window_left is None else max(0, min(seqlen_k, r0 + shift - window_left))
        if hi_vis <= lo_vis:
            lse[:, :, :, r0:r1] = neg_inf
            continue
        w = hi_vis - lo_vis
        scores = torch.einsum(
            "bthgd,bshd->bhgts", qg[:, r0:r1], kf[:, lo_vis:hi_vis]
        )  # (b, hkv, g, n, w)

        if varlen:
            mask = _build_score_mask(
                n,
                w,
                window_size,
                device,
                row_offset=r0,
                seqlen_q_total=seqlen_q,
                col_offset=lo_vis,
                seqlen_k_total=seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                key_leftpad=key_leftpad,
            )
            if mask is not None:
                # (b, n-or-1, w) -> broadcast over (hkv, g)
                scores = torch.where(mask.unsqueeze(1).unsqueeze(1), ninf_f, scores)
        elif window_left is not None or window_right is not None:
            # Only the diagonal strips need masking: columns in
            # [all_lo, all_hi) are visible to every row of the block, columns
            # outside [lo_vis, hi_vis) were never computed.
            all_lo = lo_vis if window_left is None else min(hi_vis, (r1 - 1) + shift - window_left)
            all_hi = hi_vis if window_right is None else max(lo_vis, r0 + shift + window_right + 1)
            all_lo = max(all_lo, lo_vis)
            all_hi = min(all_hi, hi_vis)
            if all_lo >= all_hi:  # no all-visible middle: mask the whole block
                strips = [(lo_vis, hi_vis)]
            else:
                strips = [(lo_vis, all_lo), (all_hi, hi_vis)]
            for c0, c1 in strips:
                if c1 <= c0:
                    continue
                smask = _build_score_mask(
                    n,
                    c1 - c0,
                    window_size,
                    device,
                    row_offset=r0,
                    seqlen_q_total=seqlen_q,
                    col_offset=c0,
                    seqlen_k_total=seqlen_k,
                )
                if smask is None:
                    continue
                a, b_ = c0 - lo_vis, c1 - lo_vis
                scores[..., a:b_] = torch.where(
                    smask.view(1, 1, 1, n, c1 - c0), ninf_f, scores[..., a:b_]
                )

        m = scores.amax(dim=-1)  # (b, hkv, g, n)
        p = torch.softmax(scores, dim=-1)  # fused shifted exp/sum, fp32
        pmax = p.amax(dim=-1)
        finite = torch.isfinite(m)  # False only when the whole row is masked
        # softmax of an all(-inf) row is NaN; torch.where selects, it does not
        # propagate NaN from the unselected branch.
        l_blk = torch.where(
            finite,
            m - torch.log(torch.where(finite, pmax, torch.ones_like(pmax))),
            torch.full_like(m, neg_inf),
        )
        lse[:, :, :, r0:r1] = l_blk
    return lse.reshape(batch, nheads_q, seqlen_q)


def _attention_backward_chunked(
    dout: torch.Tensor,
    dlse: Optional[torch.Tensor],
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
    q_chunk_size: int = _Q_CHUNK_SIZE_BWD,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-flat Q-chunked recompute backward (the flash-attention backward
    in torch ops). Returns ``(dq, dk, dv)`` in the input dtypes.

    Two recompute strategies per chunk (Phase 3b):

    - **SDPA** (default): recompute the chunk with
      ``F.scaled_dot_product_attention`` on fp32 leaves and differentiate
      through it. Causal/window/varlen masking is expressed as a boolean
      keep-mask from :func:`_build_score_mask` (ALiBi adds an fp32 additive
      bias from :func:`_alibi_bias` — same builders as the forward, no second
      mask implementation). ~2.5x faster than the manual recompute (measured,
      M4 Max), gradients verified against fp64 to the same error as the
      manual path. Fully-masked rows are unmasked-then-zeroed exactly like
      :func:`_sdpa_out`, so they contribute exactly-0 gradients.
    - **manual**: differentiate :func:`_attention_forward` per chunk — kept
      for what SDPA cannot express (softcap, learnable_sink) and for
      gradients arriving through lse (``dlse``), which need the recompute to
      produce a differentiable lse.

    fp32 leaves both ways: per-chunk dK/dV come back fp32, the cross-chunk
    accumulation below is exact, and dK/dV are rounded to the storage dtype
    exactly once at the end (pinned by test_bwd_chunk_accumulation_exact).
    """
    batch, seqlen_q, nheads_q, head_dim = q.shape
    seqlen_k = k.shape[1]
    device = q.device
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    window_size = _window_ints(window_size)
    if causal:
        window_size = (window_size[0], 0)
    window_left, window_right = window_size
    varlen = seqused_q is not None or seqused_k is not None or key_leftpad is not None
    shift = seqlen_k - seqlen_q
    neg_inf = float("-inf")

    # seqlen_k == 1 stays manual: a single-key softmax is the constant 1, so
    # dq and dk are *exactly* 0 — the manual recompute preserves that exact
    # cancellation (autograd's softmax backward subtracts the same product
    # from itself), while SDPA's backward computes dP and rowsum(dout*out)
    # separately and leaves ~1-ulp noise where an exact zero is guaranteed
    # (pinned by test_bwd_chunk_accumulation_exact and the (1,1) parity case,
    # whose fp32-vs-fp64 error budget is legitimately zero). No performance
    # is lost — SDPA has nothing to fuse over one key.
    # fp32 stays manual too (dtype gate, see the constants block up top).
    use_sdpa = (
        softcap == 0.0
        and learnable_sink is None
        and dlse is None
        and seqlen_k > 1
        and q.dtype is not torch.float32
    )
    # Global varlen diagonal bound for K-slicing — same derivation and same
    # sync-cost tradeoff as in _lse_forward (key_leftpad cancels out).
    varlen_shift_bound = None
    if use_sdpa and varlen and window_right is not None and seqlen_q > 256:
        sk_end = (
            seqused_k.to(torch.long)
            if seqused_k is not None
            else torch.tensor(seqlen_k, device=device)
        )
        sq_eff = (
            seqused_q.to(torch.long)
            if seqused_q is not None
            else torch.tensor(seqlen_q, device=device)
        )
        varlen_shift_bound = int((sk_end - sq_eff).max())
    q_chunk = q_chunk_size
    if use_sdpa:
        # Keep one recompute chunk's score block bounded (SDPA's MPS backward
        # materializes (b, h, n, w); past INT_MAX elements it hard-fails).
        q_chunk = min(
            q_chunk_size,
            max(16, _SDPA_BWD_CHUNK_ELEMENTS // max(1, batch * nheads_q * seqlen_k)),
        )

    k_leaf = k.detach().float().requires_grad_()
    v_leaf = v.detach().float().requires_grad_()
    dq_chunks = []
    dk_acc = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv_acc = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    for row_start in range(0, seqlen_q, q_chunk):
        row_end = min(row_start + q_chunk, seqlen_q)
        n = row_end - row_start
        dout_chunk = dout[:, row_start:row_end]
        if use_sdpa:
            # Visible K range for this chunk (same arithmetic as
            # _lse_forward); slicing the leaf keeps autograd exact — the
            # returned dk/dv are full-shaped with zeros outside the slice.
            if varlen:
                lo, hi = 0, seqlen_k
                if varlen_shift_bound is not None:
                    hi = max(
                        0, min(seqlen_k, (row_end - 1) + varlen_shift_bound + window_right + 1)
                    )
            else:
                hi = (
                    seqlen_k
                    if window_right is None
                    else max(0, min(seqlen_k, (row_end - 1) + shift + window_right + 1))
                )
                lo = (
                    0
                    if window_left is None
                    else max(0, min(seqlen_k, row_start + shift - window_left))
                )
            if hi <= lo:
                # Nothing visible for any row of this chunk.
                dq_chunks.append(torch.zeros((batch, n, nheads_q, head_dim), device=device))
                continue
            mask = _build_score_mask(
                n,
                hi - lo,
                window_size,
                device,
                row_offset=row_start,
                seqlen_q_total=seqlen_q,
                col_offset=lo,
                seqlen_k_total=seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                key_leftpad=key_leftpad,
            )
            attn_mask = None
            all_masked = None
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)  # (1, n, w)
                all_masked = mask.all(dim=-1, keepdim=True)  # (b-or-1, n-or-1, 1)
                if alibi_slopes is None:
                    attn_mask = (~mask | all_masked).unsqueeze(1)  # boolean keep
                else:
                    bias = _alibi_bias(
                        alibi_slopes,
                        n,
                        hi - lo,
                        device,
                        row_offset=row_start,
                        seqlen_q_total=seqlen_q,
                        col_offset=lo,
                        seqlen_k_total=seqlen_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        key_leftpad=key_leftpad,
                    )  # (1-or-b, h, n, w) fp32
                    bias = torch.where(
                        mask.unsqueeze(1), torch.full((), neg_inf, device=device), bias
                    )
                    attn_mask = torch.where(
                        all_masked.unsqueeze(1), torch.zeros((), device=device), bias
                    )
            elif alibi_slopes is not None:
                attn_mask = _alibi_bias(
                    alibi_slopes,
                    n,
                    hi - lo,
                    device,
                    row_offset=row_start,
                    seqlen_q_total=seqlen_q,
                    col_offset=lo,
                    seqlen_k_total=seqlen_k,
                )
            q_leaf = q[:, row_start:row_end].detach().float().requires_grad_()
            with torch.enable_grad():
                out_chunk = _sdpa_bshd(
                    q_leaf,
                    k_leaf[:, lo:hi],
                    v_leaf[:, lo:hi],
                    attn_mask=attn_mask,
                    scale=softmax_scale,
                )
                if all_masked is not None:
                    out_chunk = torch.where(
                        all_masked.unsqueeze(-1),
                        torch.zeros((), dtype=out_chunk.dtype, device=device),
                        out_chunk,
                    )
            dq_chunk, dk_chunk, dv_chunk = torch.autograd.grad(
                out_chunk, (q_leaf, k_leaf, v_leaf), dout_chunk.float()
            )
        else:
            q_chunk_leaf = q[:, row_start:row_end].detach().requires_grad_()
            with torch.enable_grad():
                out_chunk, lse_chunk = _attention_forward(
                    q_chunk_leaf,
                    k_leaf,
                    v_leaf,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    learnable_sink=learnable_sink,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    key_leftpad=key_leftpad,
                    _row_offset=row_start,
                    _seqlen_q_total=seqlen_q,
                )
            outputs = (out_chunk,)
            grad_outputs = (dout_chunk,)
            if dlse is not None:
                outputs = outputs + (lse_chunk,)
                grad_outputs = grad_outputs + (dlse[:, :, row_start:row_end],)
            dq_chunk, dk_chunk, dv_chunk = torch.autograd.grad(
                outputs, (q_chunk_leaf, k_leaf, v_leaf), grad_outputs
            )
        dq_chunks.append(dq_chunk)
        dk_acc += dk_chunk.float()
        dv_acc += dv_chunk.float()
    if dq_chunks:
        dq = torch.cat(dq_chunks, dim=1)
    else:
        dq = torch.zeros((batch, 0, nheads_q, head_dim), device=device)
    return dq.to(q.dtype), dk_acc.to(k.dtype), dv_acc.to(v.dtype)


class MPSFlashAttnFunc(torch.autograd.Function):
    """Memory-bounded attention for MPS.

    forward (under ``no_grad`` — no O(seqlen^2) activations are ever saved):

    - **split** (Phase 3b, default whenever the flags are mask-shaped):
      ``out`` from :func:`_sdpa_out` (Apple's fused SDPA — the measured
      torch-level ceiling) and ``lse`` from the streaming fp32
      :func:`_lse_forward` pass. Measured 1.2-2.3x faster than the chunked
      core at every benchmarked shape (the split is never slower — the lse
      pass alone costs less than the chunked core's fp32 out+lse sweep), so
      there is no crossover heuristic; ``force_chunked`` pins the old path
      for tests and benchmarks.
    - **chunked online-softmax core** for what SDPA cannot express (softcap,
      ALiBi, learnable_sink) or when the mask budget is exceeded.

    backward: memory-flat Q-chunked recompute via
    :func:`_attention_backward_chunked` (SDPA-powered per chunk where
    expressible, the manual differentiable core otherwise). dK/dV are
    accumulated in fp32 across chunks and rounded to the storage dtype once.
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
        need_lse=True,
        force_chunked=False,
    ):
        with torch.no_grad():
            out = None
            splittable = (
                not force_chunked
                and q.dtype is not torch.float32  # fp32 dtype gate, see top of file
                and softcap == 0.0
                and alibi_slopes is None
                and learnable_sink is None
                and q.shape[1] > 0
                and k.shape[1] > 0
            )
            if splittable:
                out = _sdpa_out(
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    key_leftpad=key_leftpad,
                )
            if out is not None:
                if need_lse:
                    lse = _lse_forward(
                        q,
                        k,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        key_leftpad=key_leftpad,
                    )
                else:
                    # The caller discards lse; skip the pass entirely. An
                    # autograd.Function output must still be a tensor.
                    lse = torch.empty(0, dtype=torch.float32, device=q.device)
            else:
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
        if dout is None:  # only lse was used downstream
            dout = q.new_zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]))
        if dlse is not None and dlse.numel() == 0:
            dlse = None  # the need_lse=False placeholder output
        dq, dk, dv = _attention_backward_chunked(
            dout,
            dlse,
            q,
            k,
            v,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
            learnable_sink=learnable_sink,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            key_leftpad=key_leftpad,
            q_chunk_size=ctx.q_chunk_size,
        )
        return (dq, dk, dv) + (None,) * 13


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
    _force_chunked: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Memory-bounded MPS attention entry point shared by both seams.

    Returns ``(out, lse)``; ``lse`` is ``None`` when ``return_lse=False``
    (SDPA does not expose lse and the split path skips the lse pass — a
    fabricated lse would be a lie, so none is returned).

    ``seqused_q`` / ``seqused_k`` / ``key_leftpad`` are optional ``(batch,)``
    int tensors declaring per-batch used lengths for the padded-varlen /
    kv-cache formulation (semantics in :func:`_build_score_mask`). Rows past
    ``seqused_q[i]`` produce ``out = 0`` (and ``lse = -inf``).

    Routing (Phase 3b):

    - ``return_lse=False`` + mask-shaped flags (no softcap/ALiBi/sink;
      causal, local windows and varlen/leftpad are all fine —
      :func:`_sdpa_out` expresses them as a boolean mask): one differentiable
      SDPA call, autograd provides the backward. Guarded under grad mode by
      ``_SDPA_AUTOGRAD_MAX_ELEMENTS``: past it SDPA's own backward would
      materialize the full (b, h, sq, sk) score matrix — INT_MAX hard-fail
      at 16k on MPS — so the call routes into :class:`MPSFlashAttnFunc`,
      whose recompute backward is memory-flat at near-identical speed.
    - everything else: :class:`MPSFlashAttnFunc` — split forward (SDPA out +
      streaming fp32 lse) when expressible, chunked online-softmax core
      otherwise, Q-chunked recompute backward always.

    ``_force_chunked`` pins the chunked core (tests and benchmarks only).
    """
    plain_flags = softcap == 0.0 and alibi_slopes is None and learnable_sink is None
    if not return_lse and plain_flags and not _force_chunked and q.shape[1] > 0 and k.shape[1] > 0:
        grad_mode = torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
        )
        score_elements = q.shape[0] * q.shape[2] * q.shape[1] * k.shape[1]
        if not grad_mode or score_elements <= _SDPA_AUTOGRAD_MAX_ELEMENTS:
            out = _sdpa_out(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                key_leftpad=key_leftpad,
            )
            if out is not None:
                return out, None
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
        return_lse,
        _force_chunked,
    )
    return out, (lse if return_lse else None)
