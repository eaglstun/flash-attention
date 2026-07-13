"""Variable-length (varlen) driver for the MPS attention backend.

One differentiable driver around the dense core
(:func:`flash_attn.mps.core.mps_flash_attn_func`), shared by both seams:

- the FA2 ``flash_attn_gpu`` ABI adapter (``fa2_backend.py``) wraps it in
  ``no_grad`` for ``varlen_fwd`` and in ``enable_grad`` + ``autograd.grad``
  for ``varlen_bwd``;
- the FA4 ``flash_attn.cute.interface`` seam (``fa4_backend.py``) calls it
  directly — every op here (index gather/scatter, ``F.pad``) is
  differentiable, so autograd provides the backward.

Two execution strategies (Phase 3a, docs/apple_silicon/BENCHMARKS.md):

- **batched** (default): unpad -> right-pad every sequence to the batch max
  -> ONE call into the dense core with per-batch ``seqused_q``/``seqused_k``
  masking -> repack. One kernel-launch sequence instead of one per sequence.
  The core routes it to masked SDPA when no lse is needed and the flags
  allow, else to the KV-chunked online-softmax path (which produces lse).
- **looped** (fallback): the Phase 1 per-sequence Python loop. Still used
  when padding waste would exceed the launch savings — with wildly skewed
  lengths (one 8k sequence among 128-token ones) the padded batch does
  ``batch * max_len^2`` work and can lose to the loop. The heuristic below
  picks per call; ``benchmarks/mps/bench_varlen.py --skew`` measures it.

Padded positions contribute exactly zero: key columns past ``seqused_k[i]``
are hard-masked (score ``-inf``) in the core, and query rows past
``seqused_q[i]`` come back as ``out = 0`` / ``lse = -inf`` and are dropped
(packed) or zero-filled (batched-4D) on repack. Gradients through the pad
gather/scatter are exact: pad positions receive no upstream gradient at all.

Layout conventions (matching the CUDA extensions):

- Packed mode (tensor is 3D ``(total_tokens, nheads, head_dim)``): sequence
  boundaries come from ``cu_seqlens`` (``(batch + 1,)``, int32/int64);
  ``seqused`` (``(batch,)``) optionally caps each sequence to its first
  ``seqused[i]`` tokens.
- Batched mode (tensor is 4D ``(batch, seqlen, nheads, head_dim)``):
  ``seqused`` gives the used length per batch element (default: the full
  ``seqlen``). Output positions past the used length are zero-filled
  (``lse`` gets ``-inf`` there).
- ``lse`` is fp32, ``(nheads, total_q)`` for packed q, ``(batch, nheads,
  seqlen_q)`` for batched q. Fully-masked rows get ``-inf`` (the core's
  native convention; the FA2 adapter flips the sign — see fa2_backend.py).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from flash_attn.mps.core import _SDPA_MASK_MAX_ELEMENTS, mps_flash_attn_func

__all__ = ["mps_flash_attn_varlen"]

# Batched-vs-looped crossover constants, all measured on M4 Max / torch 2.13
# (benchmarks/mps/bench_varlen.py --skew reproduces the sweep; retuned for
# the Phase 3b split forward, whose rates are ~3x the old fp32 core's).
# Shared shortcut: zero padding waste (every sequence used in full at the
# batch max — pad_area == real_area) always picks batched; it is the same
# math with fewer launches, and the uniform reshape makes the padding free
# (measured: 64x128 no-lse 0.79 ms batched vs 2.59 loop; 8x2048 lse 21.6 vs
# 20.4 — worst case a near-tie). Three regimes otherwise:
#
# 1. lse demanded (or softcap/ALiBi/sink -> the chunked core): cost model
#    below. The batched side scales with the PADDED score area (~1.3 ms per
#    M elements through the split/masked path). The loop side has a term the
#    old waste-only threshold couldn't express: per-sequence calls are
#    ~0.45 ms when lengths repeat but ~2.5-3.5 ms per DISTINCT length (the
#    lse pass compiles/caches MPS graph executables per shape, and a batch
#    of all-distinct shapes churns that cache — measured: 32 all-distinct
#    ragged seqs loop at 111 ms vs 14.5 ms for 32 seqs of only 2 distinct
#    lengths and comparable area). Measured checks: 32 x ragged 64..1024 ->
#    model 44.6 vs 98.6, actual 42 vs 111 (batched); 1x4096+31x128 -> model
#    699 vs 28.6, actual 805 vs 17.3 (looped); 1x1024+31x512 -> model 44.6
#    vs 24.4, actual 41.3 vs 14.5 (looped).
# 2. SDPA-eligible (plain flags, no lse) at inference: each loop iteration
#    is ONE dense `is_causal` SDPA launch (~0.045 ms, no per-shape churn —
#    only the lse pass pays that) — cheaper than the batched path's
#    boolean-mask build + masked SDPA (~0.55 ms per M padded elements +
#    ~1 ms fixed) for ragged batches (32 ragged: loop 4.2 ms vs batched
#    19.9). Uniform batches take the shortcut above instead.
# 3. SDPA-eligible under autograd (the FA2 varlen_bwd recompute, FA4
#    training): one masked-SDPA backward beats `batch` small SDPA backwards,
#    so batched wins unless the per-call padding waste exceeds
#    _PER_CALL_AREA_SDPA_GRAD (masked SDPA burns padding an order of
#    magnitude faster than the lse path).
_PER_CALL_AREA_SDPA_GRAD = 4_000_000
# Regime-1 cost model (milliseconds):
#   loop    = call * batch + shape * n_distinct_lengths + rate_l * real_area
#   batched = fixed + rate_b * padded_area
_LSE_LOOP_CALL_MS = 0.45
_LSE_LOOP_SHAPE_MS = 2.5
_LSE_LOOP_MS_PER_MAREA = 0.55
_LSE_BATCHED_FIXED_MS = 1.0
_LSE_BATCHED_MS_PER_MAREA = 1.3
# Regime-2 cost model (milliseconds): loop = launch * batch + rate * real,
# batched = fixed + rate * padded. Coarse on purpose — the decisions it takes
# are 2x-scale, not 10x-scale, everywhere near the boundary.
_LOOP_SDPA_LAUNCH_MS = 0.045
_LOOP_SDPA_MS_PER_MAREA = 0.10
_BATCHED_SDPA_FIXED_MS = 1.0
_BATCHED_SDPA_MS_PER_MAREA = 0.55


def _seq_lengths(x, cu_seqlens, seqused, batch, name):
    """Per-sequence (start, used_len, full_len) triples for one of q/k/v."""
    if x.dim() == 3:
        if cu_seqlens is None:
            raise ValueError(f"packed (3D) {name} requires cu_seqlens_{name[0]}")
        cu = cu_seqlens.tolist()
        if len(cu) != batch + 1:
            raise ValueError(
                f"cu_seqlens_{name[0]} has {len(cu)} entries, expected batch + 1 = {batch + 1}"
            )
        starts = cu[:-1]
        fulls = [cu[i + 1] - cu[i] for i in range(batch)]
    elif x.dim() == 4:
        starts = [None] * batch  # batched: index by batch dim, not token offset
        fulls = [x.shape[1]] * batch
    else:
        raise ValueError(f"{name} must be 3D (packed) or 4D (batched), got {x.dim()}D")
    if seqused is not None:
        used = [min(int(u), f) for u, f in zip(seqused.tolist(), fulls)]
    else:
        used = fulls
    return starts, used, fulls


def _slice_seq(x, i, start, n):
    """(1, n, nheads, head_dim) view of sequence i."""
    if x.dim() == 3:
        return x[start : start + n].unsqueeze(0)
    return x[i : i + 1, :n]


def _pad_indices(starts, used, max_len, batch, device):
    """Index tensors mapping packed tokens <-> a (batch, max_len) padded grid.

    Returns ``(src_idx, flat_idx)``: token ``src_idx[j]`` of the packed tensor
    lives at flattened position ``flat_idx[j]`` of the ``(batch * max_len)``
    padded tensor. Only the first ``used[i]`` tokens of each sequence are
    mapped; everything else is padding.
    """
    starts_t = torch.tensor(starts, device=device, dtype=torch.long)
    used_t = torch.tensor(used, device=device, dtype=torch.long)
    pos = torch.arange(max_len, device=device)
    valid = pos.unsqueeze(0) < used_t.unsqueeze(1)  # (batch, max_len)
    src_idx = (starts_t.unsqueeze(1) + pos.unsqueeze(0))[valid]
    rows = torch.arange(batch, device=device, dtype=torch.long)
    flat_idx = (rows.unsqueeze(1) * max_len + pos.unsqueeze(0))[valid]
    return src_idx, flat_idx


def _pad_packed(x, src_idx, flat_idx, batch, max_len):
    """(total, h, d) packed -> (batch, max_len, h, d) left-aligned, zero-padded.

    Differentiable: gather + out-of-place index_put onto a constant-zero base.
    """
    padded = x.new_zeros((batch * max_len,) + tuple(x.shape[1:]))
    padded = padded.index_put((flat_idx,), x[src_idx])
    return padded.reshape(batch, max_len, *x.shape[1:])


def _repack_out(padded, src_idx, flat_idx, total):
    """(batch, max_len, h, d) -> (total, h, d); unmapped tokens stay zero."""
    flat = padded.reshape(-1, *padded.shape[2:])
    packed = flat.new_zeros((total,) + tuple(flat.shape[1:]))
    return packed.index_put((src_idx,), flat[flat_idx])


def _repack_lse(lse_padded, src_idx, flat_idx, total):
    """(batch, nheads, max_len) fp32 -> (nheads, total); unmapped tokens -inf."""
    batch, nheads, max_len = lse_padded.shape
    flat = lse_padded.permute(0, 2, 1).reshape(batch * max_len, nheads)
    packed = torch.full(
        (total, nheads), float("-inf"), dtype=lse_padded.dtype, device=lse_padded.device
    )
    packed = packed.index_put((src_idx,), flat[flat_idx])
    return packed.transpose(0, 1).contiguous()


def _prefer_batched(batch, q_used, k_used, loop_hits_sdpa, grad_mode):
    """Pick batched vs looped; see the constants block for regimes and numbers."""
    if batch <= 1:
        return False  # a single sequence: the "loop" is already one dense call
    max_q, max_k = max(q_used), max(k_used)
    if max_q == 0 or max_k == 0:
        return False  # degenerate; the loop handles it with no compute at all
    pad_area = batch * max_q * max_k
    real_area = sum(uq * uk for uq, uk in zip(q_used, k_used))
    if pad_area == real_area:
        return True  # zero waste (uniform lengths): same math, fewer launches
    if loop_hits_sdpa:
        if grad_mode:  # regime 3
            return (pad_area - real_area) / batch <= _PER_CALL_AREA_SDPA_GRAD
        # regime 2
        loop_est = _LOOP_SDPA_LAUNCH_MS * batch + _LOOP_SDPA_MS_PER_MAREA * real_area / 1e6
        batched_est = _BATCHED_SDPA_FIXED_MS + _BATCHED_SDPA_MS_PER_MAREA * pad_area / 1e6
        return batched_est < loop_est
    # regime 1
    n_distinct = len(set(zip(q_used, k_used)))
    loop_est = (
        _LSE_LOOP_CALL_MS * batch
        + _LSE_LOOP_SHAPE_MS * n_distinct
        + _LSE_LOOP_MS_PER_MAREA * real_area / 1e6
    )
    batched_est = _LSE_BATCHED_FIXED_MS + _LSE_BATCHED_MS_PER_MAREA * pad_area / 1e6
    return batched_est < loop_est


def mps_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    _force_batched: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Varlen attention: one padded batched core call, or a per-sequence loop
    for pathologically skewed lengths (module docstring has the strategy).

    Returns ``(out, lse)``; ``lse`` is ``None`` when ``return_lse=False``.
    ``out`` matches ``q``'s layout (packed or batched); ``lse`` is fp32,
    ``(nheads, total_q)`` (packed q) or ``(batch, nheads, seqlen_q)``
    (batched q), with ``-inf`` on fully-masked / unused rows.

    ``_force_batched`` pins the strategy (tests and benchmarks only).
    """
    if q.dim() == 3 and cu_seqlens_q is not None:
        batch = cu_seqlens_q.numel() - 1
    elif q.dim() == 4:
        batch = q.shape[0]
    else:
        raise ValueError("q must be 4D (batched) or 3D packed with cu_seqlens_q")

    q_starts, q_used, q_fulls = _seq_lengths(q, cu_seqlens_q, seqused_q, batch, "query")
    k_starts, k_used, k_fulls = _seq_lengths(k, cu_seqlens_k, seqused_k, batch, "key")
    if q.dim() == 3 and batch > 0 and q_starts[-1] + q_fulls[-1] != q.shape[0]:
        raise ValueError(
            f"cu_seqlens_q covers {q_starts[-1] + q_fulls[-1]} tokens but q has {q.shape[0]}"
        )

    plain_flags = (
        softcap == 0.0
        and window_size[0] is None
        and window_size[1] is None
        and alibi_slopes is None
        and learnable_sink is None
    )
    max_q = max(q_used) if batch > 0 else 0
    max_k = max(k_used) if batch > 0 else 0
    # Would the per-sequence loop hit the dense SDPA fast path? (Per-sequence
    # `is_causal` needs equal q/k lengths; the batched masked-SDPA path needs
    # its boolean mask to fit the cap, otherwise batched means chunked core.)
    loop_hits_sdpa = (
        plain_flags
        and not return_lse
        and (not causal or all(uq == uk for uq, uk in zip(q_used, k_used)))
    )
    batched_mask_fits = batch * max_q * max_k <= _SDPA_MASK_MAX_ELEMENTS
    grad_mode = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if _force_batched is not None:
        use_batched = _force_batched
    elif loop_hits_sdpa and not batched_mask_fits:
        use_batched = False  # loop gets cheap SDPA launches; batched would not
    else:
        use_batched = _prefer_batched(batch, q_used, k_used, loop_hits_sdpa, grad_mode)
    if use_batched and batch > 0 and max_q > 0 and max_k > 0:
        return _varlen_batched(
            q,
            k,
            v,
            batch=batch,
            q_starts=q_starts,
            q_used=q_used,
            q_fulls=q_fulls,
            k_starts=k_starts,
            k_used=k_used,
            k_fulls=k_fulls,
            have_seqused_q=seqused_q is not None,
            have_seqused_k=seqused_k is not None,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            learnable_sink=learnable_sink,
            return_lse=return_lse,
        )
    return _varlen_looped(
        q,
        k,
        v,
        batch=batch,
        q_starts=q_starts,
        q_used=q_used,
        q_fulls=q_fulls,
        k_starts=k_starts,
        k_used=k_used,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        learnable_sink=learnable_sink,
        return_lse=return_lse,
    )


def _varlen_batched(
    q,
    k,
    v,
    *,
    batch,
    q_starts,
    q_used,
    q_fulls,
    k_starts,
    k_used,
    k_fulls,
    have_seqused_q,
    have_seqused_k,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    learnable_sink,
    return_lse,
):
    """Pad to the batch max, run ONE batched core call with per-batch
    ``seqused`` masks, repack. See the module docstring.

    Uniform packed lengths (every sequence used in full and all the same
    length — e.g. fixed-length batches pushed through the varlen API) skip
    the gather/index_put padding entirely: the packed tensor IS the dense
    batch, reshaped. The differentiable gathers cost ~5 ms at 8x2048 on MPS
    (advanced indexing is slow there); the reshape is free — measured, it is
    what put the batched strategy back ahead of the loop for uniform-large
    shapes in the Phase 3b re-sweep."""
    device = q.device
    max_q, max_k = max(q_used), max(k_used)
    neg_inf = float("-inf")

    q_uniform = q.dim() == 3 and all(u == f == max_q for u, f in zip(q_used, q_fulls))
    k_uniform = k.dim() == 3 and all(u == f == max_k for u, f in zip(k_used, k_fulls))

    q_src = q_flat = None
    if q.dim() == 3:
        if q_uniform:
            qp = q.reshape(batch, max_q, *q.shape[1:])
            seqused_q_arg = None
        else:
            q_src, q_flat = _pad_indices(q_starts, q_used, max_q, batch, device)
            qp = _pad_packed(q, q_src, q_flat, batch, max_q)
            # Ragged rows need the per-batch mask; uniform rows don't (the
            # padded tensor is then exactly a dense batch and the diagonal
            # already aligns, since every sequence's used length IS the
            # padded length).
            seqused_q_arg = (
                None
                if all(u == max_q for u in q_used)
                else torch.tensor(q_used, device=device, dtype=torch.long)
            )
    else:
        qp = q[:, :max_q]
        seqused_q_arg = (
            torch.tensor(q_used, device=device, dtype=torch.long) if have_seqused_q else None
        )
    if k.dim() == 3:
        if k_uniform:
            kp = k.reshape(batch, max_k, *k.shape[1:])
            vp = v.reshape(batch, max_k, *v.shape[1:])
            seqused_k_arg = None
        else:
            k_src, k_flat = _pad_indices(k_starts, k_used, max_k, batch, device)
            kp = _pad_packed(k, k_src, k_flat, batch, max_k)
            vp = _pad_packed(v, k_src, k_flat, batch, max_k)
            seqused_k_arg = (
                None
                if all(u == max_k for u in k_used)
                else torch.tensor(k_used, device=device, dtype=torch.long)
            )
    else:
        kp = k[:, :max_k]
        vp = v[:, :max_k]
        seqused_k_arg = (
            torch.tensor(k_used, device=device, dtype=torch.long) if have_seqused_k else None
        )

    out_p, lse_p = mps_flash_attn_func(
        qp,
        kp,
        vp,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        learnable_sink=learnable_sink,
        seqused_q=seqused_q_arg,
        seqused_k=seqused_k_arg,
        return_lse=return_lse,
    )

    if q.dim() == 3:
        total_q = q.shape[0]
        if q_uniform:
            out = out_p.reshape(total_q, *out_p.shape[2:])
            lse = (
                # (b, h, max_q) -> (h, total_q), matching _repack_lse's layout
                lse_p.permute(1, 0, 2).reshape(lse_p.shape[1], total_q) if return_lse else None
            )
            return out, lse
        out = _repack_out(out_p, q_src, q_flat, total_q)
        lse = _repack_lse(lse_p, q_src, q_flat, total_q) if return_lse else None
        return out, lse
    # Batched (4D) q: restore the full seqlen; rows past max_q are padding.
    seqlen_q = q.shape[1]
    if max_q < seqlen_q:
        out = F.pad(out_p, (0, 0, 0, 0, 0, seqlen_q - max_q))
        lse = F.pad(lse_p, (0, seqlen_q - max_q), value=neg_inf) if return_lse else None
    else:
        out, lse = out_p, lse_p
    return out, lse


def _varlen_looped(
    q,
    k,
    v,
    *,
    batch,
    q_starts,
    q_used,
    q_fulls,
    k_starts,
    k_used,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    learnable_sink,
    return_lse,
):
    """The Phase 1 per-sequence loop: one dense-core call per sequence.

    O(batch) kernel launches, but zero padding waste — kept as the fallback
    for pathologically skewed length distributions."""
    nheads_q = q.shape[-2]
    head_dim_v = v.shape[-1]
    neg_inf = float("-inf")

    # 0.0 * (sum of inputs): keeps zero-filled outputs (empty-key sequences,
    # padding rows) connected to the autograd graph so torch.autograd.grad
    # never sees an unused input, while contributing exactly zero gradient.
    zero_hook = 0.0 * (q.sum() + k.sum() + v.sum()) if torch.is_grad_enabled() else 0.0

    out_pieces = []  # (1, n, nheads_q, head_dim_v) per piece, packed order
    lse_pieces = []  # (1, nheads_q, n) fp32 per piece
    for i in range(batch):
        n_q, full_q = q_used[i], q_fulls[i]
        pieces_i = []
        lse_i_pieces = []
        if n_q > 0:
            q_i = _slice_seq(q, i, q_starts[i], n_q)
            n_k = k_used[i]
            if n_k == 0:
                # No visible keys: out = 0, lse = -inf (matches the core's
                # fully-masked-row convention), zero gradient everywhere.
                out_i = q.new_zeros((1, n_q, nheads_q, head_dim_v)) + zero_hook
                lse_i = torch.full(
                    (1, nheads_q, n_q), neg_inf, dtype=torch.float32, device=q.device
                )
            else:
                k_i = _slice_seq(k, i, k_starts[i], n_k)
                v_i = _slice_seq(v, i, k_starts[i], n_k)
                slopes_i = alibi_slopes
                if alibi_slopes is not None and alibi_slopes.dim() == 2:
                    slopes_i = alibi_slopes[i : i + 1]
                out_i, lse_i = mps_flash_attn_func(
                    q_i,
                    k_i,
                    v_i,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=slopes_i,
                    learnable_sink=learnable_sink,
                    return_lse=return_lse,
                )
            pieces_i.append(out_i)
            if return_lse:
                lse_i_pieces.append(lse_i)
        if full_q > n_q:
            # Rows past seqused_q: zero output, -inf lse (nothing attends).
            pad = full_q - n_q
            pieces_i.append(q.new_zeros((1, pad, nheads_q, head_dim_v)) + zero_hook)
            if return_lse:
                lse_i_pieces.append(
                    torch.full((1, nheads_q, pad), neg_inf, dtype=torch.float32, device=q.device)
                )
        if not pieces_i and q.dim() == 4:
            # Batched mode must keep one (possibly empty) piece per batch entry.
            pieces_i.append(q.new_zeros((1, 0, nheads_q, head_dim_v)))
            if return_lse:
                lse_i_pieces.append(
                    torch.empty((1, nheads_q, 0), dtype=torch.float32, device=q.device)
                )
        if pieces_i:
            out_pieces.append(torch.cat(pieces_i, dim=1) if len(pieces_i) > 1 else pieces_i[0])
        if return_lse and lse_i_pieces:
            lse_pieces.append(
                torch.cat(lse_i_pieces, dim=2) if len(lse_i_pieces) > 1 else lse_i_pieces[0]
            )

    if q.dim() == 3:
        total_q = q.shape[0]
        if out_pieces:
            out = torch.cat([p.squeeze(0) for p in out_pieces], dim=0)
        else:
            out = q.new_zeros((0, nheads_q, head_dim_v))
        if out.shape[0] != total_q:
            raise ValueError(f"cu_seqlens_q covers {out.shape[0]} tokens but q has {total_q}")
        lse = None
        if return_lse:
            lse = (
                torch.cat([p.squeeze(0) for p in lse_pieces], dim=-1)
                if lse_pieces
                else torch.empty((nheads_q, 0), dtype=torch.float32, device=q.device)
            )
        return out, lse

    # Batched (4D) q: pad each sequence back to the full seqlen.
    seqlen_q = q.shape[1]
    padded_out = [F.pad(p, (0, 0, 0, 0, 0, seqlen_q - p.shape[1])) for p in out_pieces]
    out = torch.cat(padded_out, dim=0)
    lse = None
    if return_lse:
        padded_lse = [F.pad(p, (0, seqlen_q - p.shape[-1]), value=neg_inf) for p in lse_pieces]
        lse = torch.cat(padded_lse, dim=0)
    return out, lse
