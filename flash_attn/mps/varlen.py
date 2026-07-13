"""Variable-length (varlen) driver for the MPS attention backend.

One differentiable per-sequence loop around the dense core
(:func:`flash_attn.mps.core.mps_flash_attn_func`), shared by both seams:

- the FA2 ``flash_attn_gpu`` ABI adapter (``fa2_backend.py``) wraps it in
  ``no_grad`` for ``varlen_fwd`` and in ``enable_grad`` + ``autograd.grad``
  for ``varlen_bwd``;
- the FA4 ``flash_attn.cute.interface`` seam (``fa4_backend.py``) calls it
  directly — every op here (slicing, ``torch.cat``, ``F.pad``) is
  differentiable, so autograd provides the backward.

Correct first, fast never mind: this is a Python loop over the batch, one
dense-core call per sequence. It is the honest O(batch)-kernel-launches
implementation, not a fused varlen kernel — see
docs/apple_silicon/MPS_STATUS.md for the performance caveat.

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

from flash_attn.mps.core import mps_flash_attn_func

__all__ = ["mps_flash_attn_varlen"]


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Varlen attention as a differentiable per-sequence loop over the dense core.

    Returns ``(out, lse)``; ``lse`` is ``None`` when ``return_lse=False``.
    ``out`` matches ``q``'s layout (packed or batched); ``lse`` is fp32,
    ``(nheads, total_q)`` (packed q) or ``(batch, nheads, seqlen_q)``
    (batched q), with ``-inf`` on fully-masked / unused rows.
    """
    if q.dim() == 3 and cu_seqlens_q is not None:
        batch = cu_seqlens_q.numel() - 1
    elif q.dim() == 4:
        batch = q.shape[0]
    else:
        raise ValueError("q must be 4D (batched) or 3D packed with cu_seqlens_q")
    nheads_q = q.shape[-2]
    head_dim_v = v.shape[-1]
    neg_inf = float("-inf")

    q_starts, q_used, q_fulls = _seq_lengths(q, cu_seqlens_q, seqused_q, batch, "query")
    k_starts, k_used, _ = _seq_lengths(k, cu_seqlens_k, seqused_k, batch, "key")

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
