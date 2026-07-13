"""Seam B tests: the FA4 public API (``flash_attn.cute.interface``) on MPS.

Exercises the MPS dispatch in ``flash_attn.cute.flash_attn_func`` /
``flash_attn_varlen_func`` (backed by ``flash_attn/mps/fa4_backend.py``). CPU
is the oracle, MPS is the defendant, budgets are the cute suite's own.

FA4-shaped contracts pinned here:

- Both functions ALWAYS return an ``(out, lse)`` 2-tuple, ``lse=None`` when
  ``return_lse=False`` (interface.py contract).
- ``lse`` for fully-masked rows is ``-inf`` (FA4/CuTe convention,
  flash_attn/cute/softmax.py:225) — no FA2-style sign flip.
- The backward comes from autograd (the dispatch sits above the
  autograd.Function), including gradients arriving through ``lse``.
- window semantics go through the interface's own
  ``_resolve_causal_local_window`` (including the ``left + right < 0``
  collapse quirk).
- perf knobs (``num_splits``, ``pack_gqa``, ``deterministic``) are accepted;
  cute-only features (``score_mod``/``mask_mod``, ``qv``, ``page_table``,
  block sparsity, ``gather_kv_indices``) raise ``NotImplementedError``.
"""

import pytest
import torch

from checks import check_tensor_budget
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

DEVICE = "mps"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _oracle(q, k, v, g, **kwargs):
    """Reference one precision level above the test dtype (fp64 for fp32 inputs,
    fp32-upcast otherwise — same scheme as tests/mps/test_attention_parity.py)."""
    if q.dtype == torch.float32:
        ref = [t.detach().double().requires_grad_() for t in (q, k, v)]
        ref_kwargs = dict(upcast=False, **kwargs)
    else:
        ref = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        ref_kwargs = kwargs
    pt = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    out_ref, _, lse_ref = attention_ref(*ref, None, None, return_lse=True, **ref_kwargs)
    out_pt, _, lse_pt = attention_ref(
        *pt, None, None, upcast=False, reorder_ops=True, return_lse=True, **kwargs
    )
    dref = torch.autograd.grad(out_ref, ref, g.to(out_ref.dtype))
    dpt = torch.autograd.grad(out_pt, pt, g)
    return (out_ref, lse_ref, dref), (out_pt, lse_pt, dpt)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha", ["mha", "gqa"])
@pytest.mark.parametrize(
    "window_size,softcap",
    [((None, None), 0.0), ((21, None), 0.0), ((None, None), 30.0)],
    ids=["plain", "local", "softcap"],
)
def test_fa4_flash_attn_func(dtype, causal, mha, window_size, softcap):
    batch, sq, sk, h, d = 2, 113, 203, 8, 64
    h_kv = {"mha": 8, "gqa": 2}[mha]
    torch.manual_seed(21)
    q = torch.randn(batch, sq, h, d).to(dtype)
    k = torch.randn(batch, sk, h_kv, d).to(dtype)
    v = torch.randn(batch, sk, h_kv, d).to(dtype)
    g = torch.randn(batch, sq, h, d).to(dtype)
    (out_ref, lse_ref, dref), (out_pt, lse_pt, dpt) = _oracle(
        q, k, v, g, causal=causal, window_size=window_size, softcap=softcap
    )
    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    result = flash_attn_func(
        qm,
        km,
        vm,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_lse=True,
    )
    assert isinstance(result, tuple) and len(result) == 2, (
        "must be an (out, lse) 2-tuple"
    )
    out, lse = result
    assert out.dtype == dtype
    assert lse.dtype == torch.float32 and lse.shape == (batch, h, sq)
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))

    rtol = 2.0 if softcap == 0.0 else 3.0
    extra = 0.0 if softcap == 0.0 else 3e-4
    detail = f"fa4 {mha} causal={causal} win={window_size} cap={softcap}"
    check_tensor_budget("out", out, out_ref, out_pt, dtype, rtol=rtol, detail=detail)
    check_tensor_budget("lse", lse, lse_ref, lse_pt, dtype, rtol=rtol, detail=detail)
    for name, a, r, p in [
        ("dq", dq, dref[0], dpt[0]),
        ("dk", dk, dref[1], dpt[1]),
        ("dv", dv, dref[2], dpt[2]),
    ]:
        check_tensor_budget(
            name, a, r, p, dtype, rtol=rtol, extra_atol=extra, detail=detail
        )


def test_fa4_two_tuple_contract():
    q = torch.randn(1, 16, 2, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 16, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 16, 2, 64, device=DEVICE, dtype=torch.float16)
    result = flash_attn_func(q, k, v, causal=True)  # return_lse=False
    assert isinstance(result, tuple) and len(result) == 2
    out, lse = result
    assert lse is None, "return_lse=False returns (out, None), still a 2-tuple"
    cu = torch.tensor([0, 16], dtype=torch.int32, device=DEVICE)
    result = flash_attn_varlen_func(
        q[0],
        k[0],
        v[0],
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=16,
        max_seqlen_k=16,
        causal=True,
    )
    assert isinstance(result, tuple) and len(result) == 2 and result[1] is None


def test_fa4_lse_masked_rows_minus_inf():
    """FA4 convention: fully-masked rows get lse == -inf (opposite of FA2)."""
    torch.manual_seed(22)
    q = torch.randn(1, 64, 4, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 32, 4, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 32, 4, 64, device=DEVICE, dtype=torch.float16)
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    n_masked = 64 - 32
    assert (lse[:, :, :n_masked] == float("-inf")).all(), (
        "FA4 masked-row lse must be -inf"
    )
    assert torch.isfinite(lse[:, :, n_masked:]).all()
    assert (out[:, :n_masked] == 0).all()
    assert not torch.isnan(out).any()


def test_fa4_learnable_sink():
    dtype = torch.float16
    batch, sq, sk, h, d = 2, 97, 120, 4, 64
    torch.manual_seed(23)
    sink = torch.randn(h, dtype=torch.float32)
    q = torch.randn(batch, sq, h, d).to(dtype)
    k = torch.randn(batch, sk, h, d).to(dtype)
    v = torch.randn(batch, sk, h, d).to(dtype)
    g = torch.randn(batch, sq, h, d).to(dtype)
    (out_ref, lse_ref, dref), (out_pt, lse_pt, dpt) = _oracle(
        q, k, v, g, causal=True, learnable_sink=sink
    )
    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    out, lse = flash_attn_func(
        qm, km, vm, causal=True, learnable_sink=sink.to(DEVICE), return_lse=True
    )
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))
    detail = "fa4 sink"
    check_tensor_budget("out", out, out_ref, out_pt, dtype, detail=detail)
    check_tensor_budget("lse", lse, lse_ref, lse_pt, dtype, detail=detail)
    check_tensor_budget("dq", dq, dref[0], dpt[0], dtype, detail=detail)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
def test_fa4_varlen_packed(dtype, causal):
    torch.manual_seed(24)
    lens = [45, 1, 130, 77]
    h, d = 4, 64
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    total = sum(lens)
    q = torch.randn(total, h, d).to(dtype)
    k = torch.randn(total, h, d).to(dtype)
    v = torch.randn(total, h, d).to(dtype)
    g = torch.randn(total, h, d).to(dtype)
    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    out, lse = flash_attn_varlen_func(
        qm,
        km,
        vm,
        cu_seqlens_q=cu.to(DEVICE),
        cu_seqlens_k=cu.to(DEVICE),
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        causal=causal,
        return_lse=True,
    )
    assert out.shape == (total, h, d)
    assert lse.shape == (h, total) and lse.dtype == torch.float32
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))
    for i in range(len(lens)):
        s, e = int(cu[i]), int(cu[i + 1])
        seq = [t[s:e].unsqueeze(0).detach().clone().requires_grad_() for t in (q, k, v)]
        pt = [t.detach().clone().requires_grad_() for t in seq]
        out_ref, _, lse_ref = attention_ref(
            *seq, None, None, causal=causal, return_lse=True
        )
        out_pt, _, lse_pt = attention_ref(
            *pt,
            None,
            None,
            causal=causal,
            upcast=False,
            reorder_ops=True,
            return_lse=True,
        )
        gi = g[s:e].unsqueeze(0)
        dref = torch.autograd.grad(out_ref, seq, gi.to(out_ref.dtype))
        dpt = torch.autograd.grad(out_pt, pt, gi)
        detail = f"fa4 varlen seq{i} causal={causal}"
        check_tensor_budget(
            "out", out[s:e], out_ref[0], out_pt[0], dtype, detail=detail
        )
        check_tensor_budget(
            "lse", lse[:, s:e], lse_ref[0], lse_pt[0], dtype, detail=detail
        )
        check_tensor_budget("dq", dq[s:e], dref[0][0], dpt[0][0], dtype, detail=detail)
        check_tensor_budget("dk", dk[s:e], dref[1][0], dpt[1][0], dtype, detail=detail)
        check_tensor_budget("dv", dv[s:e], dref[2][0], dpt[2][0], dtype, detail=detail)


def test_fa4_varlen_batched_seqused():
    """4D q/k/v with seqused_q/seqused_k: rows past the used length are
    zero-filled with lse == -inf."""
    torch.manual_seed(25)
    dtype = torch.float16
    batch, sq, sk, h, d = 3, 64, 80, 4, 64
    used_q = torch.tensor([64, 30, 1], dtype=torch.int32)
    used_k = torch.tensor([80, 2, 45], dtype=torch.int32)
    q = torch.randn(batch, sq, h, d).to(dtype)
    k = torch.randn(batch, sk, h, d).to(dtype)
    v = torch.randn(batch, sk, h, d).to(dtype)
    out, lse = flash_attn_varlen_func(
        q.to(DEVICE),
        k.to(DEVICE),
        v.to(DEVICE),
        seqused_q=used_q.to(DEVICE),
        seqused_k=used_k.to(DEVICE),
        causal=True,
        return_lse=True,
    )
    assert out.shape == (batch, sq, h, d)
    assert lse.shape == (batch, h, sq)
    for i in range(batch):
        uq, uk = int(used_q[i]), int(used_k[i])
        out_ref, _ = attention_ref(
            q[i : i + 1, :uq],
            k[i : i + 1, :uk],
            v[i : i + 1, :uk],
            None,
            None,
            causal=True,
        )
        out_pt, _ = attention_ref(
            q[i : i + 1, :uq],
            k[i : i + 1, :uk],
            v[i : i + 1, :uk],
            None,
            None,
            causal=True,
            upcast=False,
            reorder_ops=True,
        )
        check_tensor_budget(
            "out",
            out[i : i + 1, :uq],
            out_ref,
            out_pt,
            dtype,
            detail=f"fa4 seqused b{i}",
        )
        assert (out[i, uq:] == 0).all(), "rows past seqused_q must be zero"
        assert (lse[i, :, uq:] == float("-inf")).all()


def test_fa4_grad_through_lse():
    """The dispatch sits above the autograd.Function, so gradients flowing in
    through the returned lse must reach q/k/v (FA4's bwd supports dlse)."""
    torch.manual_seed(26)
    q = torch.randn(1, 48, 2, 64, device=DEVICE, requires_grad=True)
    k = torch.randn(1, 64, 2, 64, device=DEVICE, requires_grad=True)
    v = torch.randn(1, 64, 2, 64, device=DEVICE, requires_grad=True)
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    (lse.float().sum() + 0.0 * out.sum()).backward()
    assert q.grad is not None and (q.grad != 0).any()
    assert k.grad is not None and (k.grad != 0).any()
    # lse does not depend on v
    assert v.grad is None or (v.grad == 0).all()


def test_fa4_window_collapse_quirk():
    """_resolve_causal_local_window: window_size_left + right < 0 collapses to
    no window at all (interface.py:334) — the MPS path must reproduce it."""
    torch.manual_seed(27)
    q = torch.randn(1, 64, 2, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 64, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 64, 2, 64, device=DEVICE, dtype=torch.float16)
    out_collapsed, _ = flash_attn_func(q, k, v, window_size=(-8, 4))  # sum < 0
    out_plain, _ = flash_attn_func(q, k, v)
    assert torch.equal(out_collapsed, out_plain)
    # ... while a genuinely negative single-sided window is NOT infinite:
    out_neg, _ = flash_attn_func(q, k, v, window_size=(None, -4))
    assert not torch.allclose(out_neg.float(), out_plain.float())


def test_fa4_perf_knobs_accepted():
    q = torch.randn(
        1, 32, 2, 64, device=DEVICE, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(1, 32, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 32, 2, 64, device=DEVICE, dtype=torch.float16)
    out, _ = flash_attn_func(
        q, k, v, causal=True, num_splits=4, pack_gqa=True, deterministic=True
    )
    out.sum().backward()
    assert q.grad is not None


def test_fa4_unsupported_raise():
    q = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    for kwargs, pattern in [
        (dict(score_mod=lambda *a: None), "score_mod"),
        (dict(mask_mod=lambda *a: None), "mask_mod"),
        (dict(qv=torch.randn_like(q)), "qv"),
        (
            dict(gather_kv_indices=torch.zeros(1, dtype=torch.int32, device=DEVICE)),
            "gather_kv_indices",
        ),
    ]:
        with pytest.raises(NotImplementedError, match=pattern):
            flash_attn_func(q, k, v, **kwargs)
    cu = torch.tensor([0, 8], dtype=torch.int32, device=DEVICE)
    with pytest.raises(NotImplementedError, match="page_table"):
        flash_attn_varlen_func(
            q[0],
            k[0],
            v[0],
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=8,
            max_seqlen_k=8,
            page_table=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE),
        )
