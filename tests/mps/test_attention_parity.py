"""Parity harness for the MPS attention core (Phase 1a of the Apple Silicon port).

CPU is the oracle, MPS is the defendant. The oracle is
``flash_attn/cute/testing.py::attention_ref`` (pure torch, fp32-upcasting) run
on CPU; ``*_pt`` is the same oracle in the raw test dtype (``upcast=False,
reorder_ops=True``); the defendant is ``flash_attn.mps.core`` run on the
parametrized device. Error budgets are the cute suite's own
(tests/cute/test_flash_attn.py:375), applied to out, lse, dq, dk, dv.
"""

import math

import pytest
import torch

from checks import check_tensor_budget
from flash_attn.cute.testing import attention_ref
from flash_attn.mps.core import _attention_forward

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# "eager" = the plain differentiable reference core (Phase 1b commit 1).
IMPLS = ["eager"]


def run_attention(impl, q, k, v, **kwargs):
    if impl == "eager":
        return _attention_forward(q, k, v, **kwargs)
    raise ValueError(f"unknown impl {impl}")


def _alibi_bias_for_ref(slopes, seqlen_q, seqlen_k):
    """The oracle-side ALiBi bias: -slope * |i + seqlen_k - seqlen_q - j|.

    Same as tests/test_flash_attn.py::attn_bias_from_alibi_slopes (non-causal
    branch); for causal attention every *visible* position has the same value,
    so it is also the causal oracle bias.
    """
    row_idx = torch.arange(seqlen_q, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)
    if slopes.dim() == 1:
        slopes = slopes.view(1, -1, 1, 1)
    else:
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    return -slopes * relative_pos.to(dtype=slopes.dtype)


def _parity_case(
    device,
    dtype,
    impl,
    *,
    batch=4,
    seqlen_q,
    seqlen_k,
    nheads=8,
    nheads_kv=8,
    d=64,
    causal=False,
    window_size=(None, None),
    softcap=0.0,
    alibi_slopes=None,
    learnable_sink=None,
    check_grads=True,
    seed=0,
    detail="",
):
    torch.manual_seed(seed)
    q_pt = torch.randn(batch, seqlen_q, nheads, d).to(dtype).requires_grad_()
    k_pt = torch.randn(batch, seqlen_k, nheads_kv, d).to(dtype).requires_grad_()
    v_pt = torch.randn(batch, seqlen_k, nheads_kv, d).to(dtype).requires_grad_()
    attn_bias = (
        _alibi_bias_for_ref(alibi_slopes, seqlen_q, seqlen_k)
        if alibi_slopes is not None
        else None
    )
    ref_kwargs = dict(
        attn_bias=attn_bias,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        learnable_sink=learnable_sink,
        return_lse=True,
    )
    # The oracle runs one precision level above the test dtype. For fp16/bf16
    # that is the suite's usual fp32-upcast oracle. For fp32 the "same math in
    # the test dtype" baseline would be numerically identical to the oracle
    # (zero budget), so the oracle is lifted to fp64 and plain fp32 torch math
    # becomes the in-dtype baseline — same assertion, one level up.
    if dtype == torch.float32:
        q_ref = q_pt.detach().double().requires_grad_()
        k_ref = k_pt.detach().double().requires_grad_()
        v_ref = v_pt.detach().double().requires_grad_()
        out_ref, _, lse_ref = attention_ref(
            q_ref, k_ref, v_ref, None, None, upcast=False, **ref_kwargs
        )
    else:
        q_ref, k_ref, v_ref = q_pt, k_pt, v_pt
        out_ref, _, lse_ref = attention_ref(
            q_ref, k_ref, v_ref, None, None, **ref_kwargs
        )
    out_pt, _, lse_pt = attention_ref(
        q_pt, k_pt, v_pt, None, None, upcast=False, reorder_ops=True, **ref_kwargs
    )

    q = q_pt.detach().to(device).requires_grad_()
    k = k_pt.detach().to(device).requires_grad_()
    v = v_pt.detach().to(device).requires_grad_()
    out, lse = run_attention(
        impl,
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes.to(device) if alibi_slopes is not None else None,
        learnable_sink=learnable_sink.to(device)
        if learnable_sink is not None
        else None,
    )
    assert out.dtype == dtype
    assert out.shape == (batch, seqlen_q, nheads, d)
    assert lse.dtype == torch.float32, "lse must be fp32 (FA convention)"
    assert lse.shape == (batch, nheads, seqlen_q), (
        "lse must be (batch, nheads, seqlen_q)"
    )

    rtol = 2.0 if softcap == 0.0 else 3.0
    detail = f"{detail} dev={device} impl={impl}"
    check_tensor_budget("out", out, out_ref, out_pt, dtype, rtol=rtol, detail=detail)
    check_tensor_budget("lse", lse, lse_ref, lse_pt, dtype, rtol=rtol, detail=detail)

    if not check_grads:
        return
    g = torch.randn(out_pt.shape).to(dtype)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g.to(device))
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), g.to(out_ref.dtype)
    )
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), g)
    extra_atol = 0.0 if softcap == 0.0 else 3e-4
    check_tensor_budget(
        "dq", dq, dq_ref, dq_pt, dtype, rtol=rtol, extra_atol=extra_atol, detail=detail
    )
    check_tensor_budget(
        "dk", dk, dk_ref, dk_pt, dtype, rtol=rtol, extra_atol=extra_atol, detail=detail
    )
    check_tensor_budget(
        "dv", dv, dv_ref, dv_pt, dtype, rtol=rtol, extra_atol=extra_atol, detail=detail
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_kv", [(8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k", [(1, 1), (113, 203), (512, 512), (1024, 1024)]
)
def test_attention_output(
    device, impl, dtype, causal, nheads, nheads_kv, d, seqlen_q, seqlen_k
):
    _parity_case(
        device,
        dtype,
        impl,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        nheads_kv=nheads_kv,
        d=d,
        causal=causal,
        seed=hash((causal, nheads_kv, d, seqlen_q, seqlen_k)) % (2**31),
        detail=f"causal={causal} h={nheads}/{nheads_kv} d={d} sq={seqlen_q} sk={seqlen_k}",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
def test_softcap(device, impl, dtype, causal):
    _parity_case(
        device,
        dtype,
        impl,
        seqlen_q=239,
        seqlen_k=283,
        nheads=8,
        nheads_kv=2,
        d=64,
        causal=causal,
        softcap=15.0,
        seed=1,
        detail=f"softcap causal={causal}",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "window_size,causal",
    [
        ((16, 0), False),
        ((37, 53), False),
        ((None, 12), False),
        ((21, None), False),
        ((64, None), True),  # causal forces (64, 0)
    ],
)
def test_local_window(device, impl, dtype, window_size, causal):
    _parity_case(
        device,
        dtype,
        impl,
        seqlen_q=247,
        seqlen_k=311,
        causal=causal,
        window_size=window_size,
        seed=2,
        detail=f"window={window_size} causal={causal}",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("slopes_shape", ["h", "bh"])
def test_alibi(device, impl, dtype, causal, slopes_shape):
    torch.manual_seed(3)
    batch, nheads = 4, 8
    shape = (nheads,) if slopes_shape == "h" else (batch, nheads)
    alibi_slopes = torch.rand(*shape, dtype=torch.float32) * 0.3
    _parity_case(
        device,
        dtype,
        impl,
        batch=batch,
        seqlen_q=161,
        seqlen_k=208,
        nheads=nheads,
        causal=causal,
        alibi_slopes=alibi_slopes,
        seed=3,
        detail=f"alibi[{slopes_shape}] causal={causal}",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
def test_learnable_sink(device, impl, dtype, causal):
    torch.manual_seed(4)
    learnable_sink = torch.randn(8, dtype=torch.float32)
    _parity_case(
        device,
        dtype,
        impl,
        seqlen_q=113,
        seqlen_k=203,
        causal=causal,
        learnable_sink=learnable_sink,
        seed=4,
        detail=f"sink causal={causal}",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "case",
    [
        # causal with seqlen_q > seqlen_k: rows [0, seqlen_q - seqlen_k) see nothing
        dict(seqlen_q=128, seqlen_k=64, causal=True, window_size=(None, None)),
        # negative right window: rows [0, 5) see nothing
        dict(seqlen_q=97, seqlen_k=97, causal=False, window_size=(None, -5)),
    ],
    ids=["causal_q_gt_k", "negative_right_window"],
)
def test_fully_masked_rows_no_nan(device, impl, dtype, case):
    """Fully-masked rows must yield out == 0 and lse == -inf, never NaN,
    forward AND backward. The oracle's own backward NaNs out on these rows
    (the cute suite skips them), so grads are checked for finiteness, for
    exact zeros on the masked rows, and for CPU/MPS agreement of our own core.
    """
    torch.manual_seed(5)
    batch, nheads, d = 2, 4, 64
    sq, sk, causal, window_size = (
        case["seqlen_q"],
        case["seqlen_k"],
        case["causal"],
        case["window_size"],
    )
    if causal:
        n_masked = sq - sk  # rows 0..n_masked-1 fully masked (bottom-right aligned)
    else:
        n_masked = -(sk - sq + window_size[1])  # col <= row + sk - sq + wr < 0
    assert n_masked > 0

    q_cpu = torch.randn(batch, sq, nheads, d).to(dtype)
    k_cpu = torch.randn(batch, sk, nheads, d).to(dtype)
    v_cpu = torch.randn(batch, sk, nheads, d).to(dtype)
    g_cpu = torch.randn(batch, sq, nheads, d).to(dtype)

    def run(device):
        q = q_cpu.detach().to(device).requires_grad_()
        k = k_cpu.detach().to(device).requires_grad_()
        v = v_cpu.detach().to(device).requires_grad_()
        out, lse = run_attention(impl, q, k, v, causal=causal, window_size=window_size)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g_cpu.to(device))
        return out, lse, dq, dk, dv

    out, lse, dq, dk, dv = run(device)
    for name, t in [("out", out), ("lse", lse), ("dq", dq), ("dk", dk), ("dv", dv)]:
        assert not torch.isnan(t).any(), f"{name} has NaN"
    assert torch.isfinite(out).all()
    assert (out[:, :n_masked] == 0).all(), "fully-masked rows must produce out == 0"
    assert (lse[:, :, :n_masked] == float("-inf")).all(), (
        "fully-masked rows must produce lse == -inf (FA4 CuTe convention)"
    )
    assert torch.isfinite(lse[:, :, n_masked:]).all()
    assert (dq[:, :n_masked] == 0).all(), "fully-masked rows must get zero dq"
    assert (
        not torch.isinf(dq).any()
        and not torch.isinf(dk).any()
        and not torch.isinf(dv).any()
    )

    # Forward parity against the oracle still holds (it zero-fills masked rows).
    out_ref, _, lse_ref = attention_ref(
        q_cpu.float(),
        k_cpu.float(),
        v_cpu.float(),
        None,
        None,
        causal=causal,
        window_size=window_size,
        return_lse=True,
    )
    # out is stored in the test dtype, so allow its own rounding on top of the
    # fp32-math oracle: one-and-a-bit ulp at |out| ~ O(1).
    fwd_atol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    assert torch.allclose(out.detach().float().cpu(), out_ref, atol=fwd_atol)

    # Cross-device self-consistency of the core (CPU run of the same impl).
    if device != "cpu":
        out_c, lse_c, dq_c, dk_c, dv_c = run("cpu")
        torch.testing.assert_close(out.cpu(), out_c)
        torch.testing.assert_close(lse.cpu(), lse_c, equal_nan=False)
        torch.testing.assert_close(dq.cpu(), dq_c)
        torch.testing.assert_close(dk.cpu(), dk_c)
        torch.testing.assert_close(dv.cpu(), dv_c)

    # A learnable sink rescues fully-masked rows: lse becomes the sink value.
    sink = torch.randn(nheads, dtype=torch.float32)
    out_s, lse_s = run_attention(
        impl,
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        causal=causal,
        window_size=window_size,
        learnable_sink=sink.to(device),
    )
    assert not torch.isnan(out_s).any() and not torch.isnan(lse_s).any()
    assert (out_s[:, :n_masked] == 0).all()
    expected = sink.view(1, nheads, 1).expand(batch, nheads, n_masked)
    torch.testing.assert_close(lse_s[:, :, :n_masked].cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp16", "bf16"])
def test_custom_softmax_scale(device, impl, dtype):
    """softmax_scale=s must equal the oracle on q pre-scaled by s * sqrt(d)
    (the oracle hardcodes 1/sqrt(d)). Forward-only."""
    torch.manual_seed(6)
    batch, sq, sk, nheads, d = 4, 128, 160, 8, 64
    # scale * sqrt(d) == 2.0 exactly, so pre-scaling q for the oracle is lossless
    scale = 0.25
    q_ref = torch.randn(batch, sq, nheads, d).to(dtype)
    k_ref = torch.randn(batch, sk, nheads, d).to(dtype)
    v_ref = torch.randn(batch, sk, nheads, d).to(dtype)
    q_scaled = (q_ref.float() * (scale * math.sqrt(d))).to(dtype)
    if dtype == torch.float32:
        out_ref, _, lse_ref = attention_ref(
            q_scaled.double(),
            k_ref.double(),
            v_ref.double(),
            None,
            None,
            causal=True,
            upcast=False,
            return_lse=True,
        )
    else:
        out_ref, _, lse_ref = attention_ref(
            q_scaled, k_ref, v_ref, None, None, causal=True, return_lse=True
        )
    out_pt, _, lse_pt = attention_ref(
        q_scaled,
        k_ref,
        v_ref,
        None,
        None,
        causal=True,
        upcast=False,
        reorder_ops=True,
        return_lse=True,
    )
    out2, lse2 = run_attention(
        impl,
        q_ref.to(device),
        k_ref.to(device),
        v_ref.to(device),
        softmax_scale=scale,
        causal=True,
    )
    detail = f"custom_scale dev={device} impl={impl}"
    check_tensor_budget("out", out2, out_ref, out_pt, dtype, detail=detail)
    check_tensor_budget("lse", lse2, lse_ref, lse_pt, dtype, detail=detail)


@pytest.mark.parametrize("impl", IMPLS)
def test_lse_convention_explicit(impl):
    """Pin the exact lse convention: natural-log logsumexp of the scaled,
    masked scores, fp32, (batch, nheads, seqlen_q)."""
    torch.manual_seed(7)
    batch, sq, sk, nheads, d = 2, 33, 47, 3, 32
    q = torch.randn(batch, sq, nheads, d)
    k = torch.randn(batch, sk, nheads, d)
    v = torch.randn(batch, sk, nheads, d)
    _, lse = run_attention(impl, q, k, v, causal=True)
    scores = torch.einsum("bthd,bshd->bhts", q, k) * d ** (-0.5)
    row = torch.arange(sq).unsqueeze(-1)
    col = torch.arange(sk)
    scores = scores.masked_fill((col > row + sk - sq).view(1, 1, sq, sk), float("-inf"))
    manual = torch.logsumexp(scores.double(), dim=-1).float()
    assert lse.shape == (batch, nheads, sq)
    assert lse.dtype == torch.float32
    torch.testing.assert_close(lse, manual, rtol=1e-5, atol=1e-5)
