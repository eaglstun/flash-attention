"""The (formerly) pathological paths: varlen and kvcache decode, quantified
against raw batched-SDPA upper bounds.

Phase 1 shipped these as per-sequence/per-batch Python loops; Phase 3a
batched them (docs/apple_silicon/BENCHMARKS.md). This script measures the
public entry points (whatever strategy the driver picks) against the same
alternatives as the Phase 2 survey, so before/after is apples-to-apples.

Usage:
    PYTHONPATH=. python benchmarks/mps/bench_varlen.py           # main table
    PYTHONPATH=. python benchmarks/mps/bench_varlen.py --skew    # heuristic sweep
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "benchmarks/mps")
from common import median_iqr, sync, time_op  # noqa: E402

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: E402
from flash_attn.mps.varlen import mps_flash_attn_varlen  # noqa: E402


def bench(label, fn, **kw):
    times = time_op(fn, **kw)
    med, iqr = median_iqr(times)
    print(f"  {label:56s} {med * 1e3:9.2f} ms (iqr {iqr * 1e3:.2f}, n={len(times)})")
    return med


def varlen_bench():
    torch.manual_seed(0)
    nseq, hq, hk, hd = 32, 8, 8, 64
    lens = torch.randint(64, 1025, (nseq,))
    total = int(lens.sum())
    max_len = int(lens.max())
    cu = torch.zeros(nseq + 1, dtype=torch.int32)
    cu[1:] = lens.cumsum(0)
    cu_mps = cu.to("mps")
    print(
        f"\n== varlen: {nseq} packed seqs, lens 64..1024 (total={total}, max={max_len}), "
        f"h{hq} d{hd} fp16 causal =="
    )
    q = torch.randn(
        total, hq, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        total, hk, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        total, hk, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    dout = torch.randn_like(q)

    def varlen_fwd():
        with torch.no_grad():
            flash_attn_varlen_func(
                q, k, v, cu_mps, cu_mps, max_len, max_len, causal=True
            )

    bench("flash_attn_varlen_func fwd (auto strategy)", varlen_fwd, budget_s=20)

    def varlen_fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = flash_attn_varlen_func(
            q, k, v, cu_mps, cu_mps, max_len, max_len, causal=True
        )
        out.backward(dout)

    bench(
        "flash_attn_varlen_func fwd+bwd (auto strategy)",
        varlen_fwd_bwd,
        warmup=2,
        iters=5,
        budget_s=60,
    )

    def varlen_fwd_nolse():
        with torch.no_grad():
            mps_flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_mps,
                cu_seqlens_k=cu_mps,
                causal=True,
                return_lse=False,
            )

    bench(
        "varlen driver fwd, no lse (FA4 return_lse=False)",
        varlen_fwd_nolse,
        budget_s=20,
    )

    # padded-dense alternatives (upper bound on wasted FLOPs, one launch)
    qp = torch.zeros(nseq, max_len, hq, hd, device="mps", dtype=torch.float16)
    kp = torch.zeros(nseq, max_len, hk, hd, device="mps", dtype=torch.float16)
    vp = torch.zeros(nseq, max_len, hk, hd, device="mps", dtype=torch.float16)
    with torch.no_grad():
        for i in range(nseq):
            s0, s1 = int(cu[i]), int(cu[i + 1])
            qp[i, : s1 - s0] = q[s0:s1]
            kp[i, : s1 - s0] = k[s0:s1]
            vp[i, : s1 - s0] = v[s0:s1]
    qp.requires_grad_(), kp.requires_grad_(), vp.requires_grad_()
    doutp = torch.randn_like(qp)

    def padded_core_fwd():
        with torch.no_grad():
            flash_attn_func(qp, kp, vp, causal=True)

    bench(
        "padded dense core fwd (1 launch, wasted pad FLOPs)",
        padded_core_fwd,
        budget_s=20,
    )

    def padded_sdpa_fwd():
        with torch.no_grad():
            F.scaled_dot_product_attention(
                qp.transpose(1, 2),
                kp.transpose(1, 2),
                vp.transpose(1, 2),
                is_causal=True,
            )

    bench("padded dense torch-SDPA fwd", padded_sdpa_fwd, budget_s=20)

    def padded_sdpa_fwd_bwd():
        qp.grad = kp.grad = vp.grad = None
        o = F.scaled_dot_product_attention(
            qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        o.backward(doutp)

    bench("padded dense torch-SDPA fwd+bwd", padded_sdpa_fwd_bwd, budget_s=20)


def decode_bench():
    torch.manual_seed(0)
    b, cache_len, hq, hk, hd = 32, 4096, 8, 2, 128
    print(f"\n== kvcache decode: b={b} sq=1 cache={cache_len} h{hq}/{hk} d{hd} fp16 ==")
    q = torch.randn(b, 1, hq, hd, device="mps", dtype=torch.float16)
    kc = torch.randn(b, cache_len, hk, hd, device="mps", dtype=torch.float16)
    vc = torch.randn(b, cache_len, hk, hd, device="mps", dtype=torch.float16)
    cache_seqlens = torch.full((b,), cache_len, dtype=torch.int32, device="mps")

    def kvcache():
        flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens, causal=True)

    bench("flash_attn_with_kvcache (batched)", kvcache, budget_s=20)

    def sdpa_decode():
        with torch.no_grad():
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                kc.transpose(1, 2),
                vc.transpose(1, 2),
                enable_gqa=True,
            )

    bench("batched torch-SDPA decode (1 launch)", sdpa_decode, budget_s=20)

    # ragged cache lengths (the case the batched masking has to earn)
    torch.manual_seed(1)
    ragged = torch.randint(64, cache_len, (b,), dtype=torch.int32, device="mps")

    def kvcache_ragged():
        flash_attn_with_kvcache(q, kc, vc, cache_seqlens=ragged, causal=True)

    bench("flash_attn_with_kvcache, ragged cache_seqlens", kvcache_ragged, budget_s=20)


def skew_bench():
    """Batched-vs-looped sweep that calibrates the varlen strategy heuristic
    (flash_attn/mps/varlen.py constants). Prints both forced strategies plus
    what the heuristic actually picks."""
    import random

    random.seed(0)
    cases = {
        "32 x 64..1024 (ragged)": [random.randint(64, 1024) for _ in range(32)],
        "64 x 128 (uniform small)": [128] * 64,
        "256 x 32 (launch-bound)": [32] * 256,
        "8 x 2048 (uniform large)": [2048] * 8,
        "skew 1x4096 + 31x128": [4096] + [128] * 31,
        "skew 1x8192 + 63x64": [8192] + [64] * 63,
    }
    for label, lens in cases.items():
        lens_t = torch.tensor(lens)
        total = int(lens_t.sum())
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = lens_t.cumsum(0)
        cu_mps = cu.to("mps")
        torch.manual_seed(0)
        q = torch.randn(total, 8, 64, device="mps", dtype=torch.float16)
        k = torch.randn(total, 8, 64, device="mps", dtype=torch.float16)
        v = torch.randn(total, 8, 64, device="mps", dtype=torch.float16)
        print(f"\n== {label} (total={total}) ==")
        for lse in (True, False):
            for forced, name in (
                (True, "batched"),
                (False, "looped "),
                (None, "auto   "),
            ):

                def run():
                    with torch.no_grad():
                        mps_flash_attn_varlen(
                            q,
                            k,
                            v,
                            cu_seqlens_q=cu_mps,
                            cu_seqlens_k=cu_mps,
                            causal=True,
                            return_lse=lse,
                            _force_batched=forced,
                        )

                bench(f"{'lse  ' if lse else 'nolse'} {name}", run, budget_s=25)


if __name__ == "__main__":
    sync()
    if "--skew" in sys.argv:
        skew_bench()
    else:
        varlen_bench()
        decode_bench()
