"""Public FA2 entry points under torch.no_grad() vs raw SDPA (Phase 3c).

Measures the cost of the FA2 seam at inference — the path third-party code
actually hits (`from flash_attn import flash_attn_func` inside a model's
forward, no grad, no return_attn_probs). Before Phase 3c the seam always
computed a `softmax_lse` nothing could read (5-7x raw SDPA); after, the
`need_lse` gate skips it (docs/apple_silicon/BENCHMARKS.md, Phase 3c).

Also times training (grad on) at the same shapes — the gate must not touch
it — and the kvcache decode shape, plus asserts that every lse-observable
path still returns a real, finite lse.

Usage:
    PYTHONPATH=. python benchmarks/mps/bench_fa2_inference.py
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "benchmarks/mps")
from common import median_iqr, sync, time_op  # noqa: E402

from flash_attn import flash_attn_func, flash_attn_with_kvcache  # noqa: E402


def bench(label, fn, **kw):
    torch.mps.empty_cache()
    times = time_op(fn, **kw)
    med, iqr = median_iqr(times)
    print(f"  {label:48s} {med * 1e3:9.2f} ms (iqr {iqr * 1e3:.2f}, n={len(times)})")
    return med


def main():
    torch.manual_seed(0)
    dtype = torch.float16
    h, d = 8, 64
    print(f"fp16 causal h={h} d={d} — public FA2 vs raw SDPA")

    for batch, seqlen in [(8, 2048), (2, 8192)]:
        print(f"\n== {batch}x{seqlen} ==")
        q = torch.randn(batch, seqlen, h, d, device="mps", dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))

        t_sdpa = bench(
            "raw F.scaled_dot_product_attention",
            lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True),
        )

        def fa2_infer():
            with torch.no_grad():
                flash_attn_func(q, k, v, causal=True)

        t_fa2 = bench("flash_attn_func under no_grad", fa2_infer)
        print(f"  {'ratio (target ~1.2x)':48s} {t_fa2 / t_sdpa:9.2f} x")

        qg = q.clone().requires_grad_()
        kg = k.clone().requires_grad_()
        vg = v.clone().requires_grad_()

        def fa2_train():
            out = flash_attn_func(qg, kg, vg, causal=True)
            out.sum().backward()
            qg.grad = kg.grad = vg.grad = None

        bench("flash_attn_func fwd+bwd (grad on)", fa2_train, warmup=2, iters=6)

    # The lse-observable paths must still produce a real, finite lse.
    q = torch.randn(2, 512, h, d, device="mps", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    with torch.no_grad():
        _, lse, _ = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
        assert lse.shape == (2, h, 512) and bool(torch.isfinite(lse).all())
        _, lse_kv = flash_attn_with_kvcache(q, k, v, causal=True, return_softmax_lse=True)
        assert lse_kv.shape == (2, h, 512) and bool(torch.isfinite(lse_kv).all())
    print("\nlse-observable paths return real, finite lse: OK")


if __name__ == "__main__":
    sync()
    main()
