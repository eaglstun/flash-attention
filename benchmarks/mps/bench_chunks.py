"""Sweep the Phase 1 core's chunk sizes (the cheap tuning win).

- forward: ``kv_chunk_size`` (default 1024 in flash_attn/mps/core.py)
- backward: ``q_chunk_size`` (default 256)

Both are exposed as kwargs on ``mps_flash_attn_func``. Larger chunks = fewer
dispatches but more peak memory; this measures the latency/memory tradeoff on
representative training shapes.

Usage: PYTHONPATH=. python benchmarks/mps/bench_chunks.py [--seqlens 4096,16384]
"""

import argparse
import sys

import torch

sys.path.insert(0, "benchmarks/mps")
from common import TorchMemHighWater, make_qkv, median_iqr, sync, time_op  # noqa: E402

from flash_attn.mps.core import mps_flash_attn_func  # noqa: E402


def make_fwd(q, k, v, causal, kv_chunk):
    def fwd():
        with torch.no_grad():
            mps_flash_attn_func(q, k, v, causal=causal, kv_chunk_size=kv_chunk)

    return fwd


def make_fwd_bwd(q, k, v, dout, causal, q_chunk):
    def fwd_bwd():
        q.grad = k.grad = v.grad = None
        out, _ = mps_flash_attn_func(q, k, v, causal=causal, q_chunk_size=q_chunk)
        out.backward(dout)

    return fwd_bwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlens", default="4096,16384")
    p.add_argument("--head_dims", default="64,128")
    args = p.parse_args()
    seqlens = [int(x) for x in args.seqlens.split(",")]
    head_dims = [int(x) for x in args.head_dims.split(",")]

    hq = hk = 8
    dtype = torch.float16
    causal = True

    for s in seqlens:
        b = max(1, 16384 // s)
        for hd in head_dims:
            print(f"\n== fwd kv_chunk_size sweep: b={b} s={s} hd={hd} causal fp16 ==")
            q, k, v = make_qkv(b, s, hq, hk, hd, dtype)
            for kv_chunk in (256, 512, 1024, 2048, 4096, s):
                if kv_chunk > s:
                    continue
                fwd = make_fwd(q, k, v, causal, kv_chunk)
                sync()
                with TorchMemHighWater() as mem:
                    times = time_op(fwd, budget_s=20)
                med, iqr = median_iqr(times)
                print(
                    f"  kv_chunk {kv_chunk:6d}: {med * 1e3:9.2f} ms (iqr {iqr * 1e3:.2f}) "
                    f"peak +{mem.peak_extra / 2**20:7.0f} MiB",
                    flush=True,
                )
            del q, k, v
            torch.mps.empty_cache()

            print(f"== bwd q_chunk_size sweep: b={b} s={s} hd={hd} causal fp16 ==")
            for q_chunk in (128, 256, 512, 1024, 2048):
                if q_chunk > s:
                    continue
                qg, kg, vg = make_qkv(b, s, hq, hk, hd, dtype, requires_grad=True)
                dout = torch.randn_like(qg)
                fwd_bwd = make_fwd_bwd(qg, kg, vg, dout, causal, q_chunk)
                sync()
                with TorchMemHighWater() as mem:
                    times = time_op(fwd_bwd, warmup=2, iters=5, budget_s=40)
                med, iqr = median_iqr(times)
                print(
                    f"  q_chunk {q_chunk:6d}: {med * 1e3:9.2f} ms (iqr {iqr * 1e3:.2f}) "
                    f"peak +{mem.peak_extra / 2**20:7.0f} MiB",
                    flush=True,
                )
                del qg, kg, vg, dout
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
