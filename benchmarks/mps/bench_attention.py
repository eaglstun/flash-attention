"""Phase 2 main sweep: dense attention candidates on MPS.

Candidates (all take/return torch (b, s, h, d) tensors unless noted):

- ``core``        -- the Phase 1 backend as shipped: ``mps_flash_attn_func``
                     with ``return_lse=True`` (the path the FA2 seam always
                     takes: chunked online-softmax fwd, recompute bwd).
- ``sdpa``        -- plain ``F.scaled_dot_product_attention`` on MPS
                     (``enable_gqa=True``). No lse. The torch-level ceiling.
- ``mlx_bridge``  -- MLX ``mx.fast.scaled_dot_product_attention`` with the
                     torch<->MLX DLPack bridge paid on every call (transpose +
                     contiguous + sync + from_dlpack in; from_dlpack out).
- ``mlx_native``  -- same MLX kernel with arrays already resident in MLX
                     (kernel-only cost; the "whole model lives in MLX" number).

Every candidate passes the CPU-oracle correctness gate (outputs, and for
fwd_bwd also dq/dk/dv) before any timing happens. See common.py for the sync
and memory discipline.

Usage:
    python benchmarks/mps/bench_attention.py --mode fwd --out fwd.csv
    python benchmarks/mps/bench_attention.py --mode fwd_bwd --out fwd_bwd.csv
    # subset flags: --candidates core,sdpa --seqlens 512,2048 --dtypes float16
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "benchmarks/mps")
from common import (  # noqa: E402
    CsvWriter,
    Row,
    TorchMemHighWater,
    attention_flops,
    check_against_oracle,
    make_qkv,
    median_iqr,
    sync,
    time_op,
)

from flash_attn.mps.core import mps_flash_attn_func  # noqa: E402

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

MX_DTYPE = {"float16": "float16", "bfloat16": "bfloat16"}


# ---------------------------------------------------------------------------
# candidates: each returns (fwd_closure, out_grabber) or (fwdbwd_closure, grads_grabber)
# ---------------------------------------------------------------------------


class Core:
    name = "core"

    def __init__(self, q, k, v, causal, mode):
        self.causal = causal
        self.mode = mode
        if mode == "fwd":
            self.q, self.k, self.v = q, k, v
        else:
            self.q = q.detach().requires_grad_()
            self.k = k.detach().requires_grad_()
            self.v = v.detach().requires_grad_()
            self.dout = torch.randn_like(q)
        self.out = None

    def fwd(self):
        with torch.no_grad():
            self.out, _ = mps_flash_attn_func(
                self.q, self.k, self.v, causal=self.causal
            )

    def fwd_bwd(self):
        self.q.grad = self.k.grad = self.v.grad = None
        out, _ = mps_flash_attn_func(self.q, self.k, self.v, causal=self.causal)
        out.backward(self.dout)
        self.out = out

    def grads(self):
        return self.q.grad, self.k.grad, self.v.grad


class Sdpa:
    name = "sdpa"

    def __init__(self, q, k, v, causal, mode):
        self.causal = causal
        self.mode = mode
        if mode == "fwd":
            self.q, self.k, self.v = q, k, v
        else:
            self.q = q.detach().requires_grad_()
            self.k = k.detach().requires_grad_()
            self.v = v.detach().requires_grad_()
            self.dout = torch.randn_like(q)
        self.out = None

    def _sdpa(self, q, k, v):
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
            enable_gqa=q.shape[2] != k.shape[2],
        )
        return o.transpose(1, 2)

    def fwd(self):
        with torch.no_grad():
            self.out = self._sdpa(self.q, self.k, self.v)

    def fwd_bwd(self):
        self.q.grad = self.k.grad = self.v.grad = None
        out = self._sdpa(self.q, self.k, self.v)
        out.backward(self.dout)
        self.out = out

    def grads(self):
        return self.q.grad, self.k.grad, self.v.grad


class MlxBase:
    """Shared MLX pieces. (b,s,h,d) torch <-> (b,h,s,d) MLX."""

    def __init__(self, q, k, v, causal, mode):
        self.causal = causal
        self.mode = mode
        self.scale = q.shape[-1] ** -0.5
        self.q, self.k, self.v = q, k, v
        self.dout_t = torch.randn_like(q) if mode == "fwd_bwd" else None
        self.out = None
        self._grads = None
        mask = "causal" if causal else None
        scale = self.scale

        def fwd_fn(qm, km, vm):
            return mx.fast.scaled_dot_product_attention(
                qm, km, vm, scale=scale, mask=mask
            )

        self.fwd_fn = fwd_fn

        def loss_fn(qm, km, vm, doutm):
            o = fwd_fn(qm, km, vm)
            return (o.astype(mx.float32) * doutm.astype(mx.float32)).sum()

        self.grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))

    @staticmethod
    def to_mx(t):
        # (b,s,h,d) -> contiguous (b,h,s,d), then zero-copy DLPack import.
        # The sync between the torch-side copy and the MLX read is REQUIRED:
        # the buffer is shared (zero-copy), and torch's MPS stream and MLX's
        # stream are not ordered with respect to each other. Without it MLX
        # reads garbage -- caught by the correctness gate.
        tc = t.detach().transpose(1, 2).contiguous()
        sync()
        return mx.from_dlpack(tc)

    @staticmethod
    def to_torch(a):
        # zero-copy; (b,h,s,d) -> (b,s,h,d) view.
        return torch.from_dlpack(a).transpose(1, 2)


class MlxBridge(MlxBase):
    name = "mlx_bridge"

    def fwd(self):
        # the bridge is part of the timed cost, including the torch-side sync
        # a real integration needs before MLX touches shared buffers.
        qc = self.q.transpose(1, 2).contiguous()
        kc = self.k.transpose(1, 2).contiguous()
        vc = self.v.transpose(1, 2).contiguous()
        sync()
        o = self.fwd_fn(mx.from_dlpack(qc), mx.from_dlpack(kc), mx.from_dlpack(vc))
        mx.eval(o)
        self.out = self.to_torch(o)

    def fwd_bwd(self):
        qc = self.q.transpose(1, 2).contiguous()
        kc = self.k.transpose(1, 2).contiguous()
        vc = self.v.transpose(1, 2).contiguous()
        dc = self.dout_t.transpose(1, 2).contiguous()
        sync()
        qm, km, vm, dm = (mx.from_dlpack(x) for x in (qc, kc, vc, dc))
        o = self.fwd_fn(qm, km, vm)
        gq, gk, gv = self.grad_fn(qm, km, vm, dm)
        mx.eval(o, gq, gk, gv)
        self.out = self.to_torch(o)
        self._grads = tuple(self.to_torch(g) for g in (gq, gk, gv))

    def grads(self):
        return self._grads


class MlxNative(MlxBase):
    name = "mlx_native"

    def __init__(self, q, k, v, causal, mode):
        super().__init__(q, k, v, causal, mode)
        sync()
        self.qm, self.km, self.vm = self.to_mx(q), self.to_mx(k), self.to_mx(v)
        self.dm = self.to_mx(self.dout_t) if mode == "fwd_bwd" else None
        mx.eval(self.qm, self.km, self.vm, *([self.dm] if self.dm is not None else []))
        self.om = None
        self.gm = None

    def fwd(self):
        self.om = self.fwd_fn(self.qm, self.km, self.vm)
        mx.eval(self.om)

    def fwd_bwd(self):
        self.om = self.fwd_fn(self.qm, self.km, self.vm)
        self.gm = self.grad_fn(self.qm, self.km, self.vm, self.dm)
        mx.eval(self.om, *self.gm)

    def finalize(self):
        self.out = self.to_torch(self.om)
        if self.gm is not None:
            self._grads = tuple(self.to_torch(g) for g in self.gm)

    def grads(self):
        return self._grads


CANDIDATES = {
    "core": Core,
    "sdpa": Sdpa,
    "mlx_bridge": MlxBridge,
    "mlx_native": MlxNative,
}


# ---------------------------------------------------------------------------
# correctness gates (run once per candidate x flags x dtype, small shape)
# ---------------------------------------------------------------------------


def grad_oracle(q, k, v, causal, upcast=True):
    dt = torch.float32 if upcast else q.dtype
    qc = q.cpu().to(dt).detach().requires_grad_()
    kc = k.cpu().to(dt).detach().requires_grad_()
    vc = v.cpu().to(dt).detach().requires_grad_()
    from common import attention_oracle  # noqa: F401

    hq, hk = qc.shape[2], kc.shape[2]
    kf = kc.repeat_interleave(hq // hk, dim=2) if hq != hk else kc
    vf = vc.repeat_interleave(hq // hk, dim=2) if hq != hk else vc
    scale = qc.shape[-1] ** -0.5
    scores = torch.einsum("bthd,bshd->bhts", qc * scale, kf)
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.einsum(
        "bhts,bshd->bthd", torch.softmax(scores.float(), dim=-1).to(dt), vf
    )
    return out, (qc, kc, vc)


def gate_candidate(cls, dtype, causal, hq, hk, hd, mode):
    q, k, v = make_qkv(2, 512, hq, hk, hd, dtype, seed=7)
    cand = cls(q, k, v, causal, mode)
    if mode == "fwd":
        cand.fwd()
    else:
        cand.fwd_bwd()
    if hasattr(cand, "finalize"):
        cand.finalize()
    sync()
    ratio = check_against_oracle(cand.out.detach(), q, k, v, causal=causal)
    ratios = [ratio]
    if mode == "fwd_bwd":
        dout = cand.dout if hasattr(cand, "dout") else cand.dout_t
        for up in (True, False):
            out_o, leaves = grad_oracle(q, k, v, causal, upcast=up)
            out_o.backward(dout.cpu().to(out_o.dtype))
            if up:
                ref_g = [t.grad.float() for t in leaves]
            else:
                pt_g = [t.grad.float() for t in leaves]
        for g_cand, g_ref, g_pt, nm in zip(cand.grads(), ref_g, pt_g, "qkv"):
            err = (g_cand.detach().cpu().float() - g_ref).abs().max().item()
            err_pt = (g_pt - g_ref).abs().max().item()
            assert err <= 2.0 * err_pt + 1e-4, (
                f"grad gate FAILED d{nm}: err={err:.3e} budget=2*{err_pt:.3e}"
            )
            ratios.append(err / max(err_pt, 1e-30))
    return max(ratios)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

# constant token count b*s = 16384 so throughput is comparable across rows
SHAPES = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def run(args):
    writer = CsvWriter(args.out)
    names = args.candidates.split(",")
    seqlens = [int(s) for s in args.seqlens.split(",")]
    dtypes = args.dtypes.split(",")
    gqa_configs = [(8, 8), (8, 2)]
    head_dims = [64, 128]

    # -- gates first --------------------------------------------------------
    print(
        "== correctness gates (out%s) ==" % ("+grads" if args.mode == "fwd_bwd" else "")
    )
    gated = {}
    for name in names:
        cls = CANDIDATES[name]
        ok = True
        for ds in dtypes:
            for causal in (False, True):
                for hq, hk in gqa_configs:
                    try:
                        r = gate_candidate(
                            cls, DTYPES[ds], causal, hq, hk, 64, args.mode
                        )
                        print(
                            f"  {name} {ds} causal={causal} {hq}/{hk}: ratio {r:.2f} OK"
                        )
                    except AssertionError as e:
                        print(f"  {name} {ds} causal={causal} {hq}/{hk}: {e}")
                        ok = False
        gated[name] = ok
        if not ok:
            print(f"  !! {name} EXCLUDED from timing (failed gate)")

    # -- sweep ---------------------------------------------------------------
    for ds in dtypes:
        dtype = DTYPES[ds]
        for hd in head_dims:
            for hq, hk in gqa_configs:
                for causal in (False, True):
                    for b, s in SHAPES:
                        if s not in seqlens:
                            continue
                        for name in names:
                            if not gated[name]:
                                continue
                            cls = CANDIDATES[name]
                            q, k, v = make_qkv(b, s, hq, hk, hd, dtype)
                            sync()
                            note = ""
                            try:
                                with TorchMemHighWater() as mem:
                                    if HAS_MLX:
                                        mx.reset_peak_memory()
                                    cand = cls(q, k, v, causal, args.mode)
                                    fn = (
                                        cand.fwd if args.mode == "fwd" else cand.fwd_bwd
                                    )
                                    times = time_op(
                                        fn,
                                        warmup=args.warmup,
                                        iters=args.iters,
                                        budget_s=args.budget,
                                    )
                                peak = mem.peak_extra
                                if name.startswith("mlx"):
                                    peak += mx.get_peak_memory()
                            except RuntimeError as e:
                                print(
                                    f"{name} b{b} s{s} hd{hd} {hq}/{hk} c={causal}: FAILED {e}"
                                )
                                writer.write(
                                    Row(
                                        name,
                                        args.mode,
                                        b,
                                        s,
                                        hq,
                                        hk,
                                        hd,
                                        ds,
                                        causal,
                                        -1,
                                        -1,
                                        0,
                                        -1,
                                        -1,
                                        -1,
                                        note=f"RuntimeError: {e}"[:120],
                                    )
                                )
                                continue
                            med, iqr = median_iqr(times)
                            fl = attention_flops(b, s, s, hq, hd, causal, args.mode)
                            row = Row(
                                candidate=name,
                                mode=args.mode,
                                batch=b,
                                seqlen=s,
                                nheads_q=hq,
                                nheads_kv=hk,
                                head_dim=hd,
                                dtype=ds,
                                causal=causal,
                                median_ms=med * 1e3,
                                iqr_ms=iqr * 1e3,
                                iters=len(times),
                                tflops=fl / med / 1e12,
                                tokens_per_s=b * s / med,
                                peak_mem_mb=peak / 2**20,
                                note=note,
                            )
                            writer.write(row)
                            print(
                                f"{name:11s} {ds:8s} hd{hd:3d} {hq}/{hk} c={int(causal)} "
                                f"b{b:3d} s{s:6d}: {med * 1e3:9.2f} ms  "
                                f"{row.tflops:6.2f} TFLOP/s  {row.peak_mem_mb:8.0f} MiB",
                                flush=True,
                            )
                            del cand, q, k, v
                            torch.mps.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd")
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", default="core,sdpa,mlx_bridge,mlx_native")
    p.add_argument("--seqlens", default="512,1024,2048,4096,8192,16384")
    p.add_argument("--dtypes", default="float16")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--budget", type=float, default=30.0)
    args = p.parse_args()
    t0 = time.time()
    run(args)
    print(f"done in {time.time() - t0:.0f}s")
