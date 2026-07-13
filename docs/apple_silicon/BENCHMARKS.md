# FlashAttention on Apple Silicon — Phase 2 Benchmarks · Phase 3a Results

> **Phase 3a (2026-07-12) — the two batching wins, landed.** Work items 1
> and 2 of the Phase 3 list below are done; the section
> "[Phase 3a — measured results](#phase-3a--the-two-batching-wins-measured)"
> at the end of this document has the before/after numbers and what they
> honestly mean. Items 3-5 (lse fast path, SDPA-powered backward) are
> Phase 3b and untouched.

**Purpose:** decide what the Phase 3 fast path should be, with numbers. Phase 1
(correctness) is done; nothing is fused. This document benchmarks the four
candidates, gates every number on correctness, and ends with a recommendation.

**TL;DR — the recommendation is at the bottom, but the two sentences are:**
Phase 3 should be a **torch-level fast path built around MPS
`F.scaled_dot_product_attention`** (SDPA-forward + cheap lse, SDPA-powered
chunked backward, batched varlen/decode) — **not MLX and not hand-written
Metal**. MLX's fused kernel is no faster than Apple's own SDPA on forward, its
backward is unfused (O(L²) memory) and its bf16 gradients fail this port's
error budget; upstream `metal-flash-attention` is single-head Swift reference
code with no masking and no Python bridge.

---

## Machine & software

|        |                                                                                 |
| ------ | ------------------------------------------------------------------------------- |
| Chip   | Apple M4 Max, 16-core CPU (12P + 4E), **40-core GPU**                           |
| Memory | 64 GB unified                                                                   |
| OS     | macOS 26.4.1 (25E253)                                                           |
| Python | 3.14.2                                                                          |
| torch  | 2.13.0 (MPS backend)                                                            |
| mlx    | 0.32.0                                                                          |
| Branch | `feature/apple-silicon-mps`, flash_attn/mps as of Phase 1 + this phase's tuning |

Scripts: `benchmarks/mps/` (`bench_attention.py`, `bench_interop.py`,
`bench_chunks.py`, `bench_varlen.py`; tables regenerated with
`make_tables.py`). Run with `PYTHONPATH=. python benchmarks/mps/bench_attention.py --mode fwd --out fwd.csv`.

## Methodology (read before trusting any number)

- **Sync discipline.** `torch.mps.synchronize()` before **and** after every
  timed region — without the trailing sync you time kernel _launch_, not
  execution. MLX regions `mx.eval(...)` their outputs, which blocks until the
  GPU work completes. (Empirical footnote: calling `mx.synchronize()` right
  after `mx.eval` on graphs consuming `mx.from_dlpack`-imported torch-MPS
  buffers **deadlocks** in mlx 0.32.0; `mx.eval` is already a completion
  barrier, so the benchmarks rely on that.)
- **Warmup** 2–3 iterations (MPS and MLX both JIT-compile on first use), then
  up to 10 timed iterations (min 3, per-config time budget); tables report the
  **median**; IQRs are in the CSVs.
- **Peak memory.** torch.mps has no `max_memory_allocated`, and the driver
  high-water does not reset reliably within a process, so a background thread
  samples `torch.mps.current_allocated_memory()` at ~0.2 ms during the run
  (MPS ops allocate buffers eagerly on the calling thread, so transient peaks
  are visible). MLX candidates add `mx.get_peak_memory()` after a reset.
  Treat values as lower bounds with single-buffer resolution; "0" means
  below sampling resolution (< ~32 MiB), which for the fused SDPA forward is
  real: it is genuinely memory-light.
- **Correctness gates every number.** Before timing, every candidate ×
  dtype × causal × GQA config must reproduce the CPU fp32 oracle within the
  repo's own error budget (candidate error ≤ 2× the same math in the test
  dtype — the assertion style of `tests/cute/test_flash_attn.py`), on outputs
  **and, for training mode, on dq/dk/dv**. Candidates that fail are excluded
  and reported, not benchmarked. **This fired twice for real** (see
  "Correctness findings").
- **Shapes.** Constant token count `batch × seqlen = 16384` so throughput is
  comparable across rows: (32×512), (16×1024), (8×2048), (4×4096), (2×8192),
  (1×16384). Heads 8/8 (MHA) and 8/2 (GQA), head_dim 64/128, fp16 full grid,
  bf16 spot checks, causal and non-causal. TFLOP/s uses the flash-attention
  convention (fwd `4·b·h·s²·d`, ×0.5 causal; fwd+bwd = 3.5× fwd).

## Candidates

| name         | what it is                                                                                                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core`       | the Phase 1 backend as shipped: `mps_flash_attn_func` with `return_lse=True` — chunked online-softmax fwd, recompute bwd, fp32 accumulation everywhere. This is the path the FA2 seam always takes (the ABI requires lse). |
| `sdpa`       | plain `F.scaled_dot_product_attention` on MPS (`enable_gqa=True`), autograd backward. No lse. The torch-level ceiling.                                                                                                     |
| `mlx_bridge` | MLX `mx.fast.scaled_dot_product_attention`, paying the torch↔MLX DLPack bridge on every call.                                                                                                                              |
| `mlx_native` | same MLX kernel, arrays already resident in MLX (kernel-only; the "whole model lives in MLX" number).                                                                                                                      |

`metal-flash-attention` was assessed by reading the repo (see below) — it was
**not** benchmarked because upstream cannot express our problem (single-head,
no causal mask) and the only Python packaging of it is an unaudited
third-party fork; per the ground rules, nothing unverified gets a number.

---

## Results — forward (inference), fp16

Head 8/8, causal:

| batch × seqlen | core ms | core TFLOP/s | sdpa ms  | sdpa TFLOP/s | mlx_bridge ms | mlx_native ms | mlx_native TFLOP/s |
| -------------- | ------- | ------------ | -------- | ------------ | ------------- | ------------- | ------------------ |
| 32 × 512       | 11.6    | 0.74         | **0.9**  | 9.40         | 2.2           | 0.9           | 9.78               |
| 16 × 1024      | 19.5    | 0.88         | **1.6**  | 11.03        | 2.8           | 1.5           | 11.31              |
| 8 × 2048       | 36.8    | 0.93         | **2.9**  | 11.99        | 4.1           | 2.8           | 12.27              |
| 4 × 4096       | 71.5    | 0.96         | **5.5**  | 12.49        | 6.7           | 5.4           | 12.76              |
| 2 × 8192       | 144.4   | 0.95         | **10.7** | 12.85        | 12.0          | 10.6          | 12.96              |
| 1 × 16384      | 282.4   | 0.97         | **22.0** | 12.47        | 23.3          | 22.6          | 12.19              |

Head 8/8, head_dim 128, causal (same pattern):

| batch × seqlen | core ms | sdpa ms | mlx_native ms |
| -------------- | ------- | ------- | ------------- |
| 8 × 2048       | 45.7    | 6.1     | 6.3           |
| 1 × 16384      | 330.5   | 54.4    | 53.5          |

Peak memory, forward: `core` 1.5–2.1 **GiB** (fp32 chunk traffic); `sdpa`
**< 32 MiB**; `mlx` 32–160 MiB. GQA 8/2 and non-causal tables (CSVs / `make_tables.py`)
show the same shape; bf16 forward is within ~5 % of fp16 at 2048 and ~1.5×
slower at 8192 (both sdpa and mlx).

**What this says:**

1. **torch SDPA and the MLX fused kernel are the same speed** (~10–13
   TFLOP/s at every shape, both dtypes). Apple's MPS SDPA forward is already
   flash-class; MLX buys nothing on forward.
2. **The Phase 1 core is 6–25× off the ceiling**, and the reason is
   structural: the FA2 ABI always demands lse, which forces the chunked fp32
   path even for plain causal attention. The SDPA fast path exists in
   `core.py` but is unreachable from the FA2 seam.
3. The MLX bridge costs ~0.7–3 ms per call regardless of sequence length
   (details below) — irrelevant at 16k, ~2× at small shapes.

## Results — forward + backward (training), fp16

Head 8/8, head_dim 64, causal:

| batch × seqlen | core ms | core TFLOP/s | core peak | sdpa ms   | sdpa TFLOP/s | sdpa peak    | mlx_native ms | mlx TFLOP/s | mlx peak     |
| -------------- | ------- | ------------ | --------- | --------- | ------------ | ------------ | ------------- | ----------- | ------------ |
| 32 × 512       | 52.6    | 0.57         | 0.8 GiB   | 23.5      | 1.28         | 0.8 GiB      | **9.2**       | 3.26        | 0.6 GiB      |
| 16 × 1024      | 101.9   | 0.59         | 1.6 GiB   | 39.7      | 1.52         | 1.5 GiB      | **18.7**      | 3.22        | 0.9 GiB      |
| 8 × 2048       | 216.7   | 0.55         | 2.2 GiB   | 69.6      | 1.73         | 3.0 GiB      | **33.7**      | 3.57        | 1.7 GiB      |
| 4 × 4096       | 443.9   | 0.54         | 2.2 GiB   | 130.7     | 1.84         | 6.0 GiB      | **71.3**      | 3.37        | 3.2 GiB      |
| 2 × 8192       | 868.9   | 0.55         | 1.7 GiB   | 245.8     | 1.96         | **12.0 GiB** | **162.7**     | 2.96        | 6.4 GiB      |
| 1 × 16384      | 1739.1  | 0.55         | 2.2 GiB   | **FAIL**¹ | —            | —            | **370.8**     | 2.59        | **12.9 GiB** |

¹ `MPSGraph does not support tensor dims larger than INT_MAX`: torch SDPA's
backward materializes the `b·h·s²` attention matrix, and at s=16384 that is
2.1 G elements. **Plain SDPA training hard-fails at 16k context on MPS**, in
every configuration tested. This is the flash-attention memory argument,
verbatim, on Apple hardware.

Head 8/8, head_dim 128, causal:

| batch × seqlen | core ms | sdpa ms | mlx_native ms |
| -------------- | ------- | ------- | ------------- |
| 8 × 2048       | 311.6   | 105.6   | **59.9**      |
| 2 × 8192       | 1191.0  | 379.8   | **260.1**     |
| 1 × 16384      | 2231.7  | FAIL¹   | **551.3**     |

GQA 8/2 (causal, hd64) exposes a `core` own-goal: 0.32–0.38 TFLOP/s vs 0.55
for 8/8 — the chunked path materializes `repeat_interleave` copies of K/V per
chunk, so GQA is currently _slower_ than MHA. sdpa/mlx do grouped attention
natively and don't pay this.

**What this says:**

1. **MLX is the fastest correct fp16 training path measured**: 4–8× over
   `core`, 1.5–2.3× over sdpa-autograd — _even though its backward is
   unfused_ (verified in MLX source: the Metal fused-attention VJP is "NYI"
   and MLX deliberately uses the unfused path for training; the fused forward
   is even skipped under grad).
2. **But it is not memory-bounded**: O(s²) activations, 12.9 GiB at 16k and
   b=1. At 32k it would need >50 GiB — dead on this machine. `core` is the
   only candidate whose training memory is flat (~2 GiB at every length).
3. **sdpa-autograd training is 2.5–3.5× faster than `core`** up to 8k, then
   dies at 16k (INT_MAX). Its gradients passed the error-budget gate in both
   fp16 and bf16.
4. bf16 training: core and sdpa behave like fp16 (~5–20 % slower). **MLX was
   excluded — see below.**

## Correctness findings (the gates earned their keep)

- **MLX bf16 backward gradients fail the port's error budget.** dq error =
  4.63e-3 vs budget 2 × 2.17e-3 (ratio ≈ 2.14; the suite allows 2.0) on a
  plain causal 8/8 hd64 shape. fp16 gradients pass but land at ratio
  1.4–2.01 — right at the edge. Outputs (forward) are excellent in both
  dtypes (ratio 0.4–0.8). Interpretation: MLX's unfused VJP accumulates in
  the compute dtype somewhere torch accumulates in fp32. **Any Phase 3 MLX
  training path would need fp32-upcast Q/K/V (halving its speed advantage)
  or a fused kernel with fp32 accumulation to meet this port's correctness
  bar.** A fast wrong answer is worth zero: the bf16 MLX training numbers
  were therefore not recorded.
- **Zero-copy interop is a live race.** MLX reading a `from_dlpack`-shared
  buffer before torch's producing op completes returns garbage (caught by
  the gate as errors up to 1.9 absolute on the first harness draft; fixed by
  a `torch.mps.synchronize()` between the torch-side write and the MLX
  read). torch's MPS stream and MLX's stream are not ordered with respect to
  each other — any real bridge must own this synchronization.
- `mx.synchronize()` after `mx.eval` on dlpack-fed graphs can **deadlock**
  (mlx 0.32.0) — see Methodology.
- MLX's `mask="causal"` is **bottom-right aligned** for `seqlen_q ≠
seqlen_k` (verified empirically) — it matches flash-attention's
  convention, unlike torch SDPA's top-left `is_causal`. One fewer
  integration hazard than feared.

## The MLX bridge, isolated (`bench_interop.py`)

| leg                                                         | 48 MiB q+k+v (8×2048×8×64) | 96 MiB (1×16384×8×128) |
| ----------------------------------------------------------- | -------------------------- | ---------------------- |
| torch `transpose+contiguous` ×3 (layout fix)                | 1.37 ms                    | 1.86 ms                |
| `mx.from_dlpack` ×3 (**zero-copy**, verified shared memory) | **4 µs**                   | **4 µs**               |
| `mx.array` ×3 (copying import, for contrast)                | 1.64 ms                    | 2.36 ms                |
| MLX causal sdpa kernel alone                                | 2.83 ms                    | 49.87 ms               |
| `torch.from_dlpack(out)` (zero-copy)                        | **2 µs**                   | **2 µs**               |
| whole bridged call                                          | 4.23 ms                    | 58.64 ms               |

The DLPack exchange itself is free (µs, both directions, `kDLMetal` device).
The real bridge cost is the torch-side layout copy to row-major (b,h,s,d)
plus one hard sync — **~1.4–3 ms per attention call**, independent of
sequence length. It does _not_ eat the kernel win at training shapes (the
kernel is tens of ms there), but it doubles small-shape forward calls. The
"interop kills MLX" hypothesis is **rejected** — MLX dies on other grounds
(no fused backward, bf16 gradient accuracy, forward parity with SDPA).

## The known-pathological paths (`bench_varlen.py`)

| workload                                                         | Phase 1 path                  | batched alternative                                 | gap      |
| ---------------------------------------------------------------- | ----------------------------- | --------------------------------------------------- | -------- |
| varlen fwd, 32 packed seqs (64–1024 tokens, 17.3k total), causal | 53.9 ms (per-seq Python loop) | 2.98 ms (padded dense SDPA, incl. wasted pad FLOPs) | **18×**  |
| varlen fwd+bwd, same                                             | 187.0 ms                      | 75.5 ms (padded SDPA autograd)                      | 2.5×     |
| kvcache decode, b=32, sq=1, cache 4096, 8/2 hd128                | 53.9 ms (per-batch loop)      | 0.44 ms (one batched SDPA)                          | **123×** |

The decode number is the worst in this document: 32 tiny per-batch calls cost
two orders of magnitude over one batched launch. Anyone serving a model
through `flash_attn_with_kvcache` on MPS hits this every token.

## Chunk-size sweep (`bench_chunks.py`) — the cheap win, measured

Forward `kv_chunk_size`: latency is **flat within ±10 %** from 256 to
full-sequence while peak memory scales linearly (0.6 GiB at 256 → 17 GiB
unchunked at 16k). The default 1024 is already on the flat part; **no
latency win is available from forward chunk tuning** — the forward's problem
is dispatch+fp32 traffic, not chunk granularity.

Backward `q_chunk_size` (causal fp16, median of 5):

| shape         | 128     | 256 (old default) | **512 (new default)** | 1024    | 2048    |
| ------------- | ------- | ----------------- | --------------------- | ------- | ------- |
| 4×4096 hd64   | 513 ms  | 468 ms            | **438 ms**            | 429 ms  | 425 ms  |
| 4×4096 hd128  | 650 ms  | 579 ms            | **525 ms**            | 513 ms  | 507 ms  |
| 1×16384 hd64  | 1820 ms | 1714 ms           | **1647 ms**           | 1617 ms | 1616 ms |
| 1×16384 hd128 | 2127 ms | 1974 ms           | **1901 ms**           | 1845 ms | 1810 ms |

512 is 4–9 % faster than 256 with flat-or-lower peak memory; beyond 512 the
gains shrink while memory grows. **Landed:** `_Q_CHUNK_SIZE_BWD = 512`.
Also landed: the SDPA fast path now uses `enable_gqa=True` instead of
materializing `repeat_interleave` K/V copies — **on MPS only**: the parity
suite caught CPU's grouped-backward dv accumulation landing 1.26× outside the
fp32 error budget, so the CPU path keeps the materialized copies (that catch
is `tests/mps/test_attention_parity.py::test_sdpa_fast_path[True-fp32-cpu]`).
Both changes re-validated against the full `tests/mps` parity suite.

## metal-flash-attention (read, not benchmarked)

Upstream (`philipturner/metal-flash-attention`): a Swift package that
JIT-generates MSL at runtime. Has forward **and** a two-kernel backward
(designed to avoid fp32 atomics), fp16/bf16(emulated)/fp32, head dims to 256,
~83 % ALU utilization on M1 Max. But: **single-head only, no causal masking
(feature request closed "not planned"), no GQA, no sliding window, no varlen,
Swift API only, no Python bridge (open issue unanswered), dormant since
Sept 2024.** The production lineage lives in Draw Things' C++ port (ccv),
not upstream. Integrating it here means writing the missing mask/multi-head/
GQA kernel logic ourselves plus a torch extension — that is "write our own
Metal backend using MFA as reference", weeks of work, not an integration.
There is a young pip-packaged fork (`mps-flash-attn`, claims fwd+bwd, causal,
GQA, sliding window, torch custom ops; ~15 stars, one maintainer, claims
unaudited) — worth an eyes-open evaluation someday, but not a foundation this
port should stand on today, and installing unaudited native code was out of
scope for this phase. Relevant upstream motion: MLX PR #3241 proposes fused
Metal attention VJP kernels; if merged, MLX's training story changes and the
MLX question should be reopened.

---

## Recommendation for Phase 3

**Build the fast path on torch MPS SDPA. Skip MLX. Write no Metal yet.**

The evidence, compressed:

- Forward ceiling: SDPA **is** the ceiling (10–13 TFLOP/s); MLX's hand-fused
  kernel merely ties it. Our core is 6–25× below, purely because of plumbing
  (lse forces the chunked path; varlen/decode loop per sequence).
- Training: MLX leads (1.5–2.3× over sdpa-autograd) but with O(s²) memory,
  edge-of-budget fp16 gradients, **over-budget bf16 gradients**, a new
  dependency, and demonstrated cross-runtime sync hazards. sdpa-autograd is
  2.5–3.5× over core out of the box and its gradients pass the budget in both
  dtypes — and a chunked-recompute backward that calls **SDPA per Q-chunk**
  (differentiable, additive-bias mask for bottom-right causal/window/ALiBi)
  keeps memory flat while capturing most of that speedup at any length,
  including 16k+ where plain SDPA autograd hard-fails on INT_MAX.

### Phase 3 work list, in payoff order

1. **Decode/kvcache batching** — replace the per-batch loop with one batched
   SDPA + `cache_seqlens` masking. Measured gap: **123×**. Effort: ~a day.
2. **Varlen batching** — pad-and-mask (or bucket) instead of the per-sequence
   loop; measured gap **18×** fwd, 2.5× train. Effort: ~a day or two.
3. **SDPA forward + separate chunked lse** when the ABI demands lse: SDPA for
   `out` (< 32 MiB, 10–13 TFLOP/s) plus a chunked QKᵀ logsumexp pass (~half
   the forward FLOPs, no O(s²) memory). Expected: inference through the
   public APIs lands within ~2× of the SDPA ceiling instead of 6–25× off.
   Effort: days, including parity tests.
4. **SDPA-powered backward**: keep the Q-chunked recompute structure, but
   compute each chunk with SDPA (additive fp32 bias expresses bottom-right
   causal, window, ALiBi; softcap/sink stay on the fp32 manual path).
   Expected: training 2.5–4× faster than today with memory still flat;
   validated by the existing 888-test parity harness. Effort: ~a week.
5. **Fix GQA in the remaining chunked paths** via `enable_gqa`/broadcast
   instead of `repeat_interleave` (the 8/2 < 8/8 inversion above).

Expected end state: inference near Apple's ceiling, training ~3–4× today's,
all pathologies gone, **zero new dependencies, zero Metal, zero new
correctness surface beyond what the parity harness already pins.**

### When to reopen the other options

- **MLX**: if PR #3241 (fused Metal attention VJP) merges _and_ its fp32
  accumulation passes our gradient gates, MLX becomes a genuine 2×+ training
  kernel behind a bridge we now know is nearly free. Re-run
  `bench_attention.py --mode fwd_bwd` against it — the harness is ready.
- **Own MSL (bitsandbytes-shape)**: only if, after item 4, the SDPA-hybrid
  training number (~2–4 TFLOP/s) is still the bottleneck for Eric's actual
  training runs. The upside over the hybrid is maybe 2–3× (MLX's unfused 3–8
  TFLOP/s is a floor for what a fused kernel could do), and the backward is
  the hard, weeks-scale part — exactly the part MFA upstream doesn't solve
  for our feature set either.

---

## Phase 3a — the two batching wins, measured

**Landed 2026-07-12** (branch `feature/apple-silicon-mps`). Scope: work items
1 and 2 only — batch the kv-cache decode path and batch varlen. No lse fast
path, no backward changes (Phase 3b). Same machine and methodology as above
(`torch.mps.synchronize()` before and after every timed region; medians).
All numbers reproducible with `benchmarks/mps/bench_varlen.py` (and
`--skew` for the strategy sweep).

### What changed

- `flash_attn/mps/core.py`: the mask/ALiBi builders and both forwards accept
  per-batch `seqused_q` / `seqused_k` / `key_leftpad` (padded-varlen /
  kv-cache formulation, bottom-right diagonal aligned to each batch
  element's _effective_ lengths); GQA in the chunked forward now uses
  grouped einsums over native KV heads instead of `repeat_interleave`
  copies (work item 5, forward half — it was load-bearing for the decode
  win); score masking uses `torch.where` instead of `masked_fill`
  (broadcast `masked_fill` is ~6x slower on MPS — measured, and it was most
  of the first batched prototype's cost); a masked-SDPA fast path for
  ragged shapes when no lse is demanded (boolean mask, fully-masked rows
  forced to exact 0 with exact-0 grads).
- `flash_attn/mps/fa2_backend.py::fwd_kvcache`: vectorized in-place cache
  append (`index_put_`) + ONE batched masked attention call. No Python loop.
- `flash_attn/mps/varlen.py`: pad -> one batched call -> repack by default,
  per-sequence loop kept and auto-selected by a measured skew heuristic.

### Headline before/after (the Phase 2 pathology table, re-run)

Same script, same shapes, same machine; "before" re-measured on the Phase-2
code immediately before the change, "after" is the final committed state.

| workload (same shapes as Phase 2)                           | before (loop) | after (Phase 3a) | speedup  | raw batched-SDPA bound  |
| ----------------------------------------------------------- | ------------- | ---------------- | -------- | ----------------------- |
| kvcache decode, b=32, sq=1, cache 4096, 8/2 hd128           | 52.4 ms       | **6.8 ms**       | **7.7x** | 0.47 ms                 |
| kvcache decode, same but ragged `cache_seqlens`             | (loop, 53 ms) | **5.9 ms**       | ~9x      | —                       |
| varlen fwd, 32 packed seqs (64-1024 tok), causal, FA2 (lse) | 52.8 ms       | **45.2 ms**      | 1.2x     | 3.2 ms                  |
| varlen fwd, same, no lse (FA4 `return_lse=False`)           | 3.4 ms¹       | **3.3 ms**       | 1.0x     | 3.2 ms                  |
| varlen fwd+bwd, same, FA2 seam through autograd             | 227.9 ms      | **140.8 ms**     | **1.6x** | 79.5 ms (padded SDPA)   |
| varlen fwd (lse), 256 seqs x 32 tokens (launch-bound)       | 66.5 ms       | **8.3 ms**       | **8.0x** | —                       |
| varlen fwd (lse), 64 seqs x 128 tokens                      | 34.1 ms       | **6.4 ms**       | **5.3x** | —                       |
| varlen fwd (lse), 512 seqs x 32 tokens                      | 132.9 ms      | **10.2 ms**      | **13x**  | —                       |
| varlen fwd (lse), 1 x 8192 + 63 x 64 (extreme skew)         | 68 ms         | **68 ms** (loop) | 1.0x     | batched would be 15.6 s |

¹ The Phase-1 loop already hit the per-sequence SDPA fast path when no lse
was demanded; that case was never slow. The Phase-2 "18x" line compared the
FA2 entry point (which must return lse) against raw SDPA (which cannot) —
so part of that "18x" was never a batching gap at all.

### Honest accounting vs the Phase 2 headroom

- **Decode: 7.7x of the 123x, and the other 16x is not a batching problem.**
  The loop is gone (one vectorized append + one batched attention call);
  what remains is that `fwd_kvcache` must return `softmax_lse`, which SDPA
  does not expose, so the attention itself still runs on the fp32 chunked
  core (fp32 K/V traffic + manual online softmax) instead of Apple's fused
  fp16 SDPA (0.47 ms). That last ~14x is exactly Phase 3b work item 3
  (SDPA forward + a separate cheap lse pass), deliberately out of scope
  here. Ragged `cache_seqlens` cost nothing extra — they are _faster_
  (5.9 ms), because the batch's max post-append length is below the full
  cache.
- **Varlen: the "18x" number was mostly an lse problem wearing a batching
  costume.** It compared the FA2 entry point (obliged to return lse, hence
  the fp32 chunked core) against raw SDPA (which cannot return lse) — so
  most of that gap was never reachable by batching. Batched: at the
  Phase-2 benchmark shape the padded batch's wasted FLOPs nearly cancel the
  loop's launch overhead in the lse forward (52.8 -> 45.2 ms, 1.2x). The
  batching win is real and large exactly where launches dominate — **5-13x
  for many-short-sequence batches** (64x128: 5.3x; 256x32: 8.0x; 512x32:
  13x), which is the packed-training shape that motivates varlen in the
  first place — and **1.6x on the FA2 fwd+bwd path** (227.9 -> 140.8 ms:
  the recompute backward is now one masked-SDPA autograd call instead of
  32). The rest of the gap to 3.2 ms is, again, lse -> Phase 3b.
- **Skew is handled, not hidden.** A padded batch does `batch x max_len²`
  work; at 1x8192+63x64 that is 15.6 _seconds_ vs the loop's 68 ms. The
  driver picks per call with measured constants (three regimes documented
  in `flash_attn/mps/varlen.py`; sweep: `bench_varlen.py --skew`), and the
  extreme-skew row above shows the heuristic choosing the loop.

### Correctness (nothing moved)

`tests/mps/`: **911 passed** (890 pre-existing — zero tolerance changes,
worst error-budget ratios unchanged including the documented
`float32/lse = 1.0000` watch item — plus 21 new ragged/skew regression
tests in `tests/mps/test_batched_paths.py`). Repo FA2 suite on MPS
(`FLASH_ATTN_TEST_DEVICE=mps`): **14,180 passed**, dropout-free subset
across `varlen_causal` (960), `causal` (960), `varlen_output` (1920),
`output` (1152), `kvcache` (1920), `qkvpacked` (384), `varlen_qkvpacked`
(1440), `deterministic` (960), `varlen_deterministic` (960), `splitkv`
(3456), and the bwd corner cases (68) — every path this change touches,
with zero failures. Padded/masked positions contribute exactly zero
(forward and backward) by construction and by test.
