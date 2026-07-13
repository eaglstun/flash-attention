# FlashAttention on Apple Silicon (MPS) — Status

**Phase 1 complete (correctness) · Phase 3a complete (batched decode +
batched varlen) · Phase 3b complete (split lse forward + SDPA-powered
backward) · Phase 3c complete (skip the lse when nothing can read it —
public FA2 inference now within ~1.2× of raw SDPA).** This is the honesty
document: what works, what is degraded, what raises. An honest ❌ beats an
optimistic ✅ that lies.

## What this is

A **second backend behind the frozen FlashAttention API**, written in plain
differentiable PyTorch on MPS. It is **not** a port of the CUDA kernels — CuTe
DSL has no Metal target, and `nvidia-cutlass-dsl` ships Linux-only wheels.
There is exactly **one implementation of the attention math**
(`flash_attn/mps/core.py`, fp32 accumulation everywhere, memory-bounded),
verified against a CPU (and fp64) oracle by ~950 parity tests in
`tests/mps/`. Since Phase 3b the fp16/bf16 fast paths ride Apple's fused
SDPA where the semantics are mask-shaped: the forward computes `out` with
SDPA and the obligatory `lse` with a separate streaming fp32 pass (QK^T but
never V, softmax-kernel reduction, O(seqlen·block) memory), and the
Q-chunked recompute backward differentiates SDPA per chunk with masks/bias
built by the same `_build_score_mask`/`_alibi_bias` used everywhere. The
chunked online-softmax core remains the path for softcap / ALiBi(fwd) /
learnable_sink, for fp32 (see the dtype gate below), and for gradients
arriving through lse. Both public seams are thin adapters over that core:

| Seam                                    | Entry points                                                                                                            | Adapter                                                                                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **FA2** — what third-party code imports | `from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_qkvpacked_func, …, flash_attn_with_kvcache` | `flash_attn/mps/fa2_backend.py` (implements the five-function `flash_attn_2_cuda` ABI: `fwd`, `bwd`, `varlen_fwd`, `varlen_bwd`, `fwd_kvcache`) |
| **FA4** — the modern API                | `from flash_attn.cute import flash_attn_func, flash_attn_varlen_func`                                                   | `flash_attn/mps/fa4_backend.py` (dispatch above the autograd.Function; autograd provides the backward)                                          |

**Correctness first; speed is Phase 2.** Nothing here is fused. If you need
CUDA-class throughput today, this backend is not it — what it gives you is
_correct numbers and correct gradients_ through the unmodified public API.

## The one deliberate behavioral divergence: masked-row `lse`

For rows whose keys are **all masked out** (e.g. causal with
`seqlen_q > seqlen_k`, or a narrow sliding window), the two generations of
CUDA kernels genuinely disagree, and each MPS seam matches _its own_
generation:

| API                                     | `lse` on fully-masked rows | Matches                                                           |
| --------------------------------------- | -------------------------- | ----------------------------------------------------------------- |
| FA2 (`flash_attn.flash_attn_interface`) | **`+inf`**                 | CUDA kernel, non-split path (`csrc/flash_attn/src/softmax.h:180`) |
| FA4 (`flash_attn.cute`)                 | **`-inf`**                 | CuTe kernel (`flash_attn/cute/softmax.py:225`)                    |

The core natively emits `-inf`; the FA2 adapter flips the sign. `out` is
exactly `0` on those rows in both seams, gradients are exactly `0`, and no NaN
appears anywhere, forward or backward. Pinned by
`tests/mps/test_fa2_seam.py::test_fa2_lse_masked_rows_plus_inf` and
`tests/mps/test_fa4_seam.py::test_fa4_lse_masked_rows_minus_inf`.

## Feature matrix — FA2 seam (`from flash_attn import …`)

✅ supported (parity-tested) · 🟡 degraded (correct but slow / caveats) · ❌ unsupported (raises `NotImplementedError`)

| Feature                                                                                          | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| forward, fp16 / bf16 / fp32                                                                      | ✅     | fp32 accumulation always; error budget ≤ the suite's own 2× in-dtype-torch bound. fp16/bf16 with mask-shaped flags (Phase 3b): SDPA `out` + streaming fp32 lse — 3-7× of raw SDPA instead of an order of magnitude off                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| causal (bottom-right aligned), MHA / GQA / MQA                                                   | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **backward / training**                                                                          | 🟡     | correct gradients via memory-flat **Q-chunked recompute**; since Phase 3b each fp16/bf16 chunk recomputes through SDPA on fp32 leaves (masks/ALiBi bias from the same builders as the forward) — ~2.5× the manual recompute; softcap / learnable*sink / gradients-through-lse and fp32 keep the manual differentiable core. `dq/dk/dv` are filled in place per the C-extension ABI, including through the packed variants' `dqkv[:, :, i]` views. `seqlen_k == 1` always takes the manual path: single-key softmax gives \_exactly* zero dq/dk, and the manual recompute preserves the exact cancellation                                                                          |
| sliding window / local attention (`window_size`)                                                 | ✅     | FA2 encoding (negative = infinite) translated at the seam                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| softcap                                                                                          | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ALiBi (`alibi_slopes`, `(h,)` or `(b, h)`)                                                       | ✅     | for causal ALiBi the CUDA kernel uses a row-shifted bias; output and gradients are identical, `lse` differs by a per-row constant                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| custom `softmax_scale`                                                                           | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| head dims: anything ≥ 1 that fits memory                                                         | ✅     | no multiple-of-8 kernel constraint (the interface still pads to 8 before calling; harmless)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **varlen** (`flash_attn_varlen_func`, `…_qkvpacked`, `…_kvpacked`)                               | 🟡     | **batched (Phase 3a)**: pad-to-max + ONE per-batch-masked core call by default; falls back to the Phase-1 per-sequence loop when padding waste beats launch savings (extreme length skew) or when per-sequence SDPA launches win — measured heuristic in `flash_attn/mps/varlen.py`, sweep in `benchmarks/mps/bench_varlen.py --skew`. `seqused_k` supported. Uniform packed lengths pad by zero-copy reshape (Phase 3b). Since Phase 3c the lse pass only runs when a backward can consume it or the caller asked; inference pays just the masked-SDPA `out`. Still 🟡: training/lse-observable calls pay the streaming fp32 QK^T pass (Phase 3b; was the full fp32 chunked core) |
| `flash_attn_with_kvcache` (append + attend, `cache_seqlens`, `cache_batch_idx`, `cache_leftpad`) | 🟡     | in-place cache append matches CUDA (vectorized `index_put_`); attention is **one batched call** with per-batch `cache_seqlens`/`cache_leftpad` tail/head masking (Phase 3a — was a per-batch Python loop; 7.7x faster at decode shapes; ragged lengths are faster still). No backward (same as CUDA). Phase 3b: masked-SDPA `out` + streaming fp32 lse. Phase 3c: the lse pass is skipped unless `return_softmax_lse=True` (there is no backward here, so nothing else can read it) — decode b=32/cache-4096/hd128 runs at 1.9 ms uniform / 0.9 ms ragged (was 3.7/4.1 ms in 3b, 6.4 in 3a, 52.4 in Phase 2; raw batched-SDPA bound 0.5 ms)                                        |
| `deterministic`, `num_splits`, `zero_tensors`                                                    | ✅     | accepted and **ignored** — perf/legacy knobs, not semantics. (The math here is deterministic anyway.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **dropout (`dropout_p > 0`)**                                                                    | ❌     | raises. FA2's dropout mask comes from the CUDA Philox RNG _inside_ the kernel and is reproduced from `rng_state` in the backward. That RNG cannot be bit-matched on MPS, and a torch-side mask that can't be reproduced exactly in the recompute-based backward would produce **silently wrong gradients** — the exact failure mode this port exists to prevent. Set `dropout_p=0.0`                                                                                                                                                                                                                                                                                               |
| `return_attn_probs` / `S_dmask`                                                                  | ❌     | with `dropout_p == 0` it behaves exactly like CUDA: returns `(out, softmax_lse, empty S_dmask)` — the lse is real (Phase 3c pins this). Actual attention probabilities are never produced (S_dmask is only meaningful with dropout, whose parametrizations raise; the encoding is kernel-internal)                                                                                                                                                                                                                                                                                                                                                                                 |
| paged KV cache (`block_table`)                                                                   | ❌     | raises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| rotary embedding inside `fwd_kvcache` (`rotary_cos/sin`)                                         | ❌     | raises — apply rotary to q/k _before_ calling. (The in-repo rotary oracle needs triton, which doesn't exist on macOS)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `leftpad_k` in `varlen_fwd`                                                                      | ❌     | raises (not reachable from the public API; `cache_leftpad` in kvcache **is** supported)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `torch.compile` capture of the `flash_attn::*` custom ops                                        | ❌     | on MPS the interface calls the Python implementations directly, bypassing `torch.ops` — the custom-op layer runs its backend impl below the Autograd dispatch key, which breaks the recompute-based backward. Eager semantics are identical                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## Feature matrix — FA4 seam (`from flash_attn.cute import …`)

| Feature                                                                                                                                                                                       | Status  | Notes                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| forward + backward, fp16 / bf16 / fp32                                                                                                                                                        | ✅ / 🟡 | same core, same caveats: split forward + SDPA-powered Q-chunked recompute backward for fp16/bf16 (Phase 3b), manual recompute otherwise |
| `(out, lse)` 2-tuple contract (also when `return_lse=False` → `(out, None)`)                                                                                                                  | ✅      |                                                                                                                                         |
| causal, GQA/MQA, `window_size` (None = infinite; genuinely negative windows honored; the `left + right < 0` collapse quirk reproduced via the interface's own `_resolve_causal_local_window`) | ✅      |                                                                                                                                         |
| softcap, `learnable_sink`, custom `softmax_scale`                                                                                                                                             | ✅      |                                                                                                                                         |
| gradients through `lse` (dlse)                                                                                                                                                                | ✅      | the dispatch sits above the autograd.Function; `MPSFlashAttnFunc` propagates dlse                                                       |
| varlen: packed 3D + `cu_seqlens_{q,k}` (± `seqused_{q,k}` caps)                                                                                                                               | 🟡      | batched by default with the Phase-3a driver (per-sequence loop kept for skew); `lse` is `(nheads, total_q)`                             |
| varlen: batched 4D + `seqused_{q,k}`                                                                                                                                                          | 🟡      | rows past `seqused_q` are zero-filled with `lse = -inf`                                                                                 |
| `num_splits`, `pack_gqa`, `deterministic`, `max_seqlen_*`, `min_seqlen_k`                                                                                                                     | ✅      | accepted and ignored (perf hints)                                                                                                       |
| `score_mod` / `mask_mod` / `aux_tensors` / `aux_scalars`                                                                                                                                      | ❌      | raises — these are **cute-typed JIT callables**; they cannot run on torch tensors                                                       |
| block sparsity (`block_sparse_tensors*`)                                                                                                                                                      | ❌      | raises                                                                                                                                  |
| paged KV (`page_table`)                                                                                                                                                                       | ❌      | raises                                                                                                                                  |
| MLA (`qv`, hdim-512 absorption), `gather_kv_indices` / top-k                                                                                                                                  | ❌      | raises                                                                                                                                  |
| fp8                                                                                                                                                                                           | ❌      | not applicable on MPS (no fp8 torch support); dtype never reaches the backend                                                           |

## Performance caveats, stated plainly

- **`out` rides Apple's fused SDPA wherever the flags are mask-shaped**
  (plain / causal — bottom-right via boolean mask when seqlens differ — /
  GQA / local windows / varlen-`seqused` / kv-cache leftpad, fp16/bf16).
  softcap, ALiBi-in-forward and learnable_sink change the scores or the
  denominator, not just the mask, so they stay on the chunked
  online-softmax core (memory-bounded fp32 torch ops — correct, slower).
- **The `lse` obligation is only paid when something can actually read the
  lse (Phase 3c).** The FA2 interface tells the backend whether anything
  can ever observe `softmax_lse` (`_need_lse_kwargs` in
  `flash_attn_interface.py`): it is computed when a backward can run
  (grad enabled + an input requires grad — the C-extension contract saves
  it for `bwd`) or when the caller asked (`return_attn_probs=True`, which
  surfaces `softmax_lse` even with `dropout_p == 0`;
  `flash_attn_with_kvcache(..., return_softmax_lse=True)`). Otherwise the
  pass is skipped and the lse slot holds a 0-element placeholder — never
  garbage numbers; a rogue reader fails on shape, loudly. Direct callers of
  `_flash_attn_forward` / `_flash_attn_varlen_forward` / the backend ABI
  (ring-attention-style third-party code consumes `softmax_lse` from
  those) default to `need_lse=True` and always get the real thing.
  Measured (M4 Max, fp16 causal, h8 d64): public FA2 inference under
  `torch.no_grad()` went from **6.7×/5.5× of raw SDPA to 1.15×/1.04×**
  (2k/8k); decode b=32/cache-4096/hd128 from 10.4 ms to **1.8 ms**.
  Gate pinned by `tests/mps/test_fa2_seam.py::test_fa2_need_lse_gate`.
- **When the lse IS needed, it costs one streaming fp32 pass, no longer the
  whole forward (Phase 3b).** `lse` needs QK^T but never V; the pass
  computes it Q-block by Q-block through PyTorch's fused fp32 softmax
  kernel (a hand-materialized broadcast `scores - max` is ~3x slower on
  MPS). Measured (M4 Max, fp16 causal): the lse-demanding forward lands
  **3-7x of raw SDPA** (was 6-25x in Phase 2, and raw SDPA itself cannot
  return lse). The residual gap IS the fp32 discipline: scores for lse are
  computed and reduced in fp32, full stop — a "fast" fp16 lse would be a
  silently-wrong lse.
- **The backward recomputes, now through SDPA.** Memory-flat Q-chunked
  recompute as always; each fp16/bf16 chunk is recomputed with SDPA on fp32
  leaves and differentiated (gradients verified against fp64 to the same
  error as the manual recompute). Dense fp16 training (fwd+bwd with lse) is
  **2.5-3.3x faster than Phase 3a** and ~3.4-4.7x faster than the Phase-2
  core. Direct SDPA autograd (no-lse paths) is capped at 2^30 score
  elements: past that its O(s^2) backward would INT_MAX-crash (16k did,
  every phase before this one) and the chunked recompute takes over at
  near-identical speed.
- **fp32 keeps the Phase-1 paths on the lse/backward routes.** fp32's
  parity contract is ulp-referenced to an fp64 oracle and pinned
  cross-device to 4 ulp; Apple's fused kernels are accurate but not
  bit-deterministic across CPU/MPS, so fp32 stays on the hand-rolled fp32
  ops. The perf dtypes on Apple Silicon are fp16/bf16; fp32 attention is
  correct-and-slower (unchanged from Phase 1). The long-shipped no-lse fp32
  SDPA fast paths are unchanged.
- **Varlen and kv-cache decode are batched (Phase 3a), not fused.** The
  varlen driver pads to the batch max (zero-copy reshape when the packed
  lengths are uniform — Phase 3b) and makes one per-batch-masked core call;
  a per-sequence loop is kept and auto-selected by a measured cost model
  (`flash_attn/mps/varlen.py`; note the loop's cost scales with the number
  of DISTINCT sequence lengths — MPS caches graph executables per shape).
  `flash_attn_with_kvcache` is one vectorized append plus one batched
  masked attention call. Padded/masked positions contribute exactly zero,
  forward and backward (pinned by `tests/mps/test_batched_paths.py`).
- **Dropout RNG will never bit-match CUDA.** Currently moot because dropout
  raises, but if a future phase implements it, this line stays true.

## torch-MPS landmines found by this port (documented so nobody re-trips)

- Broadcast `masked_fill` with a `(b, 1, sq, sk)` mask is ~6x slower than
  the equivalent `torch.where` (Phase 3a); a broadcast `scores - max`
  materialized by hand is ~3x slower than letting `torch.softmax` do the
  shift internally (Phase 3b).
- **Integer comparisons against 0-dim CPU tensors can be silently WRONG on
  MPS at scale** (torch 2.13): `construct_local_mask` in the FA2 suite
  produced a mask with 178k wrong positions at (2048, 2048) when
  `window_size` was a `torch.randint` tensor — the reference itself was
  wrong on MPS, on CUDA the same code is fine. The suite now coerces the
  drawn window sizes to ints (identical values; the CUDA extension's
  pybind11 signature coerces to int anyway), and the backend coerces
  defensively at every entry (`_window_ints`, which also matters because
  `x[lo:hi]` with a 0-dim tensor bound silently returns the unsliced
  tensor).
- MPS fp16/fp32 SDPA forward and backward measured numerically honest
  against fp64 (errors track the manual fp32 recompute at every seqlen
  tested) — but a phantom-key lse-recovery trick was **rejected**: with
  large-spread logits SDPA's tiny attention weights carry ~30% relative
  error, so lse recovered from them was off by 0.26. Gates work.

## Verification (how we know)

- `tests/mps/` totals **948 passed** as of Phase 3c (947 pre-existing +
  the `need_lse` gate pin `test_fa2_seam.py::test_fa2_need_lse_gate`;
  fp16/bf16 route through the split forward and SDPA recompute backward by
  default, so the whole parity grid exercises them; the `float32/lse`
  worst-ratio watch item is unchanged at 1.0000, same worst case —
  re-verified after Phase 3c, `atol=0.000e+00`).
- `tests/mps/test_attention_parity.py` — 792 core parity tests: CPU (fp32/fp64)
  oracle vs the core, all dtypes × causal × GQA × head_dim × seqlen ×
  window/softcap/ALiBi/sink, forward **and** backward, plus fully-masked-row
  NaN hunts.
- `tests/mps/test_fa2_seam.py`, `tests/mps/test_fa4_seam.py` — the seam
  contracts above, through the real public entry points.
- `tests/mps/test_batched_paths.py` (Phase 3a) — the ragged/skewed corners of
  the batched decode and varlen paths: wildly varying `cache_seqlens`
  (including 0) with in-place append, single-token decode, leftpad/batch_idx/
  window/ALiBi over ragged caches, batched-vs-looped varlen strategy parity
  (fwd + bwd), zero-length sequences, fully-masked rows through the masked-
  SDPA path (exact zeros, exact zero grads), and the skew heuristic itself.
- `tests/mps/test_split_paths.py` (Phase 3b) — split-vs-chunked agreement on
  the same device across causal/window/varlen corners fwd+bwd, the fp32
  dtype gate, the SDPA-autograd INT_MAX guard (with gradient parity), the
  `seqlen_k == 1` exact-zero guarantee, and tensor-typed window sizes
  behaving exactly like ints.
- `tests/test_flash_attn.py` — the repo's own FA2 suite, run on MPS via
  `FLASH_ATTN_TEST_DEVICE=mps` (the suite was made device-parametrizable; on
  CUDA it is unchanged). **Phase-3a re-run: 14,180 passed** across
  varlen_causal / causal / varlen_output / output / kvcache / qkvpacked /
  varlen_qkvpacked / deterministic / varlen_deterministic / splitkv / bwd
  corner cases — every path the batching touches. Phase-1 run: a 1592-test
  dropout-free subset
  (output/varlen_output/causal/varlen_causal/kvcache/splitkv/deterministic/
  qkvpacked/bwd-corner-cases across seqlens 113–2048, d 64/128, fp16+bf16
  slice) — all green. **Phase 3c: unsupported features now SKIP with a
  named reason on MPS** (an autouse fixture in `tests/test_flash_attn.py`,
  no-op on CUDA): dropout parametrizations, paged-KV (`block_table`), and
  rotary-in-kvcache — the last used to die confusingly inside the tests'
  own reference code (`apply_rotary_emb` is `None` without triton). A
  bounded deterministic random sample of the 508,774-test suite
  (seed 1337, 1500 node ids): **479 passed, 1021 skipped, 0 failed** in
  100 s. Structural exclusions, stated plainly:
  - `dropout_p > 0` parametrizations skip (the backend raises
    `NotImplementedError` by design — verified explicitly before the skips
    existed: the documented error, not wrong numbers).
  - rotary kvcache cases skip: their oracle imports
    `flash_attn.layers.rotary` → triton (Linux-only).
  - paged-KV (`block_table`) cases skip: the backend raises.
  - `test_flash_attn_race_condition` was run only at small seqlens
    ((1,239)/(239,1)/(97,97)/(128,128), d=64 — all passed): each case loops
    250 fwd+bwd iterations at batch 60 to shake out CUDA races, which takes
    hours on the unfused MPS backward at seqlen ≥ 512. The determinism
    property it checks held everywhere it ran.

## Not supported, not planned here

- Making `flash_attn/cute/` CuTe kernels compile for Metal (impossible).
- Bit-exact parity with CUDA kernels (different hardware, different reduction
  order; parity is defined by the error-budget assertions above).
- Speed. That is Phase 2 (benchmark torch-SDPA vs MLX vs
  `metal-flash-attention` before writing any Metal).
