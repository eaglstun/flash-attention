"""Pivot bench_attention.py CSVs into the markdown tables used by
docs/apple_silicon/BENCHMARKS.md.

Usage: python benchmarks/mps/make_tables.py fwd_fp16.csv [fwd_bwd_fp16.csv ...]
"""

import csv
import sys
from collections import defaultdict

CAND_ORDER = ["core", "sdpa", "mlx_bridge", "mlx_native"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def table(rows, dtype, head_dim, causal, gqa, mode):
    sel = [
        r
        for r in rows
        if r["dtype"] == dtype
        and int(r["head_dim"]) == head_dim
        and r["causal"] == str(causal)
        and (int(r["nheads_q"]), int(r["nheads_kv"])) == gqa
        and r["mode"] == mode
    ]
    if not sel:
        return
    by_seq = defaultdict(dict)
    for r in sel:
        by_seq[(int(r["batch"]), int(r["seqlen"]))][r["candidate"]] = r
    cands = [c for c in CAND_ORDER if any(c in v for v in by_seq.values())]
    print(
        f"\n#### {mode} | {dtype} | head_dim {head_dim} | "
        f"{'causal' if causal else 'non-causal'} | heads {gqa[0]}/{gqa[1]}\n"
    )
    hdr = "| batch x seqlen |"
    sep = "|---|"
    for c in cands:
        hdr += f" {c} ms | {c} TFLOP/s | {c} peak MiB |"
        sep += "---|---|---|"
    print(hdr)
    print(sep)
    for (b, s), d in sorted(by_seq.items(), key=lambda kv: kv[0][1]):
        line = f"| {b} x {s} |"
        for c in cands:
            r = d.get(c)
            if r is None or float(r["median_ms"]) < 0:
                note = (r or {}).get("note", "")
                line += f" {'OOM' if 'memory' in note.lower() else 'FAIL'} | - | - |"
            else:
                line += (
                    f" {float(r['median_ms']):.1f} | {float(r['tflops']):.2f} |"
                    f" {float(r['peak_mem_mb']):.0f} |"
                )
        print(line)


def main():
    rows = []
    for path in sys.argv[1:]:
        rows.extend(load(path))
    dtypes = sorted({r["dtype"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        for dtype in dtypes:
            for hd in (64, 128):
                for gqa in ((8, 8), (8, 2)):
                    for causal in (True, False):
                        table(rows, dtype, hd, causal, gqa, mode)


if __name__ == "__main__":
    main()
