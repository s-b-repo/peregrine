#!/usr/bin/env python3
"""Drive N concurrent SSE streams and report inter-token gap percentiles.

`bench-serve-lanes.py` measures whole-request wall time through the
non-streaming path, which is the right shape for throughput — and exactly the
wrong shape for stall analysis: a 0.5 s hiccup inside a 400 s request is
invisible there. This client exists for the hiccups. It records the monotonic
arrival time of every content-bearing SSE event per stream and reports the gap
distribution (p50/p90/p95/p99/max), pooled and per-stream-worst.

Built for the kvstore writer A/B (COLI_KV_STORE_SYNC=1 vs unset): a synchronous
checkpoint serializes + fsyncs on the engine thread at sequence retirement, so
every OTHER live stream's next token waits behind it — the signature is a fat
p95/p99 tail on the streams that did NOT retire. Also reusable for the
mixed-prefill-quantum question (ideas doc: one long + 15 short clients, read
the short streams' tail).

Caveat, deliberate: a delta held back by the incremental decoder (unfinished
multi-byte character, possible tool-call marker) merges two engine ticks into
one reported gap. Both arms of an A/B share that behaviour bit for bit, so
comparisons stand; absolute gap values read slightly high.

Stdlib only, one JSON object on stdout, everything human-facing on stderr.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def percentile(xs, p):
    """Nearest-rank percentile; xs must be sorted and non-empty."""
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * len(xs) + 0.5)) - 1))
    return xs[k]


def stream_one(url, api_key, model_id, prompt, max_tokens, out, idx, timeout_s):
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    arrivals = []
    err = None
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    doc = json.loads(payload)
                except json.JSONDecodeError:
                    continue  # keep-alive or partial frame; not a content event
                choices = doc.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                if delta.get("content") or delta.get("tool_calls"):
                    arrivals.append(time.monotonic())
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        err = str(e)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    out[idx] = {
        "ttft_s": (arrivals[0] - t0) if arrivals else None,
        "events": len(arrivals),
        "gaps_s": gaps,
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8131/v1/chat/completions")
    ap.add_argument("--model", default="peregrine")
    ap.add_argument("--api-key", default="")
    ap.add_argument("-n", "--streams", type=int, default=16)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--tag", default="", help="prepended to every prompt so distinct rounds produce distinct saves")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=10800.0)
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8", errors="replace") as f:
        base = f.read()
    # A per-stream prefix too: identical prompts would share one prefix-cache
    # entry and retire into one dedup'd checkpoint — the A/B wants N saves.
    prompts = [f"[{args.tag}:{i}]\n{base}" for i in range(args.streams)]

    out = [None] * args.streams
    threads = [
        threading.Thread(
            target=stream_one,
            args=(args.url, args.api_key, args.model, prompts[i], args.max_tokens, out, i, args.timeout),
        )
        for i in range(args.streams)
    ]
    wall0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - wall0

    pooled = sorted(g for s in out if s for g in s["gaps_s"])
    per_stream_p95 = sorted(
        percentile(sorted(s["gaps_s"]), 95) for s in out if s and s["gaps_s"]
    )
    report = {
        "streams": args.streams,
        "ok": sum(1 for s in out if s and not s["error"]),
        "errors": [s["error"] for s in out if s and s["error"]],
        "wall_s": round(wall, 3),
        "events_total": sum(s["events"] for s in out if s),
        "gap_pooled_s": {
            "n": len(pooled),
            "p50": percentile(pooled, 50),
            "p90": percentile(pooled, 90),
            "p95": percentile(pooled, 95),
            "p99": percentile(pooled, 99),
            "max": pooled[-1] if pooled else None,
            "mean": statistics.fmean(pooled) if pooled else None,
        },
        "gap_worst_stream_p95_s": per_stream_p95[-1] if per_stream_p95 else None,
        "ttft_median_s": statistics.median(
            sorted(s["ttft_s"] for s in out if s and s["ttft_s"] is not None)
        )
        if any(s and s["ttft_s"] is not None for s in out)
        else None,
    }
    for k in ("p50", "p90", "p95", "p99", "max", "mean"):
        v = report["gap_pooled_s"][k]
        if v is not None:
            report["gap_pooled_s"][k] = round(v, 4)
    print(
        f"streams={report['ok']}/{args.streams} events={report['events_total']} "
        f"gap p50={report['gap_pooled_s']['p50']} p95={report['gap_pooled_s']['p95']} "
        f"p99={report['gap_pooled_s']['p99']} max={report['gap_pooled_s']['max']}",
        file=sys.stderr,
    )
    json.dump(report, sys.stdout)
    print()


if __name__ == "__main__":
    main()
