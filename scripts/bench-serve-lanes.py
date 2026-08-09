#!/usr/bin/env python3
"""Drive N concurrent streaming completions against a running peregrine-serve.

This exists because `peregrine bench` (`run_bench`) cannot measure per-sequence
prefetch at all: it calls `Model::forward_step_batched` directly, and
`Model::enqueue_seq_prefetch` — the only caller that uses a prefetch lane other
than 0 — lives in `peregrine-serve`'s batch loop. A lane-count sweep built on
`bench-arms.sh` would report a clean null result by construction.

Stdlib only, so it runs anywhere the server does. Emits one JSON object on
stdout; everything human-facing goes to stderr.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def stream_one(url, api_key, model_id, prompt, max_tokens, out, idx, timeout_s):
    """One non-streaming request. Records decoded token count and wall time.

    Deliberately NOT the SSE path, even though SSE is what a real client uses.
    Counting `choices[].delta.content` events undercounts decode work: a token
    that only extends an unfinished multi-byte character emits no delta, and text
    held back as a possible partial `<tool_call>` marker emits none either. A
    first attempt at this measured 0 tokens from two healthy streams for exactly
    that reason. `usage.completion_tokens` is the engine's own count, which is
    what a throughput number should be built on.
    """
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.monotonic()
    n = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            doc = json.loads(r.read().decode("utf-8", "replace"))
        n = int(doc.get("usage", {}).get("completion_tokens", 0))
        if n == 0:
            err = "no completion_tokens in response: " + json.dumps(doc)[:300]
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, ValueError) as e:
        err = repr(e)
    out[idx] = {
        "tokens": n,
        "seconds": time.monotonic() - t0,
        "ttft": None,
        "error": err,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model-id", default="glm-5.2")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--distinct-prompts",
        action="store_true",
        help="give each stream a different prompt so they route differently — "
        "identical prompts share a routing path and would flatter prefetch",
    )
    p.add_argument("--label", default="")
    p.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="per-request timeout, seconds. A non-streaming request returns only "
        "after prefill AND all decode tokens, so this must cover the whole "
        "request: an unfused B=16 arm measured >60 min end-to-end, and the old "
        "hardwired 3600 guillotined every stream at 0 tokens (2026-08-09).",
    )
    a = p.parse_args()

    base = (
        "Explain in detail, step by step, how a mixture-of-experts language model "
        "routes each token to a subset of its experts, and why that makes memory "
        "bandwidth rather than arithmetic the limiting factor"
    )
    prompts = [
        f"{base}. Consider specifically case number {i} and answer at length."
        if a.distinct_prompts
        else base
        for i in range(a.concurrency)
    ]

    out = [None] * a.concurrency
    threads = [
        threading.Thread(
            target=stream_one,
            args=(a.url, a.api_key, a.model_id, prompts[i], a.max_tokens, out, i, a.timeout),
        )
        for i in range(a.concurrency)
    ]
    wall0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - wall0

    res = [r for r in out if r]
    errs = [r["error"] for r in res if r["error"]]
    total = sum(r["tokens"] for r in res)
    ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
    report = {
        "label": a.label,
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
        "streams_ok": sum(1 for r in res if not r["error"]),
        "errors": errs[:4],
        "tokens_total": total,
        "wall_s": round(wall, 3),
        # Aggregate throughput across all streams — the number a lane-count sweep
        # is actually about. Per-stream rate is reported separately because a
        # change that helps aggregate can still slow any single stream.
        "tokens_per_s": round(total / wall, 3) if wall > 0 else 0.0,
        "ttft_median_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "per_stream_tokens": [r["tokens"] for r in res],
    }
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")
    print(
        f"[{a.label}] {report['tokens_total']} tok in {report['wall_s']}s "
        f"= {report['tokens_per_s']} tok/s "
        f"(streams ok {report['streams_ok']}/{a.concurrency}, "
        f"ttft med {report['ttft_median_s']}s)",
        file=sys.stderr,
    )
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
