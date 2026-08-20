#!/usr/bin/env python3
"""Open-loop arrival benchmark: fixed request *rate*, not fixed concurrency.

Both existing serving clients are closed-loop. `bench-serve-lanes.py` starts N
streams and waits for them; `bench-serve-gaps.py` does the same and reports
inter-token gaps. In a closed loop the offered load is a *consequence* of the
server's speed — if the server slows down, the client submits more slowly, and
the queue can never grow. That is the one regime in which queue time is
structurally invisible, which is why `/metrics` had no queue counter to want.

This client offers load independently of completion: requests are submitted on a
Poisson process at `--rate` per second and the client never waits for one before
sending the next. That is what makes the server's own queue the thing under
test, and what makes `--rate` sweeps show a knee rather than a smooth curve.

Reports, per run:

  * offered vs achieved request rate (they diverge once the server saturates)
  * TTFT distribution (p50/p90/p95/p99/max)
  * end-to-end latency distribution
  * server-side queue wait, read from /metrics as a delta across the run —
    the span no client-side timer can see, because it ends before the first
    token is generated
  * tokens/sec, and expert reads per token when the server is streaming

Reading it: below the knee, achieved ≈ offered and queue wait is ~0. At the
knee, queue wait climbs while achieved plateaus. Past it, queue wait grows
without bound and 503s appear (COLI_QUEUE_DEPTH shedding), which is the healthy
failure — an unbounded queue would show as latency instead.

Stdlib only, one JSON object on stdout, everything human-facing on stderr —
the same contract as the other bench clients.
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def pct(xs, p):
    """Nearest-rank percentile; `None` for an empty sample rather than a crash."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def summary(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "p50": pct(xs, 50),
        "p90": pct(xs, 90),
        "p95": pct(xs, 95),
        "p99": pct(xs, 99),
        "max": max(xs),
        "mean": statistics.fmean(xs),
    }


def metrics(base, key):
    req = urllib.request.Request(f"{base}/metrics")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def one_request(base, key, model, prompt, max_tokens, out, lock):
    """One streaming completion. Records TTFT and total latency, or the refusal.

    A 503 is data, not an error: it is `COLI_QUEUE_DEPTH` shedding, and past the
    knee it is the *expected* outcome. Counting it as a failed run would hide
    exactly the behaviour this client exists to observe.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.monotonic()
    ttft = None
    toks = 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = ev.get("choices", [{}])[0].get("delta", {}).get("content")
                if not delta:
                    continue
                if ttft is None:
                    ttft = time.monotonic() - t0
                toks += 1
        rec = {"ok": True, "ttft_s": ttft, "latency_s": time.monotonic() - t0, "deltas": toks}
    except urllib.error.HTTPError as e:
        rec = {"ok": False, "status": e.code, "latency_s": time.monotonic() - t0}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        rec = {"ok": False, "status": None, "error": str(e), "latency_s": time.monotonic() - t0}
    with lock:
        out.append(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--rate", type=float, required=True, help="offered requests/second (Poisson arrivals)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to keep offering load")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Explain what a mixture-of-experts layer does, briefly.")
    ap.add_argument(
        "--repeat-prompt",
        action="store_true",
        help="send the SAME prompt every time. Almost always wrong: peregrine-serve has a "
        "response memo for exact greedy requests, so identical prompts are answered from cache "
        "and the engine never sees the load. Only useful for measuring the memo itself.",
    )
    ap.add_argument("--seed", type=int, default=0, help="arrival-process seed, so a sweep is reproducible")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    rng = random.Random(args.seed)
    recs, lock, threads = [], threading.Lock(), []

    try:
        before = metrics(base, args.api_key)
    except (urllib.error.URLError, OSError) as e:
        print(f"cannot reach {base}/metrics: {e}", file=sys.stderr)
        return 2

    print(f"offering {args.rate}/s for {args.duration}s (Poisson, seed {args.seed})", file=sys.stderr)
    start = time.monotonic()
    submitted = 0
    while time.monotonic() - start < args.duration:
        # Distinct prompts by default. The server answers an exact repeat of a
        # greedy request from its response memo without touching the engine at
        # all — the first version of this script offered 61 requests and the
        # engine admitted *one*, which is `measurement.md`'s "benchmark that
        # measures its own cache" in a new costume. The suffix is inside the
        # user turn so the chat template and prefix cache still behave
        # realistically; only the exact-match memo is defeated.
        text = args.prompt if args.repeat_prompt else f"{args.prompt} (request {submitted})"
        t = threading.Thread(
            target=one_request,
            args=(base, args.api_key, args.model, text, args.max_tokens, recs, lock),
            daemon=True,
        )
        t.start()
        threads.append(t)
        submitted += 1
        # Exponential inter-arrival: the gap does not depend on whether anything
        # has completed, which is the whole point of an open loop.
        time.sleep(rng.expovariate(args.rate))
    offered_s = time.monotonic() - start
    print(f"submitted {submitted}; draining {len(threads)} in flight", file=sys.stderr)
    for t in threads:
        t.join()
    wall = time.monotonic() - start
    after = metrics(base, args.api_key)

    ok = [r for r in recs if r.get("ok")]
    refused = [r for r in recs if not r.get("ok") and r.get("status") == 503]
    failed = [r for r in recs if not r.get("ok") and r.get("status") != 503]

    def delta(path, default=0):
        a, b = after, before
        for k in path:
            a = (a or {}).get(k) if isinstance(a, dict) else None
            b = (b or {}).get(k) if isinstance(b, dict) else None
        if a is None or b is None:
            return default
        return a - b

    q_wait_us = delta(["queue", "wait_us"])
    q_admits = delta(["queue", "admits"])
    d_tokens = delta(["decode", "tokens_emitted"])
    d_reads = delta(["ecache", "hits"]) + delta(["ecache", "misses"])
    d_disk = delta(["ecache", "disk_reads"])
    d_energy = delta(["energy_uj"], default=None) if after.get("energy_uj") is not None else None

    out = {
        "offered_rate": args.rate,
        "submitted": submitted,
        "offer_window_s": round(offered_s, 3),
        "wall_s": round(wall, 3),
        "completed": len(ok),
        "refused_503": len(refused),
        "failed": len(failed),
        # Achieved falls below offered exactly when the server saturates; that
        # divergence is the knee this client exists to find.
        "achieved_rate": round(len(ok) / wall, 4) if wall > 0 else None,
        "ttft_s": summary([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
        "latency_s": summary([r["latency_s"] for r in ok]),
        # Server-side, from /metrics: the span that ends before the first token,
        # so no client-side timer can see it.
        "queue_wait_us_mean": round(q_wait_us / q_admits, 1) if q_admits else None,
        "queue_wait_us_max": after.get("queue", {}).get("max_us"),
        "queue_admits": q_admits,
        "tokens_emitted": d_tokens,
        "tokens_per_s": round(d_tokens / wall, 3) if wall > 0 else None,
        # Only meaningful when experts stream; a resident model reads none.
        "expert_reads_per_token": round(d_reads / d_tokens, 2) if d_tokens and d_reads else None,
        "disk_reads_per_token": round(d_disk / d_tokens, 2) if d_tokens and d_disk else None,
        "uj_per_token": round(d_energy / d_tokens, 1) if d_energy and d_tokens else None,
    }
    print(json.dumps(out, indent=2))
    if out["refused_503"]:
        print(
            f"note: {out['refused_503']} requests were shed with 503 — COLI_QUEUE_DEPTH is doing its job; "
            "past the knee this is the healthy failure, an unbounded queue would show as latency instead",
            file=sys.stderr,
        )
    if out["queue_wait_us_mean"] is None:
        print("note: no queue counters — the server predates them, or nothing was admitted", file=sys.stderr)
    if out["completed"] and q_admits < out["completed"] // 2:
        print(
            f"WARNING: {out['completed']} requests completed but the engine admitted only {q_admits}. "
            "Most were answered from the response memo, so this measured the cache and not the engine "
            "(are you passing --repeat-prompt?).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
