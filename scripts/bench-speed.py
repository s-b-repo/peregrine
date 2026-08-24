#!/usr/bin/env python3
"""Extract token/sec from the latest peregrine-serve benchmark result."""
import json, sys, glob, os

results_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/cortix/peregrine/model-bench/results"
latest = sorted(glob.glob(f"{results_dir}/*.json"), key=os.path.getmtime)[-1]

with open(latest) as f:
    d = json.load(f)

for srv, data in d.get("servers", {}).items():
    if "8132" not in srv:
        continue
    for r in data.get("results", []):
        if r.get("error"):
            continue
        tok = len(r["text"].split())
        sec = r["latency_ms"] / 1000.0
        if sec > 0 and tok > 0:
            print(f"{tok/sec:.2f}")
            exit(0)
print("0.0")
