#!/usr/bin/env python3
"""Automated benchmark: tests both peregrine-serve (:8132) and llama-server (:8080).
Saves results as JSON for speed extraction by auto-improve-loop.sh."""
import json, subprocess, time, os, urllib.request, urllib.error
from datetime import datetime

TESTS = [
    {"name": "arithmetic", "prompt": "What is 2+2?", "expected": "4"},
    {"name": "poetry", "prompt": "Write a 3-line poem about code", "expected_words": ["code","logic","silent"]},
    {"name": "reasoning", "prompt": "If x=5, y=3, what is x+y?", "expected": "8"},
]

RESULTS_DIR = "/home/cortix/peregrine/model-bench/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def call_model(port, prompt, max_tokens=8, api_key=None):
    """Call a model via the local API and return (response, latency_ms)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    data = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        body = json.loads(resp.read())
        latency_ms = int((time.time() - start) * 1000)
        text = body["choices"][0]["message"]["content"].strip()
        return {"text": text, "latency_ms": latency_ms, "error": None}
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {"text": "", "latency_ms": latency_ms, "error": str(e)}

def get_metrics(port, api_key=None):
    """Fetch server metrics."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers=headers)
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read())
    except:
        return {}

def run_bench():
    """Run standardized tests on both servers."""
    api_key = os.environ.get("PEREGRINE_API_KEY", "")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "servers": {
            "peregrine_serve_8132": {},
            "llama_server_8080": {}
        }
    }

    for server_name, port in [("peregrine_serve_8132", 8132), ("llama_server_8080", 8080)]:
        metrics_before = get_metrics(port, api_key)
        results = []
        for test in TESTS:
            r = call_model(port, test["prompt"], max_tokens=8, api_key=api_key)
            r["test_name"] = test["name"]
            r["prompt"] = test["prompt"]
            results.append(r)
        metrics_after = get_metrics(port, api_key)

        result["servers"][server_name] = {
            "results": results,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after
        }

    # Save result
    path = f"{RESULTS_DIR}/{run_id}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {path}")
    return result

if __name__ == "__main__":
    run_bench()
