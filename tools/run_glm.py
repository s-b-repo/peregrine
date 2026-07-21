#!/usr/bin/env python3
"""Drive the peregrine engine's stdio serve mode: tokenize a prompt, generate,
and decode the reply. This is the coherence-test / chat harness for GLM-5.2 (and
any model in the int4/int8 container format peregrine loads).

  # real model (needs <dir>/tokenizer.json):
  run_glm.py --model /run/media/veracrypt1/glm52_i4 --prompt "Explain recursion." --ngen 64

  # plumbing test against the tiny synthetic model (no tokenizer):
  peregrine build /tmp/demo && run_glm.py --model /tmp/demo --raw-ids "1 5 9 2" --ngen 8

The engine speaks: emit READY, then per request line `GEN <ngen> <id...>` ->
space-separated generated ids + END. Greedy/deterministic on the engine side.
"""
import argparse, os, subprocess, sys

READY = "\x01\x01READY\x01\x01"
END = "\x01\x01END\x01\x01"


def find_engine():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in ["target/release/peregrine", "target/debug/peregrine"]:
        p = os.path.join(here, "..", c)
        if os.path.exists(p):
            return os.path.abspath(p)
    return "peregrine"  # rely on PATH


def read_until_ready(proc):
    for line in proc.stdout:
        if line.strip() == READY:
            return
    raise RuntimeError("engine exited before READY")


def gen(proc, ids, ngen):
    proc.stdin.write(f"GEN {ngen} {' '.join(map(str, ids))}\n")
    proc.stdin.flush()
    out = []
    for line in proc.stdout:
        s = line.rstrip("\n")
        if s == END:
            break
        if s:
            out += [int(t) for t in s.split()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--raw-ids", default=None, help="space-separated token ids (skip tokenizer)")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    a = ap.parse_args()

    tok = None
    if a.raw_ids is not None:
        ids = [int(t) for t in a.raw_ids.split()]
    else:
        assert a.prompt is not None, "need --prompt or --raw-ids"
        from tokenizers import Tokenizer
        tpath = os.path.join(a.model, "tokenizer.json")
        tok = Tokenizer.from_file(tpath)
        text = a.prompt
        if a.chat:
            # minimal GLM chat framing; the exact template lives in
            # tokenizer_config.json (chat_template) — refine once we can eyeball output.
            text = f"[gMASK]<sop><|user|>\n{a.prompt}<|assistant|>\n"
        ids = tok.encode(text).ids
        print(f"[prompt] {len(ids)} tokens", file=sys.stderr)

    engine = find_engine()
    env = dict(os.environ, COLI_MODEL=a.model)
    proc = subprocess.Popen(
        [engine, a.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    try:
        read_until_ready(proc)
        out_ids = gen(proc, ids, a.ngen)
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
    finally:
        proc.terminate()

    print(f"[generated] {len(out_ids)} ids: {out_ids}")
    if tok is not None:
        print("[text]", tok.decode(out_ids))


if __name__ == "__main__":
    main()
