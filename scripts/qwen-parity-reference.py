#!/usr/bin/env python3
"""Dump the HF bf16 reference for the Track C parity gate.

Teacher-forces N tokens of a text through the original checkpoint and writes
{"tokens": [...], "argmax": [...]} — the file `peregrine flip-rate
<container> --reference-json ref.json` consumes. Token ids ship IN the dump,
so tokenization is out of the comparison entirely (the tokenizer-parity gate
already proved gigatoken == HF id-for-id; this keeps the two gates
independent anyway).

Box notes (cortix, 46 GB RAM, no CUDA usable for bf16-27B):
  - Needs torch (CPU wheel, ~200 MB: pip install torch --index-url
    https://download.pytorch.org/whl/cpu), transformers >= the qwen3_5
    release, accelerate, safetensors. DO NOT install while the model trickle
    is running — pip competes for the same 1.5 MB/s downlink.
  - bf16-27B does not fit RAM: device_map="auto" + max_memory + offload_folder
    spill layers to disk. One forward over ~128 positions reads every weight
    once (~56 GB ≈ minutes of stripe I/O) and computes bf16 on CPU — budget
    2-4 h wall. That is the price of a real reference, paid once.
  - Point --offload at the stripe, not the 98%-full root.

Usage:
  python3 scripts/qwen-parity-reference.py \
      --model /srv/modelstripe/qwen/Qwen3.8-27B \
      --text bench-data/2026-08-13-route-min-share/corpus.txt \
      --tokens 128 --offload /srv/modelstripe/qwen/offload \
      --out bench-data/qwen-parity/ref-128.json
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--offload", required=True, help="disk offload dir (use the stripe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ram-gb", type=float, default=30.0, help="RAM the weights may occupy (leave headroom)")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(
            f"missing dependency: {e}\n"
            "install AFTER the model trickle finishes:\n"
            "  pip install --user torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install --user 'transformers>=4.56' accelerate safetensors"
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.text, encoding="utf-8", errors="replace") as f:
        text = f.read()
    ids = tok(text, add_special_tokens=False)["input_ids"][: args.tokens]
    if len(ids) < args.tokens:
        sys.exit(f"corpus tokenizes to only {len(ids)} tokens; need {args.tokens}")

    os.makedirs(args.offload, exist_ok=True)
    t0 = time.monotonic()
    print(f"loading {args.model} bf16 with disk offload (this is the slow part)...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={"cpu": f"{args.max_ram_gb}GiB"},
        offload_folder=args.offload,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"loaded in {time.monotonic() - t0:.0f}s; teacher-forcing {len(ids)} positions...", file=sys.stderr)

    t1 = time.monotonic()
    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits = model(input_ids=input_ids).logits[0]  # [N, vocab]
        argmax = logits.argmax(dim=-1).tolist()

    out = {
        "tokens": [int(t) for t in ids],
        # Same indexing as peregrine's teacher_forcing: argmax[i] is the
        # model's next-token prediction AT position i.
        "argmax": [int(a) for a in argmax],
        "meta": {
            "model": args.model,
            "text": args.text,
            "dtype": "bfloat16",
            "forward_s": round(time.monotonic() - t1, 1),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(
        f"wrote {args.out}: {len(ids)} positions, forward {out['meta']['forward_s']}s\n"
        f"gate: ./target/release/peregrine flip-rate <int4-container> --reference-json {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
