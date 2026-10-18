#!/usr/bin/env python3
"""Total size of a striped peregrine container, in MB.

A striped container's weights live in the per-device expert directories named by
`model_paths.json`; the model directory itself holds only sidecars. Sizing the
stage off the model directory reports ~20 MB for a 114 GB container, which then
plans a stage 200x too small.

`du $MODEL` is wrong for exactly this reason, which is why this file exists
rather than a one-liner in the shell script.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(0)
        return 2
    model = sys.argv[1]
    total = 0
    paths = []
    index = os.path.join(model, "model_paths.json")
    if os.path.exists(index):
        try:
            with open(index) as handle:
                paths = json.load(handle).get("paths", [])
        except (OSError, ValueError):
            paths = []
    for root_dir in paths or [model]:
        for root, _dirs, files in os.walk(root_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    print(total // (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
