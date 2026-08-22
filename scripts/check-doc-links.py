#!/usr/bin/env python3
"""Report markdown links that point at files which do not exist.

Docs in this repo cross-reference each other heavily (docs/todo.md alone is cited
from ~30 files), so a file move silently rots links unless something checks.
Relative links only — external URLs and in-page anchors are skipped.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "target", ".claude", "node_modules"}
SKIP_PREFIX = ("peregrine-wt-",)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def should_skip(rel: str) -> bool:
    parts = rel.split(os.sep)
    return any(p in SKIP_DIRS for p in parts) or any(
        p.startswith(SKIP_PREFIX) for p in parts
    )


def main() -> int:
    broken = []
    checked = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(SKIP_PREFIX)
        ]
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            if should_skip(rel):
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for m in LINK.finditer(text):
                target = m.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0].split(" ", 1)[0]
                if not target:
                    continue
                checked += 1
                resolved = os.path.normpath(os.path.join(dirpath, target))
                if not os.path.exists(resolved):
                    line = text[: m.start()].count("\n") + 1
                    broken.append((rel, line, target))

    for rel, line, target in sorted(broken):
        print(f"{rel}:{line}: broken link -> {target}")
    print(f"\n{checked} relative links checked, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
