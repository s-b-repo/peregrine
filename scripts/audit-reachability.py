#!/usr/bin/env python3
"""audit-reachability.py — find `pub fn`s with no production caller.

The [R] gate of docs/BAD_PATTERNS.md. It catches a defect class the grep-based
audit structurally cannot: code that compiles, passes its own unit tests, is
documented as live, and is reached by no production path. Five instances shipped
in peregrine and every one was found by hand:

  * `solve_residency_greedy` — roadmap-complete, documented in docs/gpu-cuda.md
    as the initial-placement policy, zero production callers.
  * `Placement::GpuSpill` — produced by `LaneBalancer::choose` and unit-tested;
    the only consumer matched `Placement::Gpu` and let it fall through to CPU.
  * `COLI_REGBUF` / `register_read_buffers` — an advertised env knob read by no
    code, gating an `IORING_OP_READ_FIXED` path nothing calls.
  * `open_l3_miss_counter` / `COLI_PERF_COUNTERS` — documented in five files as
    feeding the prefetch tuner; the consumer does not exist.
  * `checkout_tagged` / `checkin_tagged` — the slab pool's generation-tagged
    safety API. The untagged `checkout`/`checkin` are used; the variants that
    prevent a straggler write into a recycled slab are not.

Method: collect every `pub fn` under crates/*/src, then count references outside
(a) its own definition, (b) `use` / `pub use` re-exports, and (c) `#[cfg(test)]`
regions. Zero remaining references means nothing but tests can reach it.

Informational, not a hard gate: a workspace legitimately carries public API for
its binaries and for future consumers, so a nonzero count is expected. The value
is the *diff* — a symbol appearing here that a doc or todo.md calls shipped is
the bug. Run it when marking something complete.

Usage: scripts/audit-reachability.py [--list]
"""
import re
import sys
from pathlib import Path

DEF_RE = re.compile(r"^\s*pub(?:\(crate\))?\s+(?:unsafe\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[(<]")


def test_mask(lines):
    """True for lines inside a `#[cfg(test)]` item, tracked by brace depth."""
    mask = [False] * len(lines)
    depth = 0
    in_test = False
    test_depth = 0
    pending = False
    for i, ln in enumerate(lines):
        if "#[cfg(test)]" in ln or "cfg(all(test" in ln:
            pending = True
        opens, closes = ln.count("{"), ln.count("}")
        if pending and opens:
            in_test, test_depth, pending = True, depth, False
        mask[i] = in_test
        depth += opens - closes
        if in_test and depth <= test_depth:
            in_test = False
    return mask


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    files = sorted(
        p for p in root.glob("crates/*/src/**/*.rs") if "peregrine-token" not in str(p)
    )
    defs: dict[str, tuple[Path, int]] = {}
    prod: list[tuple[Path, int, str]] = []
    for f in files:
        lines = f.read_text(errors="replace").splitlines()
        mask = test_mask(lines)
        for i, ln in enumerate(lines):
            if ln.strip().startswith(("//", "///", "*", "#!")):
                continue
            if not mask[i]:
                prod.append((f, i + 1, ln))
                m = DEF_RE.match(ln)
                if m:
                    defs.setdefault(m.group(1), (f, i + 1))

    suspect = []
    for name, (dfile, dline) in sorted(defs.items()):
        word = re.compile(r"\b" + re.escape(name) + r"\b")
        if not any(
            word.search(ln)
            for f, lineno, ln in prod
            if not (f == dfile and lineno == dline)
            and not ln.strip().startswith(("use ", "pub use"))
        ):
            suspect.append((name, dfile.relative_to(root), dline))

    print(f"[R] pub fns with no production caller (INFO): {len(suspect)}")
    if "--list" in sys.argv:
        for name, f, ln in suspect:
            print(f"      {f}:{ln}: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
