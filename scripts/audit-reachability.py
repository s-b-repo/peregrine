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
    # Every definition site per name, not just the first. A symbol declared twice
    # — the `#[cfg(target_os = "linux")]` implementation and its
    # `#[cfg(not(...))]` stub — used to have its second declaration counted as a
    # caller of the first, so the pair vouched for each other and the symbol
    # looked reachable. That hid `register_read_buffers`/`read_fixed`, i.e. this
    # script's own headline example (`COLI_REGBUF`), from this script.
    defs: dict[str, list[tuple[Path, int]]] = {}
    prod: list[tuple[Path, int, str]] = []
    for f in files:
        lines = f.read_text(errors="replace").splitlines()
        mask = test_mask(lines)
        in_block = False
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            # Track block comments properly instead of treating every line that
            # starts with `*` as a comment continuation. That shortcut also ate
            # dereference assignments, and `*o = f16_to_f32(...)`
            # (safetensors.rs) was this scan's only reference to a live
            # production function — so a reachable symbol was reported dead.
            was_in_block = in_block
            if "/*" in stripped and "*/" not in stripped:
                in_block = True
            elif "*/" in stripped:
                in_block = False
            if was_in_block or stripped.startswith(("//", "///", "#!")):
                continue
            if not mask[i]:
                prod.append((f, i + 1, ln))
                m = DEF_RE.match(ln)
                if m:
                    defs.setdefault(m.group(1), []).append((f, i + 1))

    suspect = []
    for name, sites in sorted(defs.items()):
        word = re.compile(r"\b" + re.escape(name) + r"\b")
        declared_at = set(sites)
        if not any(
            word.search(ln)
            for f, lineno, ln in prod
            if (f, lineno) not in declared_at and not ln.strip().startswith(("use ", "pub use"))
        ):
            dfile, dline = sites[0]
            suspect.append((name, dfile.relative_to(root), dline))

    print(f"[R] pub fns with no production caller (INFO): {len(suspect)}")
    # Known blind spot, stated so absence of a symbol here is not read as proof.
    # References are matched by BARE NAME across the workspace, with no type
    # resolution, so a method whose name collides with any called function
    # anywhere scores as reached. `Indexer::{load,select,reset}` (dsa.rs) are
    # invisible for exactly this reason — `load`, `select` and `reset` are called
    # constantly on other types — which is why the whole DSA module went
    # unreported while being entirely unreachable. Treat common method names
    # (`new`, `load`, `get`, `push`, `len`, `clear`, `reset`, `select`) as
    # unchecked, not as clean.
    print("      note: bare-name matching — common method names are unchecked, not clean")
    if "--list" in sys.argv:
        for name, f, ln in suspect:
            print(f"      {f}:{ln}: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
