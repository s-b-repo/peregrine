#!/usr/bin/env bash
# audit-bad-patterns.sh — a re-runnable quality gate for the peregrine workspace,
# adapted from the rustsploit audit. It catches crash-vectors and UB that
# `cargo clippy` does not flag by default, across ALL crates (clippy denies live
# per-crate and can drift; this gate is workspace-wide).
#
# Usage:
#   scripts/audit-bad-patterns.sh            # full report (shows offending lines)
#   scripts/audit-bad-patterns.sh --strict   # exit non-zero on any P/U hit (CI gate)
#
# Scope: Rust under crates/*/src. Comment lines are ignored. `unsafe` is EXPECTED
# in peregrine-io (io_uring + aligned alloc), peregrine-cuda (FFI), and
# peregrine-kernels (SIMD intrinsics); it is only reported for the pure-logic
# crates. Numeric `as` casts and array indexing are intentional in a kernel engine
# and are NOT flagged.
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

# peregrine-token is vendored third-party code (marcelroed/gigatoken, MIT):
# it keeps upstream style (unwrap/expect in tests + engine internals, unsafe
# SIMD), so it is exempt from the engine crates' panic-free gate. The thin
# facade in its lib.rs is covered by the parity + round-trip test suite
# instead. See docs/BAD_PATTERNS.md.
mapfile -t FILES < <(find crates -path '*/src/*' -name '*.rs' -type f ! -path 'crates/peregrine-token/*' | sort)

# scan <regex>: matching non-comment lines as "file:line:code"
scan() {
  grep -rEn -- "$1" "${FILES[@]}" 2>/dev/null \
    | grep -vE ':[0-9]+:[[:space:]]*(//|///|\*)' \
    | grep -v 'audit-allow:'
}

# run_section <out-var> <pattern...>: print per-pattern counts (+ lines unless --strict),
# set <out-var> to the section total.
run_section() {
  local __var="$1"; shift
  local total=0 re hits c
  for re in "$@"; do
    hits="$(scan "$re")"
    c="$(printf '%s' "$hits" | grep -c . || true)"
    if [ "$c" -gt 0 ]; then
      printf '  %-40s %d\n' "$re" "$c"
      [ "$STRICT" -eq 0 ] && printf '%s\n' "$hits" | sed 's/^/      /'
      total=$((total + c))
    fi
  done
  printf -v "$__var" '%d' "$total"
}

# P: panicking error handling — a panic crashes the serve loop mid-inference.
P_PATTERNS=(
  '\.unwrap\(\)' '\.expect\(' '\.unwrap_or_default\(\)'
  '\.unwrap_err\(\)' '\.expect_err\('
  'panic!\(' 'unreachable!\(' 'todo!\(' 'unimplemented!\('
)
# B: silent error swallowing — a discarded Result hides real failures. Advisory
# operations must handle the Err explicitly (peregrine_io::note_advisory_err,
# COLI_DEBUG=1 surfaces them); control-flow errors must match concrete variants
# (`Err(VarError::NotPresent)`, `TryRecvError::Empty`) — never a bare wildcard.
# Every discard position matches BOTH the `_` wildcard and the `_name` binding
# (`Err(_e)`, `map_err(|_err|`), so renaming the ignored binding can't dodge the
# gate. `or_else(|_` also catches `unwrap_or_else(|_` / `map_or_else(|_` by
# substring. `Err(e) => {}` binds the error and still drops it — same bug.
B_PATTERNS=(
  'let[[:space:]]+_[[:space:]]*='
  'if[[:space:]]+let[[:space:]]+Ok\('
  'while[[:space:]]+let[[:space:]]+Ok\('
  'Err\(_[a-zA-Z0-9_]*\)'
  'Err\(\.\.\)'
  'Err\([a-zA-Z0-9_]+\)[[:space:]]*=>[[:space:]]*(\{[[:space:]]*\}|\(\))'
  'map_err\(\|_[a-zA-Z0-9_]*\|'
  'or_else\(\|_[a-zA-Z0-9_]*\|'
  '\.to_str\(\)\.ok\(\)'
  '\.ok\(\)[[:space:]]*;'
  '\.err\(\)[[:space:]]*;'
)
# U: undefined-behavior / concurrency footguns. `transmute` is handled separately
# below so peregrine-par's single documented lifetime-erasure can be exempted.
U_PATTERNS=(
  '\bstatic[[:space:]]+mut\b' '\bmem::forget\b' 'assume_init\(\)'
  'unwrap_unchecked'
)

echo "== peregrine bad-patterns audit (${#FILES[@]} files) =="
echo "[P] panicking error handling  (STRICT)"; run_section P_TOTAL "${P_PATTERNS[@]}"
echo "[B] silent error swallowing   (STRICT)"; run_section B_TOTAL "${B_PATTERNS[@]}"
# `let _named = fallible()` swallows exactly like `let _ =` while dodging the
# wildcard regex, so it's strict too — EXCEPT the RAII keep-alive idiom
# (`let _g = gpu_guard();`), where the named binding is the point: `let _ =`
# would drop the guard immediately, which would be the actual bug.
LETNAMED="$(scan 'let[[:space:]]+_[a-zA-Z0-9_]+[[:space:]]*=' | grep -vF 'guard()' || true)"
LN_C="$(printf '%s' "$LETNAMED" | grep -c . || true)"
if [ "$LN_C" -gt 0 ]; then
  printf '  %-40s %d\n' 'let _named = (non-guard)' "$LN_C"
  [ "$STRICT" -eq 0 ] && printf '%s\n' "$LETNAMED" | sed 's/^/      /'
  B_TOTAL=$((B_TOTAL + LN_C))
fi
echo "[U] UB / concurrency footguns (STRICT)"; run_section U_TOTAL "${U_PATTERNS[@]}"
# transmute is a strict footgun everywhere EXCEPT peregrine-par, which encapsulates
# one documented lifetime-erasure (borrowed closure → persistent worker pool, made
# sound by a fork-join barrier) behind a safe API — the same tolerance granted to
# `unsafe` in io/cuda/kernels.
TRANSMUTE="$(scan '\btransmute\b' | grep -vE 'crates/peregrine-par/' || true)"
TM_C="$(printf '%s' "$TRANSMUTE" | grep -c . || true)"
if [ "$TM_C" -gt 0 ]; then
  printf '  %-40s %d\n' '\btransmute\b (outside peregrine-par)' "$TM_C"
  [ "$STRICT" -eq 0 ] && printf '%s\n' "$TRANSMUTE" | sed 's/^/      /'
  U_TOTAL=$((U_TOTAL + TM_C))
fi

# I: unsafe outside the crates where it's expected (informational, not a gate).
# peregrine-par encapsulates one pattern: sending a borrowed closure pointer to a
# persistent worker pool behind a fork-join barrier (safe API, `#SAFETY`-documented).
UNSAFE="$(scan '\bunsafe\b' | grep -vE 'crates/peregrine-(io|cuda|kernels|par)/' || true)"
I_TOTAL="$(printf '%s' "$UNSAFE" | grep -c . || true)"
echo "[I] unsafe outside peregrine-io/cuda/kernels (INFO): $I_TOTAL"
[ "$I_TOTAL" -gt 0 ] && [ "$STRICT" -eq 0 ] && printf '%s\n' "$UNSAFE" | sed 's/^/      /'

# L: `let Ok(x) = … else { … }` drops the Err on the floor — the let-else twin
# of the gated `if let Ok(`. The codebase uses it as the sanctioned style for
# best-effort reads (sysfs probes, optional JSON caches), so it is reported for
# review rather than gated; promoting it to strict means ErrorKind-aware
# matches at every site. See docs/BAD_PATTERNS.md.
LETELSE="$(scan '\blet[[:space:]]+Ok\(' | grep -vE '(if|while)[[:space:]]+let[[:space:]]+Ok\(' || true)"
L_TOTAL="$(printf '%s' "$LETELSE" | grep -c . || true)"
echo "[L] let-else Ok(..) err-discards (INFO): $L_TOTAL"
[ "$L_TOTAL" -gt 0 ] && [ "$STRICT" -eq 0 ] && printf '%s\n' "$LETELSE" | sed 's/^/      /'

# C: suppression attributes — hiding a lint instead of fixing it. The workspace
# removed all 14 of these by fixing what they suppressed; this keeps it that way,
# so the claim in docs/BAD_PATTERNS.md is enforced rather than asserted.
# `#[deny(...)]` belongs at the crate root, not sprinkled per-item; `#[ignore]`
# rots a test silently.
C_PATTERNS=(
  '#\[allow\(' '#\[ignore' '#\[deny\('
)
echo "[C] lint suppression            (STRICT)"; run_section C_TOTAL "${C_PATTERNS[@]}"

# Q: Cargo.toml hygiene — a wildcard or unpinned-git dependency makes the build
# non-reproducible, which for a numerics engine means the bit-identity anchors
# can move under you without a commit.
Q_TOTAL=0
QHITS="$(grep -rEn '^[[:space:]]*[a-z0-9_-]+[[:space:]]*=[[:space:]]*"\*"' --include=Cargo.toml . 2>/dev/null | grep -v '/target/' || true)"
QHITS="$QHITS$(grep -rEn 'git[[:space:]]*=[[:space:]]*"[^"]+"' --include=Cargo.toml . 2>/dev/null | grep -v '/target/' | grep -vE '(rev|tag|branch)[[:space:]]*=' || true)"
Q_TOTAL="$(printf '%s' "$QHITS" | grep -c . || true)"
echo "[Q] Cargo.toml hygiene          (STRICT): $Q_TOTAL"
[ "$Q_TOTAL" -gt 0 ] && [ "$STRICT" -eq 0 ] && printf '%s\n' "$QHITS" | sed 's/^/      /'

# R: reachability — code that compiles, passes its own tests, is documented as
# live, and no production path reaches it. Grep cannot see this class; it needs a
# definition/reference pass, so it lives in a companion script. Informational: a
# workspace legitimately exposes API its binaries don't call. The signal is a
# symbol here that a doc or docs/todo.md calls shipped. See docs/BAD_PATTERNS.md [R].
R_TOTAL=0
if command -v python3 >/dev/null 2>&1 && [ -x scripts/audit-reachability.py ]; then
  R_LINE="$(scripts/audit-reachability.py $([ "$STRICT" -eq 0 ] && echo --list))"
  printf '%s\n' "$R_LINE" | head -1
  [ "$STRICT" -eq 0 ] && printf '%s\n' "$R_LINE" | tail -n +2
  R_TOTAL="$(printf '%s' "$R_LINE" | head -1 | grep -oE '[0-9]+$' || echo 0)"
else
  echo "[R] reachability (INFO): skipped (needs python3)"
fi

echo "---"
echo "P=$P_TOTAL (strict)  B=$B_TOTAL (strict)  U=$U_TOTAL (strict)  C=$C_TOTAL (strict)  Q=$Q_TOTAL (strict)  I=$I_TOTAL (info)  L=$L_TOTAL (info)  R=$R_TOTAL (info)"
if [ "$STRICT" -eq 1 ] && { [ "$P_TOTAL" -gt 0 ] || [ "$B_TOTAL" -gt 0 ] || [ "$U_TOTAL" -gt 0 ] || [ "$C_TOTAL" -gt 0 ] || [ "$Q_TOTAL" -gt 0 ]; }; then
  echo "FAIL: strict-section hits present"; exit 1
fi
echo "OK"
