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

mapfile -t FILES < <(find crates -path '*/src/*' -name '*.rs' -type f | sort)

# scan <regex>: matching non-comment lines as "file:line:code"
scan() {
  grep -rEn -- "$1" "${FILES[@]}" 2>/dev/null \
    | grep -vE ':[0-9]+:[[:space:]]*(//|///|\*)'
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
# U: undefined-behavior / concurrency footguns.
U_PATTERNS=('\bstatic[[:space:]]+mut\b' '\btransmute\b' '\bmem::forget\b' 'assume_init\(\)')

echo "== peregrine bad-patterns audit (${#FILES[@]} files) =="
echo "[P] panicking error handling  (STRICT)"; run_section P_TOTAL "${P_PATTERNS[@]}"
echo "[U] UB / concurrency footguns (STRICT)"; run_section U_TOTAL "${U_PATTERNS[@]}"

# I: unsafe outside the crates where it's expected (informational, not a gate).
UNSAFE="$(scan '\bunsafe\b' | grep -vE 'crates/peregrine-(io|cuda|kernels)/' || true)"
I_TOTAL="$(printf '%s' "$UNSAFE" | grep -c . || true)"
echo "[I] unsafe outside peregrine-io/cuda/kernels (INFO): $I_TOTAL"
[ "$I_TOTAL" -gt 0 ] && [ "$STRICT" -eq 0 ] && printf '%s\n' "$UNSAFE" | sed 's/^/      /'

echo "---"
echo "P=$P_TOTAL (strict)  U=$U_TOTAL (strict)  I=$I_TOTAL (info)"
if [ "$STRICT" -eq 1 ] && { [ "$P_TOTAL" -gt 0 ] || [ "$U_TOTAL" -gt 0 ]; }; then
  echo "FAIL: strict-section hits present"; exit 1
fi
echo "OK"
