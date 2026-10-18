#!/usr/bin/env bash
# audit-bad-patterns.sh — the workspace-wide bad-pattern gate.
#
# It catches crash-vectors, UB and silent-failure classes that `cargo clippy`
# does not flag by default, across ALL crates (clippy denies live per-crate and
# can drift; this gate is workspace-wide). Sections A/B/C/J/M/N/T are the gate;
# the rest are informational. The matrix is documented in docs/BAD_PATTERNS.md.
#
# Usage:
#   scripts/audit-bad-patterns.sh                # full report (shows offending lines)
#   scripts/audit-bad-patterns.sh --strict       # exit non-zero on any strict hit (CI)
#   scripts/audit-bad-patterns.sh --section A    # one section (repeatable, aliases accepted)
#   scripts/audit-bad-patterns.sh --files f.txt  # scan only the .rs files listed in f.txt
#   scripts/audit-bad-patterns.sh --max-report 0 # print every hit instead of the first 20
#
# Scope: Rust under crates/*/src, excluding the two vendored crates (see below).
# Comment lines are ignored. A line carrying `// audit-allow: <reason>` is
# dropped from the count; a bare `// audit-allow:` with no reason is itself a
# strict hit (section C), because a silent exemption is indistinguishable from a
# rule nobody enforced.
#
# Letter map (unifur's matrix was merged in; old peregrine letters are aliases):
#   old  new  meaning
#   P -> A      panicking error handling
#   B -> B      silent error swallowing  (the old info-only [L] let-else twin is [B]/info)
#   U -> N      UB / concurrency footguns
#   C -> C      lint suppression and lint-bypass idioms
#   Q -> T      Cargo.toml hygiene
#   I -> E      `unsafe` / numeric casts (informational here)
#   R -> Z      shipped-but-unreachable (informational; needs audit-reachability.py)
set -u -o pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

STRICT=0
MAX_REPORT=20
SHOW_BUDGETS=0
ONLY_SECTIONS=()
FILE_LIST=""

# Per-section debt budgets. `--strict` fails when a section EXCEEDS its budget,
# so a ratcheted section is a real gate on new code while the existing debt stays
# visible and can only shrink. A budget of 0 is an absolute gate.
#
# These numbers are a baseline measured on the full file set (crates/**, tests and
# examples included). They are not targets to raise: a section that needs raising
# means new code added a hit, which is a review conversation, not a bookkeeping
# update. Regenerate with `--budgets` after any deliberate reduction.
#
#   A 285  panicking error handling
#   B 109  swallowed errors
#   C   8  lint suppressions (five in src, the rest in tests/examples)
#   J/M/N/T 0  secrets, injection, UB, Cargo hygiene — absolutely gated
#   D/E/F/G  informational classes with large, mostly intentional hit counts
#            (indexing, numeric casts and `unsafe`, blocking calls, `println!`)
#
# These numbers are UNCHANGED by the peregrine-ctx exemption below, on purpose.
# That exemption removed 413 section-A hits of upstream code from the measured
# set, so the ratchet is now loose rather than tight: measured A=37 (22 in
# examples, 8 in src, 7 in tests), B=67, C=7 against budgets of 285/109/8. The
# old "240 of these are in integration tests and examples" rationale no longer
# describes the count and has been dropped rather than restated.
#
# Loosening a ratchet without re-ratcheting it is its own kind of drift, so it is
# recorded here rather than left implicit: the honest next step is to regenerate
# these numbers with `--budgets` and move them down to the measured counts, in
# its own reviewable change. That was not done in the same change as the
# exemption, because a budget edit and a scope edit should be argued separately.
#
# Three of these have already moved down once, and the ratchet is only honest
# while that direction continues: A 290 → 285 (the five `.unwrap`/`.expect` sites
# clippy's `deny` was already failing on, converted to named errors), F 216 → 215
# and J/style 20 → 19 (`iobench`'s `sh -c 'ulimit -l'` became a direct `getrlimit`).
declare -A BUDGETS=(
    [A]=285 [B]=109 [C]=8 [J]=0 [M]=0 [N]=0 [T]=0
    [D]=478 [E]=2323 [F]=215 [G]=623 [B/info]=33 [C/info]=54 [J/info]=19
    [H]=0 [I]=0 [K]=0 [O]=0 [P]=0 [Z]=65
)
budget_of() { printf '%s' "${BUDGETS[$1]:--1}"; }

usage() { sed -n '/^# Usage:/,/^# Scope:/p' "$0" | sed 's/^# \?//; $d'; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --strict) STRICT=1; shift ;;
        --section)
            [ "$#" -ge 2 ] || { echo "--section needs a section letter" >&2; exit 2; }
            ONLY_SECTIONS+=("$2"); shift 2 ;;
        --files)
            [ "$#" -ge 2 ] || { echo "--files needs a path" >&2; exit 2; }
            FILE_LIST="$2"; shift 2 ;;
        --max-report)
            [ "$#" -ge 2 ] || { echo "--max-report needs a count" >&2; exit 2; }
            MAX_REPORT="$2"; shift 2 ;;
        --budgets) SHOW_BUDGETS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; while [ "$#" -gt 0 ]; do ONLY_SECTIONS+=("$1"); shift; done ;;
        -?*) echo "unknown flag: $1 (try --help)" >&2; exit 2 ;;
        # Legacy callers (scripts/auto-improve-loop-unbreakable.sh) invoke this as
        # `audit-bad-patterns.sh "$ROOT"`. The old script ignored arguments
        # entirely; keep accepting a bare positional so that caller keeps working.
        *) echo "note: ignoring legacy positional argument '$1'" >&2; shift ;;
    esac
done

case "$MAX_REPORT" in
    ''|*[!0-9]*) echo "--max-report needs a non-negative integer" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# file set
# ---------------------------------------------------------------------------
# peregrine-token is vendored third-party code (marcelroed/gigatoken, MIT): it
# keeps upstream style (unwrap/expect in tests + engine internals, unsafe SIMD),
# so it is exempt from the engine crates' panic-free gate. The thin facade in its
# lib.rs is covered by the parity + round-trip test suite instead.
# See docs/BAD_PATTERNS.md.
#
# peregrine-ctx is a second vendored crate, and it was missing from that list.
# It is a verbatim port of subnext/DCR (crate `dcr`, pinned at 609451b): of its
# 34 src files, 32 are byte-identical to upstream, and the two that differ
# (lib.rs, main.rs) differ only by the `dcr::` -> `peregrine_ctx::` prefix
# rename and the CLI binary name. Its 22 tests and 2 of its 3 examples are
# upstream byte-for-byte after that same rename. The audit was therefore
# counting upstream's hits against peregrine's budget — the re-sync of subnext
# moved 165 section-A hits that peregrine never wrote. That is the same position
# peregrine-token is in, so it gets the same exemption.
#
# This is a scope correction, not a waiver, and the boundary is drawn at
# authorship rather than at the directory. Three files in peregrine-ctx are
# peregrine's own work and stay inside the gate:
#
#   src/lib.rs            module list, re-exports, VERSION, port doc comment
#   src/main.rs           the CLI binary rename
#   examples/ctxbench.rs  a peregrine-only benchmark harness, no upstream peer
#
# Those are the files peregrine can be asked to change; exempting them would
# exempt peregrine's own code. Note the exemption is deliberately *not*
# "whatever has no upstream counterpart today": interop.rs, sessions.rs, skills.rs
# and fit.rs are byte-identical to upstream, and tests/*.rs are rename-only, so
# all of them are exempt here despite being recent additions. A future file with
# no upstream peer is NOT automatically exempt — add it to the list above and
# argue it.
#
# scripts/audit-reachability.py implements section [Z] over the same file set and
# carries the same exclusion, so [Z] is not counting upstream while every other
# section is.
if [ -n "$FILE_LIST" ]; then
    if [ ! -f "$FILE_LIST" ]; then
        echo "file list $FILE_LIST does not exist" >&2
        exit 2
    fi
    FILES=()
    while IFS= read -r path || [ -n "$path" ]; do
        [ -z "$path" ] && continue
        [ -f "$path" ] || { echo "file list entry does not exist: $path" >&2; exit 2; }
        FILES+=("$path")
    done < "$FILE_LIST"
else
    # `find`'s default -print applies to the whole expression, so this reads as
    # "(not vendored) OR (one of the three authored files)".
    mapfile -t FILES < <(find crates -name '*.rs' -type f \
        ! -path 'crates/peregrine-token/*' \
        ! -path 'crates/peregrine-ctx/*' \
        -o -path 'crates/peregrine-ctx/src/lib.rs' \
        -o -path 'crates/peregrine-ctx/src/main.rs' \
        -o -path 'crates/peregrine-ctx/examples/ctxbench.rs' | sort)
fi
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "no Rust files to scan (is the file list empty?)" >&2
    exit 2
fi

want() {  # want <letter...>: true when no --section filter, or this letter was asked for
    [ "${#ONLY_SECTIONS[@]}" -eq 0 ] && return 0
    local asked alias
    for asked in "${ONLY_SECTIONS[@]}"; do
        # old letters keep working
        case "$asked" in
            P|I2) alias=A ;;
            U) alias=N ;;
            Q) alias=T ;;
            I) alias=E ;;
            R) alias=Z ;;
            L) alias=B ;;
            *)  alias="$asked" ;;
        esac
        [ "$alias" = "$1" ] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
# scan <regex>: matching non-comment lines as "file:line:code", with reasoned
# `audit-allow` suppressions removed. `EXCLUDE` (optional) drops whole paths,
# which is how the crates where a construct is *expected* are exempted.
EXCLUDE=""
scan() {
    local hits
    hits="$(grep -En -- "$1" "${FILES[@]}" 2>/dev/null \
        | grep -vE ':[0-9]+:[[:space:]]*(//|///|\*)' \
        | grep -vE 'audit-allow:[[:space:]]*[^[:space:]]')"
    if [ -n "$EXCLUDE" ]; then
        hits="$(printf '%s\n' "$hits" | grep -vE -- "$EXCLUDE")"
    fi
    printf '%s' "$hits"
}

# show <label> <hits>: one row, with the offending lines capped at MAX_REPORT.
show() {
    local label="$1" hits="$2" n
    [ -n "$hits" ] || return 0
    n="$(printf '%s\n' "$hits" | grep -c . || true)"
    printf '  %-48s %5d\n' "$label" "$n"
    if [ "$MAX_REPORT" -eq 0 ]; then
        printf '%s\n' "$hits" | sed 's/^/      /'
    else
        printf '%s\n' "$hits" | head -n "$MAX_REPORT" | sed 's/^/      /'
        [ "$n" -gt "$MAX_REPORT" ] && \
            printf '      ... %d more (--max-report %d for all)\n' "$((n - MAX_REPORT))" "$MAX_REPORT"
    fi
    return 0
}

# count <regex>: number of hits, honouring EXCLUDE.
count() { local h; h="$(scan "$1")"; printf '%s' "$h" | grep -c . || true; }

# section <out-var> <label> <regex...>: print the rows, set <out-var> to the total.
section() {
    local __var="$1" label="$2"; shift 2
    local total=0 re hits c
    for re in "$@"; do
        hits="$(scan "$re")"
        [ -n "$hits" ] || continue
        show "$re" "$hits"
        c="$(printf '%s\n' "$hits" | grep -c . || true)"
        total=$((total + c))
    done
    printf -v "$__var" '%d' "$total"
    return 0
}

A_TOTAL=0; B_TOTAL=0; B_INFO=0; C_TOTAL=0; C_INFO=0; D_TOTAL=0; E_TOTAL=0
E_INFO=0; F_TOTAL=0; G_TOTAL=0; H_TOTAL=0; I_TOTAL=0; J_TOTAL=0; J_INFO=0
K_TOTAL=0; M_TOTAL=0; N_TOTAL=0; O_TOTAL=0; P_TOTAL=0; T_TOTAL=0; Z_TOTAL=0
OVER_BUDGET=0
OVER_LIST=""

# gated <key> <count>: compare against the ratchet budget and record any section
# that is over. Nothing is added up into one grand total on purpose — a single
# number across sections hides which section regressed, which is the only thing
# the gate is for.
gated() {
    local key="$1" count="$2" budget
    budget="$(budget_of "$key")"
    if [ "$budget" -lt 0 ]; then
        return 0
    fi
    if [ "$count" -gt "$budget" ]; then
        OVER_BUDGET=$((OVER_BUDGET + 1))
        OVER_LIST="$OVER_LIST $key:$count/$budget"
        printf '  %-48s %5d  OVER BUDGET %d\n' "[$key] ratchet" "$count" "$budget"
    elif [ "$SHOW_BUDGETS" -eq 1 ]; then
        printf '  %-48s %5d  budget %d, %d to spend\n' "[$key] ratchet" "$count" "$budget" "$((budget - count))"
    fi
}

# crates where `unsafe` is the point of the crate, not a smell
UNSAFE_CRATES='crates/peregrine-(io|cuda|kernels|par)/'

echo "== peregrine bad-patterns audit (${#FILES[@]} files) =="

# ---------------------------------------------------------------------------
# [A] panicking error handling — STRICT
# ---------------------------------------------------------------------------
# A panic aborts the process: the release profile is panic = "abort", so an
# unwrap inside the serve loop is not a caught error, it is a dead server and a
# dropped in-flight batch. Tests are not exempt (clippy.toml says so too).
# The compound shapes from the unifur matrix (.first().unwrap(), .lock().unwrap(),
# .parse().unwrap(), ...) are all subsumed by `\.unwrap\(\)`; only the base
# patterns are listed so the counts stay honest.
if want A; then
    echo "[A] panicking error handling    (STRICT)"
    section A_TOTAL 'A' \
        '\.unwrap\(\)' '\.expect\(' '\.unwrap_or_default\(\)' \
        '\.unwrap_err\(\)' '\.expect_err\(' \
        'panic!\(' 'unreachable!\(' 'todo!\(' 'unimplemented!\('
    gated A "$A_TOTAL"
fi

# ---------------------------------------------------------------------------
# [B] silent error swallowing — STRICT (+ one informational twin)
# ---------------------------------------------------------------------------
# A discarded Result hides a real failure. Advisory operations must handle the
# Err explicitly (peregrine_io::note_advisory_err, COLI_DEBUG=1 surfaces them);
# control-flow errors must match concrete variants
# (`Err(VarError::NotPresent)`, `TryRecvError::Empty`) — never a bare wildcard.
# Every discard position matches BOTH the `_` wildcard and the `_name` binding
# (`Err(_e)`, `map_err(|_err|`) so renaming the ignored binding cannot dodge the
# gate. `if let Err(_)` is the *discarding* form only: `if let Err(e) = ..` that
# then does something with `e` is handling the error, which is the point.
if want B; then
    echo "[B] silent error swallowing     (STRICT)"
    section B_TOTAL 'B' \
        'let[[:space:]]+_[[:space:]]*=' \
        'if[[:space:]]+let[[:space:]]+Ok\(' \
        'while[[:space:]]+let[[:space:]]+Ok\(' \
        'if[[:space:]]+let[[:space:]]+Err\(_' \
        'Err\(_\)' \
        'Err\(_[a-zA-Z0-9_]*\)' \
        'Err\(\.\.\)' \
        'Err\([a-zA-Z0-9_]+\)[[:space:]]*=>[[:space:]]*(\{[[:space:]]*\}|\(\))' \
        'map_err\(\|_[a-zA-Z0-9_]*\|' \
        'or_else\(\|_[a-zA-Z0-9_]*\|' \
        'unwrap_or_else\(\|_[a-zA-Z0-9_]*\|' \
        '\.to_str\(\)\.ok\(\)' \
        '\.ok\(\)[[:space:]]*;' \
        '\.err\(\)[[:space:]]*;' \
        '\.await\.ok\(\)'

    # `let _named = fallible()` swallows exactly like `let _ =` while dodging the
    # wildcard regex, so it is strict too — EXCEPT the RAII keep-alive idiom
    # (`let _g = gpu_guard();`), where the named binding is the point: `let _ =`
    # would drop the guard immediately, which would be the actual bug.
    LN="$(scan 'let[[:space:]]+_[a-zA-Z0-9_]+[[:space:]]*=' | grep -vF 'guard()' || true)"
    LN_C="$(printf '%s' "$LN" | grep -c . || true)"
    if [ "$LN_C" -gt 0 ]; then
        show 'let _named = (non-guard)' "$LN"
        B_TOTAL=$((B_TOTAL + LN_C))
    fi
    gated B "$B_TOTAL"

    # `let Ok(x) = … else { … }` drops the Err on the floor — the let-else twin
    # of the strict `if let Ok(`. The codebase uses it as the sanctioned style
    # for best-effort reads (sysfs probes, optional JSON caches), so it is
    # reported for review rather than gated; promoting it to strict means
    # ErrorKind-aware matches at every site. See docs/BAD_PATTERNS.md.
    LE="$(scan '\blet[[:space:]]+Ok\(' | grep -vE '(if|while)[[:space:]]+let[[:space:]]+Ok\(' || true)"
    B_INFO="$(printf '%s' "$LE" | grep -c . || true)"
    [ "$B_INFO" -gt 0 ] && show 'let-else Ok(..) err-discards (INFO)' "$LE"
    gated "B/info" "$B_INFO"
fi

# ---------------------------------------------------------------------------
# [C] lint suppression — STRICT (+ informational bypass idioms)
# ---------------------------------------------------------------------------
# The workspace removed all 14 `#[allow]`s by fixing what they suppressed; this
# keeps it that way, so the claim in docs/BAD_PATTERNS.md is enforced rather than
# asserted. `#[deny(...)]` belongs at the crate root, not sprinkled per-item,
# and `#[ignore]` rots a test silently.
if want C; then
    echo "[C] lint suppression            (STRICT)"
    section C_TOTAL 'C' '#\[allow\(' '#\[deny\(' '#\[ignore'
    # An `audit-allow:` with no reason is comment-shaped lint suppression.
    AN="$(scan 'audit-allow:' | grep -vE 'audit-allow:[[:space:]]*[^[:space:]]' || true)"
    AN_C="$(printf '%s' "$AN" | grep -c . || true)"
    if [ "$AN_C" -gt 0 ]; then
        show '// audit-allow: without a reason' "$AN"
        C_TOTAL=$((C_TOTAL + AN_C))
    fi
    gated C "$((C_TOTAL + AN_C))"

    # B10 from the unifur matrix: an underscore-prefixed parameter hides a value
    # from the unused-parameter lint. Common in this engine (phantom device
    # handles, unit-only args), so informational.
    UP="$(scan 'fn\s+\w+\([^)]*\b_\w+[[:space:]]*:')"
    C_INFO="$(printf '%s' "$UP" | grep -c . || true)"
    [ "$C_INFO" -gt 0 ] && show 'underscore-prefixed fn parameter (INFO)' "$UP"
    gated "C/info" "$C_INFO"
fi

# ---------------------------------------------------------------------------
# [D] panic vectors (index/slice) — INFO
# ---------------------------------------------------------------------------
# A kernel engine indexes deliberately — a literal index in an inner loop is
# the house style — so this is review-only: it points at the slices that can
# panic on a short buffer, which is where I/O-driven length bugs surface.
if want D; then
    echo "[D] panic vectors (index/slice) (INFO)"
    section D_TOTAL 'D' \
        '\b[a-zA-Z_]\w*\[[0-9]+\][^=]' \
        '&[a-zA-Z_]\w*\[\.\.\w+\]' \
        '&[a-zA-Z_]\w*\[\w+\.\.\]' \
        '&[a-zA-Z_]\w*\[\w+\.\.\w+\]' \
        '\.split_at\(' '\.chars\(\)\.nth\('
    gated D "$D_TOTAL"
fi

# ---------------------------------------------------------------------------
# [E] numeric / unsafe — INFO (unsafe outside the expected crates is the point)
# ---------------------------------------------------------------------------
# Numeric `as` casts are intentional here (the kernels mirror C ports), and
# `unsafe` is EXPECTED in peregrine-io (io_uring + aligned alloc), peregrine-cuda
# (FFI), peregrine-kernels (SIMD intrinsics) and peregrine-par (one documented
# lifetime-erasure). Both are reported, neither gated. See [N] for the UB class.
if want E; then
    echo "[E] numeric / unsafe            (INFO)"
    section E_TOTAL 'E' \
        '\bas\s+u8\b' '\bas\s+i8\b' \
        '\bas\s+u16\b' '\bas\s+i16\b' \
        '\bas\s+u32\b' '\bas\s+i32\b' \
        '\bas\s+u64\b' '\bas\s+i64\b' \
        '\bas\s+usize\b' '\bas\s+isize\b' \
        '\bas\s+f32\b' '\bas\s+f64\b'
    EXCLUDE="$UNSAFE_CRATES"
    section E_INFO 'E' \
        '\bunsafe\b' \
        '\bextern\s+"C"' \
        '\bas\s+\*const\b' '\bas\s+\*mut\b' \
        '\bmem::zeroed\(' '\bmem::uninitialized\(' '\bmaybe_uninit\('
    EXCLUDE=""
    gated E "$((E_TOTAL + E_INFO))"
fi

# ---------------------------------------------------------------------------
# [F] blocking calls — INFO
# ---------------------------------------------------------------------------
# The reactor exists to keep blocking syscalls off the hot path, so a blocking
# std::fs call inside a request path is a finding; one in a bench or a setup
# path is not. Grep cannot tell the two apart, so it reports.
if want F; then
    echo "[F] blocking / process calls    (INFO)"
    section F_TOTAL 'F' \
        'std::thread::sleep' \
        'std::process::Command' \
        'std::fs::File' \
        'std::fs::read\b' '\.read_to_string\(' \
        'std::fs::write\b' '\.write_all\(' \
        'std::net::TcpStream' 'std::net::UdpSocket' \
        'std::io::stdin'
    gated F "$F_TOTAL"
fi

# ---------------------------------------------------------------------------
# [G] logging hygiene — INFO
# ---------------------------------------------------------------------------
if want G; then
    echo "[G] logging hygiene             (INFO)"
    section G_TOTAL 'G' \
        '\bdbg!\(' \
        'format!\(.*"\{:\?\}".*\b[eE]rr\b' \
        '\.context\(""\)' \
        'println!\('
    gated G "$G_TOTAL"
fi

# ---------------------------------------------------------------------------
# [H] HTTP layer — INFO
# ---------------------------------------------------------------------------
# A client built per request loses connection reuse; an unchecked status is a
# silent 500. This engine speaks its own serve protocol, so zero hits is the
# expected result and the rule stays as the tripwire.
if want H; then
    echo "[H] HTTP layer                  (INFO)"
    section H_TOTAL 'H' \
        'reqwest::Client::new\(\)' \
        'reqwest::Client::builder\(\)' \
        '\.send\(\)\.await\?[^.]' \
        '\.text\(\)\.await\?[^.]'
    gated H "$H_TOTAL"
fi

# ---------------------------------------------------------------------------
# [I] iterator glitches — INFO
# ---------------------------------------------------------------------------
if want I; then
    echo "[I] iterator glitches           (INFO)"
    section I_TOTAL 'I' \
        '\.collect::<Result<Vec<_>,\s*_>>\(\)\.unwrap' \
        '\.zip\([^)]*\)\.unwrap' \
        '\.iter\(\)\.find\([^)]*\)\.expect\('
    gated I "$I_TOTAL"
fi

# ---------------------------------------------------------------------------
# [J] secrets (STRICT) + style (INFO)
# ---------------------------------------------------------------------------
# Secrets never belong in a tree that is pushed; the shapes below are the ones
# that actually leak. Style is the unifur matrix's style/secrets section, minus
# the rules that are dead here (comment-only lines are already stripped, so a
# `TODO`/`FIXME` regex could never fire).
if want J; then
    echo "[J] secrets                     (STRICT)"
    section J_TOTAL 'J' \
        'Bearer[[:space:]]+[A-Za-z0-9_.-]{40,}' \
        'sk-[A-Za-z0-9]{20,}' \
        'AKIA[A-Z0-9]{16}' \
        'ghp_[A-Za-z0-9]{36}' \
        'xox[baprs]-[A-Za-z0-9-]{10,}' \
        'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY' \
        '"(admin|root|password|secret)"[[:space:]]*,[[:space:]]*"(admin|root|password|secret)"'
    gated J "$J_TOTAL"

    echo "[J] style                       (INFO)"
    section J_INFO 'J' \
        '== ""' \
        '\.len\(\) == 0' \
        '\.len\(\) > 0' \
        '\.clone\(\)\s*\.clone' \
        '\.to_string\(\)\s*\.to_string' \
        'String::from\(format!' \
        '&\*format!\(' \
        'Box<dyn' \
        'use\s+\w+\s+as\s+_;'
    gated "J/info" "$J_INFO"
fi

# ---------------------------------------------------------------------------
# [K] crypto — INFO
# ---------------------------------------------------------------------------
# A non-CSPRNG where reproducibility is not the intent, or MD5/SHA-1/ECB where
# the digest is load-bearing (the merkle tree is), is a finding. Reproducible
# test RNGs are fine, so this is informational.
if want K; then
    echo "[K] crypto                      (INFO)"
    section K_TOTAL 'K' \
        '\bmd5::' '\bsha1::' \
        '\bdes::' '\b3des\b' \
        '\bRC4\b' \
        'aes-128-ecb|aes_128_ecb' \
        'rand::thread_rng\(\)' \
        'rand::random\('
    gated K "$K_TOTAL"
fi

# ---------------------------------------------------------------------------
# [M] injection — STRICT
# ---------------------------------------------------------------------------
if want M; then
    echo "[M] SQL / command injection     (STRICT)"
    section M_TOTAL 'M' \
        'format!\("(SELECT|INSERT|UPDATE|DELETE)[^"]*\{' \
        'std::process::Command::new\("/bin/sh"\)' \
        'std::process::Command::new\("sh"\)' \
        'Command::new\([^)]*\)[[:space:]]*\.arg\("-c"\)'
    gated M "$M_TOTAL"
fi

# ---------------------------------------------------------------------------
# [N] UB / concurrency footguns — STRICT
# ---------------------------------------------------------------------------
# `transmute` is a strict footgun everywhere EXCEPT peregrine-par, which
# encapsulates one documented lifetime-erasure (borrowed closure → persistent
# worker pool, made sound by a fork-join barrier) behind a safe API — the same
# tolerance granted to `unsafe` in io/cuda/kernels. `mem::zeroed` is reported
# under [E] instead: every hit is a zero-initialised POD libc struct.
if want N; then
    echo "[N] UB / concurrency footguns    (STRICT)"
    section N_TOTAL 'N' \
        '\bstatic[[:space:]]+mut\b' \
        '\bmem::forget\b' \
        'assume_init\(\)' \
        'unwrap_unchecked' \
        'MaybeUninit::uninit\('
    TM="$(scan '\btransmute\b' | grep -vE "$UNSAFE_CRATES" || true)"
    TM_C="$(printf '%s' "$TM" | grep -c . || true)"
    if [ "$TM_C" -gt 0 ]; then
        show '\btransmute\b (outside peregrine-par)' "$TM"
        N_TOTAL=$((N_TOTAL + TM_C))
    fi
    gated N "$N_TOTAL"
fi

# ---------------------------------------------------------------------------
# [O] performance footguns — INFO
# ---------------------------------------------------------------------------
if want O; then
    echo "[O] performance                 (INFO)"
    section O_TOTAL 'O' \
        '\.iter\(\)\.count\(\)' \
        '\.collect::<\(\)>\(\)' \
        '\.iter\(\)\.map\(\|\w+\|\s*\w+\.clone\(\)\)\s*\.collect' \
        '\.to_string\(\)\.as_str\(\)' \
        'Vec::with_capacity\(0\)' 'String::with_capacity\(0\)' \
        'Box::new\(.*Box::new' \
        '\.clone\(\)\.iter\(\)\.enumerate\(\)'
    gated O "$O_TOTAL"
fi

# ---------------------------------------------------------------------------
# [P] API hygiene — INFO
# ---------------------------------------------------------------------------
if want P; then
    echo "[P] API hygiene                  (INFO)"
    section P_TOTAL 'P' \
        '\bpub\s+const\s+\w+\s*:\s*&str\s*=\s*"http' \
        'pub\s+struct\s+\w+\s*\{[^}]*\b(secret|password|api_key|private_key)\b'
    gated P "$P_TOTAL"
fi

# ---------------------------------------------------------------------------
# [T] Cargo.toml hygiene — STRICT
# ---------------------------------------------------------------------------
# A wildcard or unpinned-git dependency makes the build non-reproducible, which
# for a numerics engine means the bit-identity anchors can move under you
# without a commit.
if want T; then
    T_TOTAL=0
    QHITS="$(grep -rEn '^[[:space:]]*[a-z0-9_-]+[[:space:]]*=[[:space:]]*"\*"' --include=Cargo.toml . 2>/dev/null | grep -v '/target/' || true)"
    QHITS="$QHITS$(printf '\n')$(grep -rEn 'git[[:space:]]*=[[:space:]]*"[^"]+"' --include=Cargo.toml . 2>/dev/null | grep -v '/target/' | grep -vE '(rev|tag|branch)[[:space:]]*=' || true)"
    show 'unpinned/wildcard dependency' "$QHITS"
    T_TOTAL="$(printf '%s' "$QHITS" | grep -c . || true)"
    gated T "$T_TOTAL"
fi

# ---------------------------------------------------------------------------
# [Z] reachability — INFO
# ---------------------------------------------------------------------------
# Code that compiles, passes its own tests, is documented as live, and no
# production path reaches it. Grep cannot see this class; it needs a
# definition/reference pass, so it lives in a companion script. Informational: a
# workspace legitimately exposes API its binaries don't call. The signal is a
# symbol here that a doc or docs/todo.md calls shipped. See docs/BAD_PATTERNS.md [Z].
if want Z; then
    Z_TOTAL=0
    if [ -n "$FILE_LIST" ]; then
        # The companion script resolves reachability over the whole workspace; a
        # subset scan cannot make a statement about it.
        echo "[Z] reachability (INFO): skipped (--files narrows to a subset)"
    elif command -v python3 >/dev/null 2>&1 && [ -x scripts/audit-reachability.py ]; then
        R_LINE="$(scripts/audit-reachability.py $([ "$STRICT" -eq 0 ] && echo --list))"
        printf '%s\n' "$R_LINE" | head -1
        [ "$STRICT" -eq 0 ] && printf '%s\n' "$R_LINE" | tail -n +2
        Z_TOTAL="$(printf '%s\n' "$R_LINE" | head -1 | grep -oE '[0-9]+$' || echo 0)"
    else
        echo "[Z] reachability (INFO): skipped (needs python3)"
    fi
    gated Z "$Z_TOTAL"
fi

echo "---"
# Both spellings, so the letters this audit used before the unifur matrix was
# merged are never more than a glance away: P=A, U=N, Q=T, I=E, R=Z, L=B/info.
echo "gated:  A/P=$A_TOTAL B=$B_TOTAL C=$C_TOTAL J=$J_TOTAL M=$M_TOTAL N/U=$N_TOTAL T/Q=$T_TOTAL"
echo "ratchet: D=$D_TOTAL E/I=$((E_TOTAL + E_INFO)) F=$F_TOTAL G=$G_TOTAL H=$H_TOTAL I=$I_TOTAL"
echo "         J/style=$J_INFO K=$K_TOTAL O=$O_TOTAL P=$P_TOTAL R/Z=$Z_TOTAL"
echo "         B/let-else=$B_INFO  C/bypass=$C_INFO"
if [ "$OVER_BUDGET" -gt 0 ]; then
    echo "OVER BUDGET ($OVER_BUDGET):$OVER_LIST"
else
    echo "every section within its ratchet budget"
fi
if [ "$STRICT" -eq 1 ] && [ "$OVER_BUDGET" -gt 0 ]; then
    echo "FAIL: a section is over its ratchet budget"
    exit 1
fi
echo "OK"