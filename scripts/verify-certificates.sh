#!/usr/bin/env bash
# verify-certificates.sh — run the numerical certificate gate.
#
# Exists so CI, cron and the auto-improve loop have one command to call, the same
# way `audit-bad-patterns.sh` is the one command for the bad-pattern gate. Both
# exit non-zero on failure and both print the reason on stderr, which is the
# whole contract.
#
# `--release` is the default: the certificate residuals are properties of the
# arithmetic, and running the same checks against an unoptimized build would let
# a build-profile difference look like a numerical one. Pass `--debug` to skip it
# when iterating on a new certificate.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

PROFILE="release"
for arg in "$@"; do
    case "$arg" in
        --release) PROFILE="release" ;;
        --debug) PROFILE="dev" ;;
        -h|--help) sed -n '/^# /p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "== peregrine certificate gate (peregrine-verify, $PROFILE) =="
if [ "$PROFILE" = "release" ]; then
    cargo run --release -q -p peregrine-verify -- verify
else
    # A dirty release build would silently check stale binaries.
    cargo run -q -p peregrine-verify -- verify
fi
status=$?
if [ "$status" -eq 0 ]; then
    echo "CERTIFICATES OK"
else
    echo "CERTIFICATES FAILED (exit $status)" >&2
fi
exit "$status"