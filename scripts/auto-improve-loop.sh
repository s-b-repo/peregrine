#!/usr/bin/env bash
# auto-improve-loop.sh — the auto-improve optimization loop entry point.
# Delegates to the unbreakable implementation (read-only, SIG-protected).
# This wrapper exists so that `git add scripts/auto-improve-loop.sh` in the
# unbreakable script's locking step succeeds.
exec "$(dirname "$0")/auto-improve-loop-unbreakable.sh" "$@"
