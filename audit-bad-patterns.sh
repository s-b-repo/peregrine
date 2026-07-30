#!/usr/bin/env bash
# Delegates to scripts/audit-bad-patterns.sh (the canonical copy). This root
# shim exists because docs/CI referenced both paths; the previous root copy
# computed its repo root incorrectly and silently scanned zero files.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/scripts/audit-bad-patterns.sh" "$@"
