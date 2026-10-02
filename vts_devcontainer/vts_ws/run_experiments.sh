#!/usr/bin/env bash
# Verified, resumable COLD mapping and evaluation. Run from vts_ws.
set -euo pipefail
exec python3 -m vts_evaluation.run_experiments cold "$@"
