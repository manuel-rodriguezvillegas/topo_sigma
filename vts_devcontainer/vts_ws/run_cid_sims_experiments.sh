#!/usr/bin/env bash
# Verified, resumable CID-SIMS mapping and evaluation. Run from vts_ws.
set -euo pipefail
exec python3 -m vts_evaluation.run_experiments cid-sims "$@"
