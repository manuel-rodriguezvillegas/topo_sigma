#!/usr/bin/env bash
# Isolated B screening. Reads verified cached descriptors; never runs inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src/vts_core:$PWD/src/vts_evaluation:$PWD/src/vts_players${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m vts_evaluation.frame_recovery_comparison "$@"
