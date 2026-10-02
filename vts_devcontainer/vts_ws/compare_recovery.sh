#!/usr/bin/env bash
# Frozen-node drift-recovery screening; no ROS or feature extraction.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src/vts_core:$PWD/src/vts_evaluation${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m vts_evaluation.recovery_comparison "$@"
